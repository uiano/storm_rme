import abc
import itertools as it
import os
import pickle
import sys
from collections import OrderedDict
from typing import Dict

from IPython.core.debugger import set_trace
from scipy.stats import norm

from ..gsim.gfigure import GFigure
from ..map_generators.map import Map, RectangularGrid
from ..measurement_generation.sampler import InterpolationSampler
from ..simulators.map_estimation_simulator import MapEstimationSimulator
from ..utilities import dbm_to_watt, empty_array, natural_to_dB, np, watt_to_dbm


class MapEstimator():

    name_on_figs = "Unnamed"

    freq_basis = None
    estimation_fun_type = ''  # 'g2g', 's2s'

    def __init__(
            self,
            grid: RectangularGrid | None = None,
            min_service_power=None,  # Minimum rx power to consider that there service at a point
            name_on_figs=None,
            interpolation_method=None,
            d_train_params=None):
        """

        Args:

        - `grid`: Optional but possibly required for certain functionality. If
          provided, its dimensions must agree with `num_points_x` and
          `num_points_y` below. 

        - `d_train_params`: OrderedDict whose keys are parameter names and whose values
          are lists of possible values for these parameters. This dict is used
          by self.train.

         """

        self.min_service_power = min_service_power
        self.grid = grid
        self._m_all_measurements = None
        self._m_all_measurement_loc = None
        if name_on_figs is not None:
            self.name_on_figs = name_on_figs
        self.interpolation_method = interpolation_method if interpolation_method is not None else 'avg_nearest'
        self.d_train_params = d_train_params

    def save(self, file_name):
        raise NotImplementedError

    @staticmethod
    def load(file_name):
        """
        Returns the object saved in file_name.
        """
        raise NotImplementedError

    def estimate(self, map_obs: Map, test_loc=None) -> Dict[str, Map]:
        """

        If `map_obs.return_natural` is True, the estimation is done in natural
        units. The output maps have the same same value of return_natural. 


        Args:

            `map_obs`: object of class Map.
            
            `test_loc`: can be: 
            
                - None: in this case, all the grid points will be
                  considered as test_loc. 
            
                - a num_test_loc x 3 matrix with the test
                  locations.
                
                - a RectangularGrid object of the same shape as map_obs.grid. In
                  this case, the map is only estimated at the enabled grid
                  points of the grid.

        Returns:

            `d_map_est`: dict whose values are objects of class Map.
        """

        def adapt_out_units(m):
            if map_obs.return_natural:
                m = np.where(m > 0, m, 1e-24)
                return natural_to_dB(m)
            else:
                return m

        if self.estimation_fun_type != 's2s' and isinstance(
                test_loc, RectangularGrid):
            raise NotImplementedError()

        if self.estimation_fun_type == 's2s':
            # Invoke self.estimate_s2s by taking the data from map_obs.

            if test_loc is None:
                m_test_locs_sf = map_obs.grid.all_grid_points_in_matrix_form
            elif isinstance(test_loc, RectangularGrid):
                # If test_loc is a grid, then we only estimate at the enabled points
                m_test_locs_sf = test_loc.list_pts()
            else:
                m_test_locs_sf = test_loc

            d_map_estimate = self.estimate_s2s(
                measurement_loc=map_obs.m_meas_locs_sf,
                measurements=map_obs.m_meas_sf,
                test_loc=m_test_locs_sf)

            # With the output, instantiate one map per key to form `d_out`
            d_out = {
                key:
                Map(grid=map_obs.grid,
                    m_meas_locs_sf=m_test_locs_sf,
                    m_meas_sf=adapt_out_units(d_map_estimate[key]),
                    return_natural=map_obs.return_natural)
                for key in d_map_estimate.keys()
                if d_map_estimate[key] is not None
            }

        elif self.estimation_fun_type == 'g2g':
            # Invoke self.estimate_g2g by taking the data from map_obs.
            # With the output, instantiate one map per key to form `d_map_est`
            d_map_estimate = self.estimate_g2g(measurement_loc=map_obs.mask,
                                               measurements=map_obs.t_meas_gf)
            d_out = dict()
            for key in d_map_estimate.keys():
                if d_map_estimate[key] is not None:
                    map_out = Map(grid=map_obs.grid,
                                  t_meas_gf=adapt_out_units(
                                      d_map_estimate[key]),
                                  return_natural=map_obs.return_natural)
                    if test_loc is not None:
                        # Create a map in standard form
                        m_meas_sf = map_out.eval(test_loc)
                        map_out = Map(m_meas_locs_sf=test_loc,
                                      m_meas_sf=adapt_out_units(m_meas_sf),
                                      return_natural=map_obs.return_natural)
                    d_out[key] = map_out
        elif self.estimation_fun_type == 's2g':
            # Invoke self.estimate_s2g by taking the data from map_obs.
            # With the output, instantiate one map per key to form `d_map_est`
            d_map_estimate = self.estimate_s2g(
                measurement_loc=map_obs.m_meas_locs_sf,
                measurements=map_obs.m_meas_sf)
            d_out = dict()
            for key in d_map_estimate.keys():
                if d_map_estimate[key] is not None:
                    map_out = Map(grid=map_obs.grid,
                                  t_meas_gf=adapt_out_units(
                                      d_map_estimate[key]),
                                  return_natural=map_obs.return_natural)
                    if test_loc is not None:
                        # Create a map in standard form
                        m_meas_sf = map_out.eval(test_loc)
                        map_out = Map(m_meas_locs_sf=test_loc,
                                      m_meas_sf=adapt_out_units(m_meas_sf),
                                      return_natural=map_obs.return_natural)
                    d_out[key] = map_out
        else:
            raise NotImplementedError

        return d_out

    def estimate_s2s(self,
                     measurement_loc=None,
                     measurements=None,
                     building_meta_data=None,
                     test_loc=None):
        """
        Args:
            - `measurement_locs` : num_measurements x 3 matrix with the
                   3D locations of the measurements.
            - `measurements` : num_measurements x num_sources matrix
                   with the measurements at each channel.
            - `m_test_loc`: a num_test_loc x 3 matrix with the
                    locations where the map estimate will be evaluated.

        Returns:
            `d_map_estimate`: dictionary whose fields are:

           - "t_power_map_estimate" :  num_test_loc x num_sources matrix with
                estimated power of each channel at test_loc.
           - Other optional keys.
        """
        raise NotImplementedError

    def estimate_s2g(self, measurement_loc=None, measurements=None):
        """
        Args:
            - `measurement_locs` : num_measurements x 3 matrix with the
                   3D locations of the measurements.
            - `measurements` : num_measurements x num_sources matrix
                   with the measurements at each channel.

        Returns:
            `d_map_estimate`: dictionary whose fields are:

           - "t_power_map_estimate" : tensor whose (i,j,k)-th entry is the
           estimated power of the i-th source at grid point (j,k).
           - Other optional keys.
        """
        raise NotImplementedError

    def estimate_g2g(
            self,
            measurement_loc=None,
            measurements=None,
            building_meta_data=None,  # deprecated
    ):
        """
        Args:
            - `measurements`: num_sources x self.grid.num_points_y x
              self.grid.num_points_x

            - `measurement_locs`: self.grid.num_points_y x
              self.grid.num_points_x matrix. The [i,j]-th entry is 1 if
              measurements[:,i,j] contains an observation and 0 otherwise.

        Returns:
            `d_map_estimate`: dictionary whose fields are:

           - "t_power_map_estimate" :  tensor whose (i,j,k)-th entry is the
           estimated power of the i-th channel at grid point (j,k). 
           - Other optional keys.
        """
        raise NotImplementedError

    def estimate_from_tensors(self,
                              measurement_locs=None,
                              measurements=None,
                              building_meta_data=None,
                              test_locs=None):
        """
        This method first stores `measurement_locs` and `measurements` in a
        buffer together with previous measurement_locs and measurements provided
        through this method or through method store_measurement since the
        instantiation of the object or the last call to the reset method.

        Then the estimation method implemented by the subclass is invoked over
        the stored measurements and measurement locations.

        Args:

            if `measurements` is in the form of
                num_sources x num_points_y x num_points_x [input in grid form]
                 
                - `building_meta_data`: num_points_y x num_points_x 
                
                - `measurement_locs`: num_points_y x num_points_x is a sampling mask

            else: [input in standard form]
                - `measurements` : num_measurements x num_sources matrix
                   with the measurements at each channel.
                - `measurement_locs` : num_measurements x 3 matrix with the
                   3D locations of the measurements.
                - `building_meta_data`: num_points x 3 matrix with the 3D
                  locations of the buildings

            `test_locs`: can be None or a num_test_locs x 3 matrix with the
            locations where the map estimate will be evaluated.

        Returns:
           `d_map_estimate`: dictionary whose fields are:

           o  If `test_locs` is None, the shape of these items is num_sources x
           num_points_y x num_points_x.

                - "t_power_map_estimate" :  tensor whose (i,j,k)-th entry is the
                estimated power of the i-th channel at grid point (j,k). 
                
                - "t_power_map_norm_variance" : Contains the variance a
                posteriori normalized to the interval (0,1). 
                
                - "t_service_map_estimate": [only if self.min_service_power is
                not None]. Each entry contains the posterior probability that
                the power is above `self.min_service_power`. 
                
                - "t_service_map_entropy": [only if self.min_service_power is
                not None] Entropy of the service indicator.

          
           o   If `test_locs` is not None, `d_map_estimate` contains the same
           fields but they are matrices of shape num_test_locs x num_sources.

            """

        # 1. store the data
        measurement_locs, measurements = self.store_measurement(
            measurement_locs, measurements)

        # 2. invoke _estimate_no_storage on the stored data
        d_map_estimate = self._estimate_no_storage(
            measurement_loc=measurement_locs,
            measurements=measurements,
            building_meta_data=building_meta_data,
            test_loc=test_locs)

        return d_map_estimate

    def store_measurement(self, measurement_loc, measurements):
        """
            These method stores the provided measurement and measurement locations
            together with previous measurements and measurement locations.
        Args:
            if measurements.ndim is 3 then
                -`measurement`: is in the form of
                    num_sources x num_points_y x num_points_x
                -`measurement_loc`: is a mask in the form of
                    num_points_y x num_points_x

            else:
                -`measurement`: is in the form of num_measurements x num_sources
                -`measurement_loc`: is a mask in the form of num_measurements x 3
        Returns:
            if measurements.ndim is 3 then
                -`measurement`: is in the form of
                    num_sources x num_points_y x num_points_x
                -`measurement_loc`: is a mask in the form of
                    num_points_y x num_points_x

            else:
                -`measurement`: is in the form of num_measurements x num_sources
                -`measurement_loc`: is a mask in the form of num_measurements x 3

        """
        b_input_gf = (measurements.ndim == 3)
        if b_input_gf:
            # input data are in grid format
            if self._m_all_measurements is None:
                self._m_all_measurements = measurements
                self._m_all_measurement_loc = measurement_loc
            else:
                self._m_all_measurement_loc += measurement_loc
                self._m_all_measurements += measurements

            measurement_loc = np.minimum(self._m_all_measurement_loc, 1)
            # average the measurements at each grid point
            measurements = self._m_all_measurements / np.maximum(
                self._m_all_measurement_loc, 1)

        else:
            # input data are in standard format
            if self._m_all_measurements is None:
                self._m_all_measurements = measurements
                self._m_all_measurement_loc = measurement_loc
            else:
                self._m_all_measurements = np.vstack(
                    (self._m_all_measurements, measurements))
                self._m_all_measurement_loc = np.vstack(
                    (self._m_all_measurement_loc, measurement_loc))
            measurement_loc = self._m_all_measurement_loc
            measurements = self._m_all_measurements

        return measurement_loc, measurements

    def reset(self):
        """ Clear buffers to start estimating again."""
        self._m_all_measurements = None
        # self.m_all_mask = None
        self._m_all_measurement_loc = None
        # self.m_building_meta_data = None
        # self._m_building_meta_data_grid = None

    def train(self,
              map_generator,
              num_obs,
              evaluation_mode,
              num_mc_iter,
              verbosity=0):
        """
        This function evaluates the error indicated by `evaluation_mode` for
        each combination of parameter values in `self.d_train_params`. Then, it
        will set the parameters to the combination of values that minimizes the
        error. 

        Args:
                
        """

        l_lengths = [len(v) for v in self.d_train_params.values()]
        l_combinations = list(it.product(*self.d_train_params.values()))
        num_params = len(self.d_train_params.keys())

        def set_params(t):
            for ind_param, param_name in enumerate(self.d_train_params.keys()):
                setattr(self, param_name, t[ind_param])

        def rmse_sim(t):
            set_params(t)
            rmse = MapEstimationSimulator.get_rmse(
                map_generator=map_generator,
                estimator=self,
                num_obs=num_obs,
                evaluation_mode=evaluation_mode,
                num_mc_iterations=num_mc_iter)
            if verbosity > 0:
                print(
                    f"RMSE({evaluation_mode}) for {tuple(self.d_train_params.keys())}={t} is {rmse}"
                )
            return rmse

        v_rmse = np.array(list(map(rmse_sim, l_combinations)))

        # num_vals_param_0 x num_vals_param_1 x ...
        t_rmse = np.reshape(v_rmse, l_lengths)

        inds_arg_min = np.unravel_index(np.argmin(t_rmse), t_rmse.shape)
        set_params(
            [v[i] for v, i in zip(self.d_train_params.values(), inds_arg_min)])

        # Obtain the RMSE for each combination of parameters

        if verbosity >= 5:

            if num_params == 1:
                G = GFigure(xaxis=list(self.d_train_params.values())[0],
                            yaxis=np.ravel(t_rmse),
                            xlabel=list(self.d_train_params.keys())[0],
                            ylabel=f"Training RMSE({evaluation_mode}) [dB]")

            elif num_params == 2:
                # 3D plot where the y and x axes correspond to param 0 and param 1

                G = GFigure(xaxis=list(self.d_train_params.values())[1],
                            yaxis=list(self.d_train_params.values())[0],
                            zaxis=t_rmse,
                            xlabel=f'{list(self.d_train_params.keys())[1]}',
                            ylabel=f'{list(self.d_train_params.keys())[0]}',
                            zlabel=f"Training RMSE({evaluation_mode}) [dB]",
                            mode='surface',
                            num_subplot_rows=1)

                G.next_subplot(xaxis=list(self.d_train_params.values())[1],
                               yaxis=list(self.d_train_params.values())[0],
                               zaxis=t_rmse,
                               xlabel=f'{list(self.d_train_params.keys())[1]}',
                               ylabel=f'{list(self.d_train_params.keys())[0]}',
                               zlabel=f"Training RMSE({evaluation_mode}) [dB]",
                               mode='imshow',
                               grid=True,
                               aspect="square")
                G.add_curve(
                    xaxis=[
                        list(self.d_train_params.values())[1][inds_arg_min[1]]
                    ],
                    yaxis=[
                        list(self.d_train_params.values())[0][inds_arg_min[0]]
                    ],
                    styles='wo',
                )

            elif num_params == 3:
                # One 3D plot as above for each value of param 2

                G = GFigure(num_subplot_rows=len(
                    list(self.d_train_params.values())[2]))
                for ind in range(len(list(self.d_train_params.values())[2])):
                    G.next_subplot(
                        xaxis=list(self.d_train_params.values())[1],
                        yaxis=list(self.d_train_params.values())[0],
                        zaxis=t_rmse[..., ind],
                        xlabel=f'{list(self.d_train_params.keys())[1]}',
                        ylabel=f'{list(self.d_train_params.keys())[0]}',
                        zlabel=f"Training RMSE({evaluation_mode}) [dB]",
                        mode='surface',
                        title=
                        f'{list(self.d_train_params.keys())[2]}={list(self.d_train_params.values())[2][ind]}'
                    )

                    G.next_subplot(
                        xaxis=list(self.d_train_params.values())[1],
                        yaxis=list(self.d_train_params.values())[0],
                        zaxis=t_rmse[..., ind],
                        xlabel=f'{list(self.d_train_params.keys())[1]}',
                        ylabel=f'{list(self.d_train_params.keys())[0]}',
                        zlabel=f"Training RMSE({evaluation_mode}) [dB]",
                        mode='imshow',
                        grid=True,
                        aspect="square")
                    if ind == inds_arg_min[2]:
                        G.add_curve(
                            xaxis=[
                                list(self.d_train_params.values())[1][
                                    inds_arg_min[1]]
                            ],
                            yaxis=[
                                list(self.d_train_params.values())[0][
                                    inds_arg_min[0]]
                            ],
                            styles='wo',
                        )

            if verbosity == 6:
                G.plot()
                G.show()

            return G

    def save_estimator(self, folder_path):
        """
        Saves the estimator to a file with file name equal to the class name.
        
        Args:
            `folder_path`: path to the folder where the estimator will be saved.
        """
        os.makedirs(folder_path, exist_ok=True)
        with open(folder_path + self.__class__.__name__ + ".pickle",
                  'wb') as f:
            pickle.dump(self, f)

    def load_estimator(self, folder_path):
        """
        Loads the estimator from a file with file name equal to the class name.

        Args:
            `folder_path`: path to the folder where the estimator will be loaded from.
        """
        with open(folder_path + self.__class__.__name__ + ".pickle",
                  'rb') as f:
            estimator = pickle.load(f)
        return estimator

    ########################### Utilities for the subclasses ###############################

    def estimate_metric_per_channel(self,
                                    measurement_loc,
                                    measurements,
                                    building_meta_data=None,
                                    test_loc=None,
                                    f_power_est=None,
                                    f_service_est=None):
        """
        Concatenates the estimates returned by `f_power_est` and `f_service_est`
        by invoking these functions per channel.
        Args:
             if `measurements` is in the form of
                num_sources x num_points_y x num_points_x: # input in grid form
                - `building_meta_data`: num_points_y x num_points_x
                - `measurement_locs`: num_points_y x num_points_x is a sampling mask
            else:                         # input in standard form
                - `measurements`: num_sources x num_measurements  matrix
                   with the measurements at each channel.
                - `measurement_locs`: 3 x num_measurements matrix with the
                    3D locations of the measurements.
                - `building_meta_data`: 3 x num_points matrix with the 3D locations of the buildings

            -`f_power_est`: is a method provided by the subclass to estimate the power map per channel.
                    The input parameters of this method are:
                    Args:
                    if `estimation_func_type` is 's2s' or 's2g':
                        -`ind_channel`: an integer value corresponding to the channel(or source)
                        - `measurements`: num_sources x num_measurements  matrix
                            with the measurements at each channel.
                        - `measurement_locs`: 3 x num_measurements matrix with the
                            3D locations of the measurements.
                        
                    elif `estimation_func_type` is 'g2g':
                        - `measurements`: num_sources x num_points_y x num_points_x
                        - `building_meta_data`: num_points_y x num_points_x whose
                        value at each grid point is 1 if the grid point is inside a building
                        - `measurement_locs`: num_points_y x num_points_x, is
                            a sampling mask whose value is 1 at the grid point where
                            the samples were taken.

                    if `estimation_func_type` is 's2s':
                        - `m_test_loc`: num_test_loc x 3 matrix with the
                        locations where the map estimate will be evaluated.
                    Returns:
                        if `estimation_func_type` is 's2s':
                        - `v_estimated_pow_of_source`: num_test_loc length vector that represents
                            the estimated power at each test location for ind_channel
                        - `v_variance`: a num_test_loc length vector that provides the posterior
                            variance at each test location for ind_channel

                        elif `estimation_func_type` is 'g2g' or 's2g':
                            Two num_points_y x num_points_x matrices of
                            the estimated posterior mean and the posterior variance of ind_channel.

            -`f_service_est`: is a method provided by the subclass to estimate the service map per channel.
                    This method should take the input parameters:
                    Args:
                        if`estimation_func_type` is 's2s':
                            `mean_power`, `variance` of shape num_test_loc length vector

                        elif `estimation_func_type` is 'g2g':
                            num_points_y x num_points_x
                    Returns:
                        `service` and `entropy` of corresponding input shape.

        Returns:

            -`d_map_estimate`: dictionary whose fields are of the shape
                    num_sources x num_points_y x num_points_x
                    if estimation_fun_type is g2g, else the shape is num_test_loc x num_sources
        """

        # Now estimate metric per channel
        if self.estimation_fun_type == 's2s':
            num_sources = measurements.shape[0]
            num_test_loc = test_loc.shape[0]
            t_power_map_est = empty_array((num_test_loc, num_sources))
            t_power_map_var = empty_array((num_test_loc, num_sources))
            if self.min_service_power is not None:
                t_service_map_est = empty_array((num_test_loc, num_sources))
                t_service_map_ent = empty_array((num_test_loc, num_sources))
            for ind_src in range(num_sources):
                # t_power_map_est[:, ind_src], t_power_map_var[:, ind_src] = \
                #     f_power_est(ind_src, measurement_loc, measurements, test_loc)
                t_power_map_est[:,
                                ind_src], t_power_map_var_buffer = f_power_est(
                                    ind_src, measurement_loc, measurements,
                                    test_loc)
                if t_power_map_var_buffer is None:
                    t_power_map_var = None
                else:
                    t_power_map_var[:, ind_src] = t_power_map_var_buffer

                if self.min_service_power is not None:
                    t_service_map_est[:,
                                      ind_src], t_service_map_ent[:, ind_src] = f_service_est(
                                          t_power_map_est[:, ind_src],
                                          t_power_map_var[:, ind_src])

                    # self.estimate_service_one_gaussian_channel(self._m_all_measurements[ind_src, :], ind_src)
            # t_power_map_est = t_power_map_est.T
            # t_power_map_var = t_power_map_var.T
            # if self.min_service_power is not None:
            #     t_service_map_est = t_service_map_est.T
            #     t_service_map_ent = t_service_map_ent.T

        elif self.estimation_fun_type in ['g2g', 's2g']:
            num_sources = measurements.shape[0]
            t_power_map_est = empty_array(
                (num_sources, self.grid.num_points_y, self.grid.num_points_x))
            t_power_map_var = empty_array(
                (num_sources, self.grid.num_points_y, self.grid.num_points_x))
            if self.min_service_power is not None:
                t_service_map_est = empty_array(
                    (num_sources, self.grid.num_points_y,
                     self.grid.num_points_x))
                t_service_map_ent = empty_array(
                    (num_sources, self.grid.num_points_y,
                     self.grid.num_points_x))
            for ind_src in range(num_sources):
                # t_power_map_est[ind_src, :, :], t_power_map_var[ind_src, :, :] = \
                #     f_power_est(ind_src, measurement_loc, measurements, building_meta_data)
                t_power_map_est[
                    ind_src, :, :], t_power_map_var_buffer = f_power_est(
                        ind_src, measurement_loc, measurements,
                        building_meta_data)
                if t_power_map_var_buffer is None:
                    t_power_map_var = None
                else:
                    t_power_map_var[ind_src, :, :] = t_power_map_var_buffer

                if self.min_service_power is not None:
                    t_service_map_est[ind_src, :, :], t_service_map_ent[
                        ind_src, :, :] = f_service_est(
                            t_power_map_est[ind_src, :, :],
                            t_power_map_var[ind_src, :, :])

        else:
            raise NotImplementedError

        d_map_estimate = {"t_power_map_estimate": t_power_map_est}
        if t_power_map_var is not None:
            d_map_estimate[
                "t_power_map_norm_variance"] = t_power_map_var  #/ self.f_shadowing_covariance(0)  # ensure in (0,1)

        if self.min_service_power:
            d_map_estimate["t_service_map_estimate"] = t_service_map_est
            d_map_estimate["t_service_map_entropy"] = t_service_map_ent

        return d_map_estimate

    def estimate_service_one_gaussian_channel(self, mean_power, variance):
        """
        Returns:

            if inputs are in the form num_point_y x num_points_x matrix
            -`service`: num_point_y x num_points_x matrix where the
                (i,j)-th entry is the probability that the power at grid point
                (i,j) is greater than `self.min_service_power`.
            -`entropy`: bernoulli entropy associated with (i, j)-th grid point.

            else:
                The shape of the `service` and `entropy` are of are
                a vector of length num_test_loc
        """

        def entropy_bernoulli(service):
            # Avoid log2(0):
            service_copy = np.copy(service)
            b_zero_entropy = np.logical_or((service_copy == 0),
                                           (service_copy == 1))
            service_copy[b_zero_entropy] = .5  # dummy value

            entropy = -(1 - service_copy) * np.log2(1 - service_copy) - (
                service_copy) * np.log2(service_copy)
            entropy[b_zero_entropy] = 0

            return entropy

        service = 1 - norm.cdf(self.min_service_power, mean_power,
                               np.sqrt(variance))

        entropy = entropy_bernoulli(service)
        return service, entropy

    ########################### private methods of this class ###############################

    def _estimate_no_storage(self,
                             measurement_loc=None,
                             measurements=None,
                             building_meta_data=None,
                             test_loc=None):
        """Args:

            if `measurements` is in the form of
                num_sources x self.grid.num_points_y x self.grid.num_points_x: # input in grid form
                - `building_meta_data`: self.grid.num_points_y x self.grid.num_points_x
                - `measurement_locs`: self.grid.num_points_y x self.grid.num_points_x is a sampling mask
            else:                         # input in standard form
                - `measurements` : num_measurements x num_sources matrix
                   with the measurements at each channel.
                - `measurement_locs` : num_measurements x 3 matrix with the
                   3D locations of the measurements.
                - `building_meta_data`: num_points x 3 matrix with the 3D locations of the buildings

            `m_test_loc`: can be None or a num_test_loc x 3 matrix with the
            locations where the map estimate will be evaluated.

           Returns:

           `d_map_estimate`: dictionary whose fields are:

           - "t_power_map_estimate" :  tensor whose (i,j,k)-th entry is the
           estimated power of the i-th channel at grid point (j,k).
           - "t_power_map_norm_variance" : Contains the variance a
           posteriori normalized to the interval (0,1).
           - "t_service_map_estimate": [only if self.min_service_power
           is not None]. Each entry contains the posterior probability
           that the power is above `self.min_service_power`.
           - "t_service_map_entropy": [only if self.min_service_power
           is not None] Entropy of the service indicator.

           If `test_loc` is None, the shape of these items is num_sources
           x self.grid.num_points_y x self.grid.num_points_x.

           If `test_loc` is not None, the shape of these items is
           num_test_loc x num_sources.

        """

        # flag to check if measurements is in a grid form
        b_input_gf = (len(measurements.shape) == 3)
        b_output_gf = (test_loc is None)

        # check input
        if b_input_gf:
            assert measurement_loc.shape[0] == measurements.shape[1]
            assert measurement_loc.shape[1] == measurements.shape[2]
        else:
            assert measurement_loc.shape[1] == 3

        # input adaptation
        if b_input_gf and self.estimation_fun_type == 's2s':
            measurement_loc, measurements, building_meta_data = self._input_g2s(
                measurement_loc, measurements, building_meta_data)

        elif not b_input_gf and self.estimation_fun_type == 'g2g':
            measurement_loc, measurements, building_meta_data = self._input_s2g(
                measurement_loc, measurements, building_meta_data)

        # if b_output_gf then set all the grid points as test locations for s2s
        if b_output_gf:
            test_loc = self.grid.all_grid_points_in_matrix_form

        # invoke the method from the subclass
        if self.estimation_fun_type == 's2s':
            d_map_estimate = self.estimate_s2s(measurement_loc, measurements,
                                               building_meta_data, test_loc)
        elif self.estimation_fun_type == 'g2g':
            d_map_estimate = self.estimate_g2g(measurement_loc, measurements,
                                               building_meta_data)
            # d_map_estimate_old = self.estimate_g2g_old(measurement_loc_old, measurements_old, building_meta_data_old)
            # print("Norm equals", np.linalg.norm(d_map_estimate["t_power_map_norm_variance"] -
            #                                     d_map_estimate_old["t_power_map_norm_variance"]))
        else:
            raise NotImplementedError

        # output adaptation
        if b_output_gf:
            if self.estimation_fun_type == 's2s':
                d_map_estimate = self._output_s2g(d_map_estimate)
                # d_map_estimate_old = self.estimate_s2s_old(measurement_locs, measurements,
                #                                            building_meta_data, test_loc)
                # print("Norm equals", np.linalg.norm(d_map_estimate["t_power_map_estimate"] -
                #                                     d_map_estimate_old["t_power_map_estimate"]))

                return d_map_estimate
            elif self.estimation_fun_type == 'g2g':
                return d_map_estimate
            else:
                raise NotImplementedError
        else:
            if self.estimation_fun_type == 's2s':
                return d_map_estimate
            # elif self.estimation_fun_type == 'g2g':
            #     d_map_estimate = self._output_g2s(d_map_estimate)
            else:
                #TODO: pass the interpolation method instead of fix it here
                sampler = InterpolationSampler(
                    interpolation_method=self.interpolation_method,
                    grid=self.grid)
                for key in d_map_estimate.keys():
                    d_map_estimate[key] = np.array([
                        sampler.sample_map(d_map_estimate[key], loc)
                        for loc in test_loc
                    ])
                return d_map_estimate

    def _input_g2s(self,
                   measurement_loc=None,
                   measurements=None,
                   building_meta_data=None):
        """
        Args:
            - `measurements`: num_sources x self.grid.num_points_y x self.grid.num_points_x
            - `building_meta_data`: self.grid.num_points_y x self.grid.num_points_x
            - `measurement_locs`: self.grid.num_points_y x self.grid.num_points_x, a mask

        Returns:
            - `measurements` : num_measurements x num_sources matrix
               with the measurements at each channel.
            - `measurement_locs` : num_measurements x 3 matrix with the
               3D locations of the measurements.
            - `building_meta_data`: num_points x 3 matrix with the 3D locations of the buildings
        """

        m_measurements = measurements[:, measurement_loc == 1]
        measurements = m_measurements.T

        measurement_loc = self.grid.convert_grid_meta_data_to_standard_form(
            m_meta_data=measurement_loc)

        if building_meta_data is not None:
            building_meta_data = self.grid.convert_grid_meta_data_to_standard_form(
                m_meta_data=building_meta_data)

        return measurement_loc, measurements, building_meta_data

    def _input_s2g(self,
                   measurement_loc=None,
                   measurements=None,
                   building_meta_data=None):
        """
        Args:
            - `measurements` : num_measurements x num_sources matrix
               with the measurements at each channel.
            - `measurement_locs` : num_measurements x 3 matrix with the
               3D locations of the measurements.
            - `building_meta_data`: num_points x 3 matrix with the 3D locations of the buildings

        Returns:
            _`t_all_measurements_grid`: a tensor of shape
                num_sources x num_gird_points_y x num_grid_points_x
            -`m_mask`: num_gird_points_y x num_grid_points_x binary mask whose
                entry is 1 at the grid point where measurement is taken,
            -`building_meta_data_grid`: num_grid_points_y x num_grid_points_x binary mask whose
                entry is 1 at the grid points that are inside a building,
                0 otherwise.
        """
        m_all_measurements = measurements.T
        m_all_measurements_loc = measurement_loc.T
        num_sources = m_all_measurements.shape[0]

        assert self.grid
        t_all_measurements_grid = np.zeros(
            (num_sources, self.grid.num_points_y, self.grid.num_points_x))
        m_mask = np.zeros((self.grid.num_points_y, self.grid.num_points_x))

        m_all_measurements_loc_trans = m_all_measurements_loc.T

        m_all_measurements_col_index = 0  # to iterate through column of measurements

        # buffer counter to count repeated measurement in the grid point
        m_counter = np.zeros(np.shape(m_mask))

        for v_measurement_loc in m_all_measurements_loc_trans:
            # Find the nearest indices of the grid point closet to v_measurement_loc
            v_meas_loc_inds = self.grid.nearest_gridpoint_inds(
                v_measurement_loc)

            # replace the value of nearest grid point indices with measured value
            # for ind_sources in range(num_sources):
            #     t_all_measurements_grid[ind_sources, v_measurement_loc_inds[0],
            #                             v_measurement_loc_inds[1]] = v_measurement[ind_sources]

            # Add the previous measurements to the current measurement
            # at the (j,k)-th grid point
            t_all_measurements_grid[:, v_meas_loc_inds[0], v_meas_loc_inds[
                1]] += m_all_measurements[:, m_all_measurements_col_index]

            # increment counters to store repeated measurements at the (j, k)-th grid point
            m_counter[v_meas_loc_inds[0], v_meas_loc_inds[1]] += 1

            # set the value of mask to 1 at the measured grid point indices
            m_mask[v_meas_loc_inds[0], v_meas_loc_inds[1]] = 1

            m_all_measurements_col_index += 1

        # Average the measurements
        t_all_measurements_grid = np.divide(
            t_all_measurements_grid,
            m_counter,
            where=m_counter != 0,
            out=np.zeros(np.shape(t_all_measurements_grid)))

        building_meta_data_grid = np.zeros(
            (self.grid.num_points_y, self.grid.num_points_x))
        if building_meta_data is not None:
            for v_building_meta_data in building_meta_data:
                v_meta_data_inds = self.grid.nearest_gridpoint_inds(
                    v_building_meta_data)
                building_meta_data_grid[v_meta_data_inds[0],
                                        v_meta_data_inds[1]] = 1

        return m_mask, t_all_measurements_grid, building_meta_data_grid

    def _output_s2g(self, d_map_estimate=None):
        """
        Args:
            -`d_map_estimate`: whose fields are in the form
                num_test_loc x num_sources

        Returns:
            - `d_map_estimate`: whose fields are in the form
                    num_sources x self.grid.num_points_y x self.grid.num_points_x
            """
        num_sources = d_map_estimate["t_power_map_estimate"].shape[1]
        d_map_estimate["t_power_map_estimate"] = np.reshape(
            d_map_estimate["t_power_map_estimate"].T,
            (num_sources, self.grid.num_points_y, self.grid.num_points_x))
        if "t_power_map_norm_variance" in d_map_estimate.keys(
        ) and d_map_estimate["t_power_map_norm_variance"] is not None:
            d_map_estimate["t_power_map_norm_variance"] = np.reshape(
                d_map_estimate["t_power_map_norm_variance"].T,
                (num_sources, self.grid.num_points_y, self.grid.num_points_x))
        if self.min_service_power:
            d_map_estimate["t_service_map_estimate"] = np.reshape(
                d_map_estimate["t_service_map_estimate"].T,
                (num_sources, self.grid.num_points_y, self.grid.num_points_x))
            d_map_estimate["t_service_map_entropy"] = np.reshape(
                d_map_estimate["t_service_map_entropy"].T,
                (num_sources, self.grid.num_points_y, self.grid.num_points_x))
        return d_map_estimate

    ########################### Old methods ################################################
    def estimate_at_loc(self, m_measurement_loc, m_measurements, m_test_loc,
                        **kwargs):
        """Args:

            `m_measurement_loc`: num_measurements x num_dims_grid matrix

            `m_measurements`: num_measurements x num_channels matrix

            `m_test_loc`: num_test_loc x num_channels matrix

        Returns:

            dictionary with keys:

                `power_est`: num_test_loc x num_channels matrix with
                the power estimates at the test locations.

        """

        if self.freq_basis is None:
            power_est = np.array([
                self._estimate_power_map_one_channel_at_loc(
                    m_measurement_loc, m_measurements[:, ind_ch], m_test_loc,
                    **kwargs)[:, 0]
                for ind_ch in range(m_measurements.shape[1])
            ]).T
            d_est = {"power_est": power_est}
        else:
            raise NotImplementedError

        return d_est

    @abc.abstractmethod
    def _estimate_power_map_one_channel_at_loc(self, m_measurement_loc,
                                               v_measurements, m_test_loc,
                                               **kwargs):
        """Args:

            `m_measurement_loc`: num_meas x num_dims matrix

            `v_measurements`: length-num_meas vector

            `m_test_loc`: num_test_loc x num_dims matrix with the
            locations where the map estimate will be evaluated.

          Returns:

            length num_meas vector with the power estimates at
            locations `m_test_loc`.

        """
        pass
