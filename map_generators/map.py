import os
import pickle
import random
from collections import defaultdict

import numpy as np
from numpy.typing import NDArray
#from util.communications import dbm_to_natural, natural_to_dbm, dbm_to_db, db_to_natural, natural_to_db
import pandas as pd
from sklearn.utils import shuffle

from ..gsim.gfigure import GFigure
from ..map_generators.grid import RectangularGrid
from ..measurement_generation.sampler import InterpolationSampler
from ..utilities import dB_to_natural, natural_to_dB


class Map():
    """
    This class stores the measurement locations and measurements
    that lie inside the rectangle (or patch) selected randomly from
    the large map.
    """

    def __init__(self,
                 m_meas_locs_sf=None,
                 m_meas_sf=None,
                 grid: RectangularGrid | None = None,
                 gridpt_spacing=None,
                 t_meas_gf=None,
                 interpolation_method='avg_nearest',
                 grid_quantization_mode='db_mean',
                 return_natural=False,
                 m_tx_loc=None):
        """
        Args:

        - `m_meas_locs_sf`: num_measurements x 3 matrix containing 3D
          measurement locations. 

        - `m_meas_sf`: num_measurements x num_ch matrix containing measurements
          corresponding to `m_meas_locs_sf` measurement locations

        - `t_meas_gf`: num_ch x num_grid_points_y x num_grid_points_x matrix. A
          NaN entry means that the corresponding grid point is not known (or
          measured).
        
        - `grid`: object of class Grid. Must be None if `gridpt_spacing` is not
          None. 

        - `gridpt_spacing`: if provided, a default RectangularGrid with this
          spacing is set. 
        
        - `grid_quantization_mode`: can be 'db_mean', 'db_median',
          'natural_mean', 'natural_median'. It determines how the measurements
          are quantized to the grid points.

        - `return_natural`: if True, the measurements are returned in natural
          units. Otherwise, they are returned in dB units. Note that the map is
          always instantiated by providing dB units.

        - `m_tx_loc`: None or num_tx x 3 matrix containing the 3D coordinates of
          the transmitter locations. None means that no information about the
          transmitter locations is available.

        """

        # Input check
        if m_meas_locs_sf is not None:
            assert m_meas_sf is not None and t_meas_gf is None
            assert m_meas_sf.ndim == 2
            assert m_meas_locs_sf.ndim == 2
            assert m_meas_locs_sf.shape[0] == m_meas_sf.shape[0]
            assert m_meas_locs_sf.shape[1] == 3
        else:
            assert m_meas_sf is None and grid is not None and t_meas_gf is not None

        self.interpolation_method = interpolation_method
        self._m_meas_locs_sf = m_meas_locs_sf
        self._m_meas_sf = m_meas_sf
        self._t_meas_gf = None  # to bypass setter checks
        if gridpt_spacing is not None:
            assert grid is None
            self.set_default_grid(gridpt_spacing)
        else:
            self.grid = grid
        self._t_meas_gf = t_meas_gf

        self._grid_point_to_meas_assignment = None

        self._grid_quantization_mode = grid_quantization_mode
        self.return_natural = return_natural
        self.m_tx_loc = m_tx_loc

    def adapt_out_units(self, meas_dB: np.ndarray):
        if self.return_natural:
            return dB_to_natural(meas_dB)
        else:
            return meas_dB

    @property
    def grid_quantization_mode(self):
        return self._grid_quantization_mode

    @grid_quantization_mode.setter
    def grid_quantization_mode(self, grid_quantization_mode):
        self._grid_quantization_mode = grid_quantization_mode
        self._t_meas_gf = None

    def set_default_grid(self, gridpt_spacing):

        # If gridpt_spacing is a scalar, broadcast it to a vector with two equal entries:
        if np.isscalar(gridpt_spacing) or len(gridpt_spacing) == 1:
            gridpt_spacing = np.array([gridpt_spacing, gridpt_spacing])

        self.grid = RectangularGrid(
            gridpoint_spacing=gridpt_spacing,
            num_points_x=int(
                np.round(max(self.m_meas_locs_sf[:, 0]) / gridpt_spacing[0])) +
            1,
            num_points_y=int(
                np.round(max(self.m_meas_locs_sf[:, 1]) / gridpt_spacing[1])) +
            1,
            height=0)

    @property
    def m_meas_locs_sf(self):
        if self._m_meas_locs_sf is None:
            # Convert the measurement locations to standard form. The output
            # will be of shape num_meas x 3, where num_meas is the number of
            # measurements.
            assert self.mask is not None
            self._m_meas_locs_sf = self.grid.all_grid_points_in_matrix_form[
                np.ravel(self.mask)]
        return self._m_meas_locs_sf

    @property
    def m_meas_sf(self):
        if self._m_meas_sf is None:
            # Conversion from grid form to standard form. The output will be of
            # shape num_meas x num_ch, where num_meas is the number of
            # measurements, i.e., the number of non-nan fibers in t_meas_gf. At
            # most, num_meas can be num_grid_points_y * num_grid_points_x.
            num_ch = self._t_meas_gf.shape[0]
            self._m_meas_sf = np.transpose(self._t_meas_gf, (1, 2, 0)).reshape(
                (-1, num_ch))
            # Remove the rows that contain all nan entries
            self._m_meas_sf = self._m_meas_sf[self.mask.reshape(-1), :]

        return self.adapt_out_units(self._m_meas_sf)

    @property
    def t_meas_gf(self) -> NDArray:
        """
        Returns a tensor of shape num_ch x num_pts_y x num_pts_x containing the
        measurements in grid form. A NaN entry means that the corresponding
        grid point is not known (or measured).
        """
        if self._t_meas_gf is None:
            self._t_meas_gf = self._to_grid_form(self._m_meas_locs_sf,
                                                 self._m_meas_sf)
        return self.adapt_out_units(self._t_meas_gf)

    @property
    def mask(self):
        # Mask is num_pts_y x num_pts_x matrix. A 1 entry means that the
        # corresponding grid point is known (or measured).

        new_val = ~np.all(np.isnan(self.t_meas_gf), axis=0)

        # old_val = ~self.get_nan_indicator(self.t_meas_gf)
        # print("Asserting refactoring")
        # assert np.all(old_val == new_val)
        return new_val

    @staticmethod
    def get_nan_indicator(t_map):
        """
        Returns the indices of the nan entries in the map.
        """
        return np.isnan(np.sum(t_map, axis=0))

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, grid):
        if self._t_meas_gf is not None:
            raise Exception(
                "Cannot change grid after measurements have been set.")
        else:
            self._grid = grid

    def _to_grid_form(self, m_meas_locs_sf, m_meas_sf):
        """
        Args:
            see `__init__` method.

        Returns:
            _`t_meas_gf`: num_ch x num_grid_points_y x num_grid_points_x matrix. A
          NaN entry means that the corresponding grid point is not known (or
          measured).
        """
        assert self.grid, "A grid must be set. "
        num_points_y = self.grid.num_points_y
        num_points_x = self.grid.num_points_x

        t_meas_gf = np.zeros((m_meas_sf.shape[-1], num_points_y, num_points_x))

        t_mask = np.zeros((1, num_points_y, num_points_x))

        m_all_measurements_col_index = 0  # to iterate through column of measurements

        lll_lmeas = [[[[] for _ in range(num_points_x)]
                      for _ in range(num_points_y)]
                     for _ in range(m_meas_sf.shape[-1])]

        for v_measurement_loc in m_meas_locs_sf:

            # Find the indices of the grid point that is closest to v_measurement_loc
            v_meas_loc_inds = self.grid.nearest_gridpoint_inds(
                v_measurement_loc)

            for in_ch in range(m_meas_sf.shape[-1]):
                # Store the measurements at the (j,k)-th grid point as a list of measurements
                lll_lmeas[in_ch][v_meas_loc_inds[0]][
                    v_meas_loc_inds[1]].append(
                        m_meas_sf.T[in_ch, m_all_measurements_col_index])

            m_all_measurements_col_index += 1

            # set the value of mask to 1 at the measured grid point indices
            t_mask[0, v_meas_loc_inds[0], v_meas_loc_inds[1]] = 1

        if self.grid_quantization_mode == 'db_mean':
            # take the mean of each inner list of lll_lmeas and if the list is empty, set the value to 0 .
            t_meas_gf = np.array([[
                np.mean(lll_lmeas[in_ch][j][k])
                if lll_lmeas[in_ch][j][k] else 0 for k in range(num_points_x)
            ] for j in range(num_points_y)])

        elif self.grid_quantization_mode == 'natural_mean':
            # take the mean of each inner list of lll_lmeas in natural form and if the list is empty, set the value to 0.

            t_meas_gf = np.array([[
                np.mean(dB_to_natural(np.array(lll_lmeas[in_ch][j][k])))
                if lll_lmeas[in_ch][j][k] else 0 for k in range(num_points_x)
            ] for j in range(num_points_y)])

        elif self.grid_quantization_mode == 'db_median':

            # take the median of each inner list of lll_lmeas and if the list is empty, set the value to 0.
            t_meas_gf = np.array([[
                np.median(lll_lmeas[in_ch][j][k])
                if lll_lmeas[in_ch][j][k] else 0 for k in range(num_points_x)
            ] for j in range(num_points_y)])

        elif self.grid_quantization_mode == 'natural_median':

            # take the median of each inner list of lll_lmeas in natural form and if the list is empty, set the value to 0.
            t_meas_gf = np.array([[
                np.median(dB_to_natural(np.array(lll_lmeas[in_ch][j][k])))
                if lll_lmeas[in_ch][j][k] else 0 for k in range(num_points_x)
            ] for j in range(num_points_y)])

        # Convert the measurements to dB if the mode is natural_mean or natural_median
        if self.grid_quantization_mode == 'natural_mean' or self.grid_quantization_mode == 'natural_median':
            t_meas_gf = np.where(t_meas_gf != 0, natural_to_dB(t_meas_gf), 0)

        t_meas_gf = np.where(t_mask == 1, t_meas_gf, np.nan)
        return t_meas_gf

    @property
    def grid_point_to_meas_assignment(self):
        """
        Returns a list of length num_grid_points = (num_grid_points_y *
        num_grid_points_x) where the i-th element is a list that contains the
        indices of the measurements that correspond to the i-th grid point.
        """
        if self._grid_point_to_meas_assignment is None:
            l_l_grid_point_to_meas_assignment = [
                []
                for _ in range(self.grid.num_points_y * self.grid.num_points_x)
            ]
            for meas_ind, v_measurement_loc in enumerate(self.m_meas_locs_sf):
                v_meas_loc_inds = self.grid.nearest_gridpoint_inds(
                    v_measurement_loc)
                l_l_grid_point_to_meas_assignment[np.ravel_multi_index(
                    v_meas_loc_inds,
                    (self.grid.num_points_y,
                     self.grid.num_points_x))].append(meas_ind)
            self._grid_point_to_meas_assignment = l_l_grid_point_to_meas_assignment
        return self._grid_point_to_meas_assignment

    def get_num_meas(self, sampling_mode):
        if sampling_mode == 'grid' or sampling_mode == 'grid_standard':
            num_meas = np.sum(np.ravel(self.mask))
        elif sampling_mode == 'uniform_standard':
            num_meas = self.m_meas_locs_sf.shape[0]
        else:
            raise NotImplementedError

        return num_meas

    def get_obs(self,
                sampling_mode='uniform_standard',
                num_obs=None,
                frac_obs=None,
                return_nobs=False):
        """
        Args:
            'sampling_mode': 
                'uniform_standard': in this case, map_obs contains a randomly
                selected subset of num_meas measurements in self.m_meas_locs_sf
                and self.m_meas_sf.

                'grid_standard': in this case, a subset of num_meas grid points
                in self.grid is selected uniformly at random. map_obs contains
                the measurements in self.m_meas_locs_sf and self.m_meas_sf that
                correspond to the selected grid points. Note that the number of
                measurements in map_obs will generally differ from num_meas.

                'grid': map_obs contains a subset of measurements in
                self.t_meas_gf. map_obs.t_meas_gf will contain num_meas pixels
                with measurements (i.e. fibers different from nan).

            'num_obs': Number of observations. If None, then `frac_obs` must be
            provided.

            'frac_obs': Fraction of observations. If None, then `num_obs` must
            be provided.

            'return_nobs': If False, then only the map with observations is
            returned. If True, then the map with observations and the map with
            non-observations are returned.
        Returns:
            map_obs
            map_nobs
        """

        def get_obs_grid_form(t_map, num_obs, return_nobs):
            """
            Args:
                't_map': num_ch x num_grid_points_y x num_grid_points_x matrix.
                A NaN entry means that the corresponding grid point is not known
                (or measured).

                'num_obs': Number of measurements.
            
            Returns:
                't_map_obs': num_ch x num_grid_points_y x num_grid_points_x
                tensor containing the observations and nan for the
                non-observations. 
                
                't_map_nobs': num_ch x num_grid_points_y x num_grid_points_x
                tensor containing the non-observations and nan for the
                observations.
            """
            v_has_meas = np.argwhere(np.ravel(self.mask))
            observed_grid_points = len(v_has_meas)
            if observed_grid_points < num_obs:
                raise ValueError(
                    'Number of observations is larger than the number of measurements.'
                )
            v_obs_pixel_ind = np.random.choice(observed_grid_points,
                                               num_obs,
                                               replace=False)
            v_observation_mask = v_has_meas[v_obs_pixel_ind]
            m_map = np.reshape(t_map, (t_map.shape[0], -1))
            m_map_obs = np.full(np.shape(m_map), np.nan)
            m_map_obs[:, v_observation_mask] = m_map[:, v_observation_mask]
            t_map_obs = np.reshape(m_map_obs, t_map.shape)
            if not return_nobs:
                return t_map_obs

            m_nobs = np.copy(m_map)
            m_nobs[:, v_observation_mask] = np.nan
            t_map_nobs = np.reshape(m_nobs, t_map.shape)
            return t_map_obs, t_map_nobs

        def get_meas_inds_from_pixel_inds(grid_2_meas, l_pixel_indices):
            """ Get the measurement indices from the pixel indices.

            Args:
                'grid_2_meas': a list where the i-th element is a list that
                contains the indices of the measurements that correspond to the
                i-th grid point.

                'l_pixel_indices': a list of pixel indices to be chosen.

            Returns:
                A list of measurement indices corresponding to l_pixel_indices
                pixel indices
            """
            return sum(list(map(grid_2_meas.__getitem__, l_pixel_indices)), [])

        assert (num_obs is None) ^ (frac_obs is None)

        return_natural_save = self.return_natural
        self.return_natural = False  # To operate in dB units

        # Obtain num_obs
        num_meas = self.get_num_meas(sampling_mode)
        num_obs = int(num_meas * frac_obs) if frac_obs is not None else num_obs

        # Return map(s) with the selected observations
        if sampling_mode == 'grid':
            if not return_nobs:
                rmap = Map(t_meas_gf=get_obs_grid_form(self.t_meas_gf, num_obs,
                                                       return_nobs),
                           grid=self.grid,
                           return_natural=return_natural_save)
            else:
                t_map_obs, t_map_nobs = get_obs_grid_form(
                    self.t_meas_gf, num_obs, return_nobs)
                rmap = Map(t_meas_gf=t_map_obs,
                           grid=self.grid,
                           return_natural=return_natural_save), Map(
                               t_meas_gf=t_map_nobs,
                               grid=self.grid,
                               return_natural=return_natural_save)
        elif sampling_mode == 'uniform_standard':
            v_obs_ind = np.random.choice(num_meas, num_obs, replace=False)
            if not return_nobs:
                rmap = Map(m_meas_locs_sf=self.m_meas_locs_sf[v_obs_ind],
                           m_meas_sf=self.m_meas_sf[v_obs_ind],
                           grid=self.grid,
                           return_natural=return_natural_save)
            else:
                v_nobs_ind = np.setdiff1d(np.arange(num_meas), v_obs_ind)
                rmap = Map(m_meas_locs_sf=self.m_meas_locs_sf[v_obs_ind],
                           m_meas_sf=self.m_meas_sf[v_obs_ind],
                           grid=self.grid,
                           return_natural=return_natural_save), Map(
                               m_meas_locs_sf=self.m_meas_locs_sf[v_nobs_ind],
                               m_meas_sf=self.m_meas_sf[v_nobs_ind],
                               grid=self.grid,
                               return_natural=return_natural_save)
        elif sampling_mode == 'grid_standard':
            l_valid_pixel_indices = [
                ind for ind, grid_point in enumerate(
                    self.grid_point_to_meas_assignment) if len(grid_point) > 0
            ]
            l_obs_pixel_indices = np.random.choice(l_valid_pixel_indices,
                                                   num_obs,
                                                   replace=False)
            l_obs_indices = get_meas_inds_from_pixel_inds(
                self.grid_point_to_meas_assignment, l_obs_pixel_indices)
            if not return_nobs:
                rmap = Map(m_meas_locs_sf=self.m_meas_locs_sf[l_obs_indices],
                           m_meas_sf=self.m_meas_sf[l_obs_indices],
                           grid=self.grid,
                           return_natural=return_natural_save)
            else:
                l_nobs_pixel_indices = np.setdiff1d(l_valid_pixel_indices,
                                                    l_obs_pixel_indices)
                l_nobs_indices = get_meas_inds_from_pixel_inds(
                    self.grid_point_to_meas_assignment, l_nobs_pixel_indices)
                rmap = Map(
                    m_meas_locs_sf=self.m_meas_locs_sf[l_obs_indices],
                    m_meas_sf=self.m_meas_sf[l_obs_indices],
                    grid=self.grid,
                    return_natural=return_natural_save), Map(
                        m_meas_locs_sf=self.m_meas_locs_sf[l_nobs_indices],
                        m_meas_sf=self.m_meas_sf[l_nobs_indices],
                        grid=self.grid,
                        return_natural=return_natural_save)
        else:
            raise NotImplementedError

        self.return_natural = return_natural_save
        return rmap

    def plot_meas_locs(self):

        return GFigure(xaxis=self.m_meas_locs_sf[:, 0],
                       yaxis=self.m_meas_locs_sf[:, 1],
                       styles='x',
                       xlabel='x',
                       ylabel='y',
                       title='Measurement locations')

    def shift_meas_locs(self, v_shift):
        """
        This method shifts the measurement locations by the vector `v_shift`.
        """
        self.m_meas_locs_sf += np.array(v_shift)

    def eval(self, m_loc_sf):
        """
        Evaluates the map at given points by interpolating the grid-form
        representation of the map.

        Args:
            m_loc_sf: num_points x 3 matrix containing the 3D coordinates of the
            points where the map is to be evaluated.

        Returns:
            m_map: num_points x num_ch matrix containing the map evaluated at
            the points in m_loc_sf.
        
        """

        sampler = InterpolationSampler(
            interpolation_method=self.interpolation_method, grid=self.grid)

        return_natural_save = self.return_natural
        self.return_natural = False
        m_meas_sf = np.array(
            [sampler.sample_map(self.t_meas_gf, loc) for loc in m_loc_sf])
        self.return_natural = return_natural_save

        return m_meas_sf

    def plot(self, b_dB=True, mode='surface', m_meas_locs_sf=None):
        """
        Args:

        - `b_dB`: if True, the measurements are plotted in dBm scale, otherwise
          in linear scale.

        - `m_meas_locs_sf`: num_measurements x 3 matrix containing the 3D
          coordinates of the measurement locations. If provided, they are
          plotted as crosses. 
        
        """
        assert self.grid

        # t_meas_on_grid, t_mask = grid.convert_measurements_to_grid_form(
        #     self.m_meas_locs_sf.T, self.m_meas_sf.T)
        # t_meas_on_grid = np.where(t_mask == 1, t_meas_on_grid, np.nan)[0]

        t_meas_on_grid = self.t_meas_gf[0]

        if not b_dB:
            t_meas_on_grid = dB_to_natural(t_meas_on_grid)
            zlabel = 'Power [natural units]'
        else:
            zlabel = 'Power [dBm]'
            # Subtract 103 to get the power in dBm
            t_meas_on_grid -= 103

        G = GFigure(xaxis=self.grid.t_coordinates[0],
                    yaxis=self.grid.t_coordinates[1],
                    zaxis=t_meas_on_grid,
                    xlabel='x [m]',
                    ylabel='y [m]',
                    zlabel=zlabel,
                    xlim=[self.grid.min_x, self.grid.max_x],
                    ylim=[self.grid.min_y, self.grid.max_y],
                    global_color_bar=True,
                    zinterpolation=None,
                    mode=mode)

        if m_meas_locs_sf is not None:
            G.add_curve(xaxis=m_meas_locs_sf[:, 0],
                        yaxis=m_meas_locs_sf[:, 1],
                        styles='+k')

        return G

    def plot_grid_quantization(self, b_plot_tx_loc=True):
        G = self.grid.plot_grid_quantization(self.m_meas_locs_sf)

        if b_plot_tx_loc and self.m_tx_loc is not None:
            G.add_curve(xaxis=self.m_tx_loc[:, 0],
                        yaxis=self.m_tx_loc[:, 1],
                        styles='^b')

        return G
