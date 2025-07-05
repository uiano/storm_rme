import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ..gsim.gfigure import GFigure


class MapEstimationSimulator:

    def simulate_one_run(map_generator,
                         estimator,
                         evaluation_mode,
                         l_num_obs,
                         num_max_attempts=1000):
        """
        Generates a random map using `map_generator`

        Estimates the map using `estimator`  
        
        Returns:

            - l_mse: list of MSEs for each element of `l_num_obs`

            - energy: average energy of the true map
            
            - dist: Euclidean distance between the m_tx_loc and the center of
              the map. If m_tx_loc is None, dist is None.
        
        """

        l_mse = []
        l_energy = []

        if evaluation_mode == 'grid_nobs' or evaluation_mode == 'grid_all':
            sampling_mode = 'grid'
        else:
            sampling_mode = evaluation_mode

        attempts = 0
        while True:
            rmap = map_generator.generate_map()
            if rmap.get_num_meas(sampling_mode) >= max(l_num_obs) + 1:
                break
            attempts += 1

            if attempts == 20:
                print(
                    "Warning: could not generate a map with enough measurements. Trying again."
                )

            if attempts == num_max_attempts:
                raise ValueError(
                    "Reached maximum number of attempts to generate a map with enough measurements."
                )

        # calculate the Euclidean distance between the m_tx_loc and the center of the map
        if rmap.m_tx_loc is not None and rmap.grid is not None:
            if evaluation_mode in ['grid_nobs', 'grid_all']:
                # calculate the center of the map given the a matrix of 3D points
                center_point = [(rmap.grid.max_x - rmap.grid.min_x) / 2,
                                (rmap.grid.max_y - rmap.grid.min_y) / 2,
                                rmap.grid.z_value()]
            else:
                center_point = np.mean(rmap.m_meas_locs_sf, axis=0)

            # calculate the Euclidean distance between the m_tx_loc and the center of the map
            dist = np.linalg.norm(rmap.m_tx_loc - center_point)
        else:
            dist = None

        for num_obs in l_num_obs:

            rmap_obs, rmap_nobs = rmap.get_obs(sampling_mode=sampling_mode,
                                               num_obs=num_obs,
                                               return_nobs=True)
            d_map_estimate = estimator.estimate(
                rmap_obs,
                test_loc=rmap_nobs.m_meas_locs_sf if
                (evaluation_mode == 'uniform_standard'
                 or evaluation_mode == 'grid_standard') else None)
            if evaluation_mode == 'uniform_standard' or evaluation_mode == 'grid_standard':
                mse = np.linalg.norm(
                    rmap_nobs.m_meas_sf -
                    d_map_estimate["t_power_map_estimate"].m_meas_sf,
                    'fro')**2 / len(rmap_nobs.m_meas_sf)

                energy = np.linalg.norm(rmap_nobs.m_meas_sf, 'fro')**2 / len(
                    rmap_nobs.m_meas_sf)

            elif evaluation_mode == 'grid_nobs':
                mse = (np.linalg.norm(
                    np.where(rmap_nobs.mask == 1, rmap_nobs.t_meas_gf, 0) -
                    np.where(rmap_nobs.mask == 1,
                             d_map_estimate["t_power_map_estimate"].t_meas_gf,
                             0)))**2 / np.sum(rmap_nobs.mask)

                energy = (np.linalg.norm(
                    np.where(rmap_nobs.mask == 1, rmap_nobs.t_meas_gf,
                             0)))**2 / np.sum(rmap_nobs.mask)

            elif evaluation_mode == 'grid_all':
                mse = (np.linalg.norm(
                    np.where(rmap.mask == 1, rmap.t_meas_gf, 0) - np.where(
                        rmap.mask == 1, d_map_estimate["t_power_map_estimate"].
                        t_meas_gf, 0)))**2 / np.sum(rmap.mask)

                energy = (np.linalg.norm(
                    np.where(rmap.mask == 1, rmap.t_meas_gf, 0)))**2 / np.sum(
                        rmap.mask)
            else:
                raise ValueError

            l_mse.append(mse)
            l_energy.append(energy)

        return l_mse, np.mean(l_energy), dist

    def get_mse_vs_num_obs(map_generator,
                           estimator,
                           l_num_obs,
                           evaluation_mode,
                           num_mc_iterations,
                           normalize=False):
        l_v_mse = []
        l_energy = []
        for _ in tqdm(range(num_mc_iterations)):
            v_mse_error, energy, _ = MapEstimationSimulator.simulate_one_run(
                map_generator=map_generator,
                estimator=estimator,
                l_num_obs=l_num_obs,
                evaluation_mode=evaluation_mode)
            l_v_mse.append(v_mse_error)  # num_mc_iterations x len(l_num_obs)
            l_energy.append(energy)

        # Average the v_mse for each element of `l_num_obs`
        v_mse = np.mean(l_v_mse, 0)  # length = len(l_num_obs)
        if normalize:
            v_mse = v_mse / np.mean(l_energy)
        return v_mse

    def get_mse_vs_dist(map_generator,
                        estimator,
                        l_num_obs,
                        evaluation_mode,
                        num_mc_iterations,
                        normalize=False):
        """
        Returns:
            
                - l_metrics: list of tuples (legend_str, v_dist, v_mse, style)
        """
        l_v_mse = []
        l_energy = []
        l_dist = []
        l_metrics = []
        for _ in range(num_mc_iterations):
            v_mse_error, energy, dist = MapEstimationSimulator.simulate_one_run(
                map_generator=map_generator,
                estimator=estimator,
                l_num_obs=l_num_obs,
                evaluation_mode=evaluation_mode)
            l_v_mse.append(v_mse_error)  # num_mc_iterations x len(l_num_obs)
            l_energy.append(energy)
            l_dist.append(dist)  # num_mc_iterations x 1
            assert dist is not None, "Distance is None, which means the dataset does not have a transmitter location."

        def get_legend_str(num_obs):
            if (len(l_num_obs) > 1):
                return '$N_{obs}$ = ' + f"{num_obs}"
            else:
                return ""

        # get the v_mse for each element of `l_num_obs` from `l_v_mse`
        m_v_mse = np.array(l_v_mse)  # num_mc_iterations x len(l_num_obs)
        for ind in range(len(l_num_obs)):

            # get v_mse for each element of `l_num_obs`
            v_mse = m_v_mse[:, ind]

            # Pair the distances with their corresponding MSEs
            paired_list = list(zip(l_dist, v_mse))

            # Sort the paired list by the first element of each tuple (distance)
            paired_list.sort(key=lambda x: x[0])

            # Unzip the paired list back into two separate lists
            l_dist_sorted, l_mse_sorted = zip(*paired_list)

            # Convert back to lists
            l_dist_sorted = list(l_dist_sorted)
            l_mse_sorted = list(l_mse_sorted)

            l_metrics.append((get_legend_str(l_num_obs[ind]), l_dist_sorted,
                              l_mse_sorted, '.'))

        return l_metrics

    def get_rmse(map_generator, estimator, num_obs, evaluation_mode,
                 num_mc_iterations):
        """
        `num_obs`: tuple with the minimum and maximum number of observations. If
        scalar, it is assumed that minimum=maximum=num_obs.
        """
        # Broadcast num_obs
        t_num_obs = (num_obs, num_obs) if np.isscalar(num_obs) else num_obs

        l_mse = []
        for ind_mc in range(num_mc_iterations):

            if ind_mc % 100 == 0:
                print(f"--- MC iteration {ind_mc + 1}/{num_mc_iterations}")

            num_obs = np.random.randint(t_num_obs[0], t_num_obs[1] + 1)
            mse_error = MapEstimationSimulator.simulate_one_run(
                map_generator=map_generator,
                estimator=estimator,
                l_num_obs=[num_obs],
                evaluation_mode=evaluation_mode)[0][0]
            l_mse.append(mse_error)  # num_mc_iterations x len(l_num_obs)

        mse = np.mean(l_mse)
        return np.sqrt(mse)

    @staticmethod
    def plot_rmse(l_metrics, xlabel, ylabel, title=None):
        """
        `l_metrics` is a list of tuples (legend_str, v_x, v_y)
        """

        G_mse = GFigure(num_subplot_rows=1)

        G_mse.select_subplot(0,
                             xlabel=xlabel,
                             ylabel=ylabel,
                             title=title if title is not None else "")

        for str_legend, v_x, v_y, style in l_metrics:

            G_mse.add_curve(xaxis=v_x,
                            yaxis=np.sqrt(v_y),
                            legend=str_legend,
                            styles=style)

        return G_mse

    @staticmethod
    def compare_estimators_monte_carlo(map_generator,
                                       num_mc_iterations,
                                       l_estimators,
                                       l_num_obs=[],
                                       evaluation_mode='uniform_standard',
                                       normalize=False):
        """
            'evaluation_mode': 

                - 'uniform_standard': calculates the MSE on non-observation
                locations of the estimators using the standard method of
                sampling

                - 'grid_nobs': calculates the MSE on non-observation pixels of
                  the
                estimators using the grid method of sampling

                - 'grid_standard': calculates the MSE on non-observation
                  locations
                of the estimators using the grid method of sampling

                - 'grid_all': calculates the MSE on all available pixels of the
                estimators using the grid method of sampling

                - `normalize`: if True, the NMSE is returned instead of the MSE.

            Returns:

                - l_mse: list of tuples (legend_str, l_num_obs, v_mse, style).
                  Each entry corresponds to one estimator in `l_estimators`.
        """

        l_mse = []
        for estimator in l_estimators:

            v_mse = MapEstimationSimulator.get_mse_vs_num_obs(
                map_generator, estimator, l_num_obs, evaluation_mode,
                num_mc_iterations, normalize)

            l_mse.append((estimator.name_on_figs, l_num_obs, v_mse, '.-'))

        return l_mse

    def test_rmse_from_blocks(test_data,
                              estimator,
                              num_measurements,
                              num_iter=None,
                              shuffle=True):
        """Estimates RMSE using the test data in `test_data`. The format of
        `test_data` is the one of MeasurementDataset.data. For each
        entry in this Dataset, `num_measurements` measurements are
        selected at random if `shuffle==True`. Then, `estimator` is
        requested to estimate the map given these observed
        measurements at the locations of the measurements that were
        not observed. This operation is performed `num_iter` times,
        cycling over `test_data` if necessary. If `num_iter` is None,
        then each entry of `test_data` is used exactly once.

        """

        if isinstance(test_data, tf.data.Dataset):
            if num_iter is not None:
                test_data = test_data.repeat(num_iter)
            ds_it = iter(test_data)

        if num_iter is None:
            num_iter = len(test_data)

        mse = 0
        for ind_iter in range(num_iter):

            if isinstance(test_data, list):
                m_locations, m_measurements = test_data[ind_iter %
                                                        len(test_data)]
            elif isinstance(test_data, tf.data.Dataset):
                m_locations, m_measurements = next(ds_it)
                m_locations = m_locations.numpy()
                m_measurements = m_measurements.numpy()

            assert num_measurements < len(
                m_measurements), "Not enough measurements"

            if shuffle:
                v_permutation = np.random.permutation(len(m_locations))
                m_locations = m_locations[v_permutation]
                m_measurements = m_measurements[v_permutation]

            m_estimates = estimator.estimate_at_loc(
                m_locations[:num_measurements],
                m_measurements[:num_measurements],
                m_locations[num_measurements:])["power_est"]

            debug = False
            if debug:
                print(f"""estimator = {estimator.__class__},
                input loc = {m_locations[:num_measurements]},
                input meas = {m_measurements[:num_measurements]},
                test loc = {m_locations[num_measurements:]},
                estimate = {m_estimates}
                """)

            mse += np.sum((m_measurements[num_measurements:] - m_estimates)**
                          2) / m_estimates.size

        rmse = np.sqrt(mse / num_iter)

        return rmse
