import os
import pickle

import numpy as np
from IPython.core.debugger import set_trace

from ..map_generators.grid import RectangularGrid
from ..map_generators.map import Map
#from map_generators.grid import RectangularGrid, Grid

from ..utilities import (dB_to_natural, mat_argmax, mat_argmin, natural_to_dB,
                         project_to_interval)


class MapGenerator():

    def __init__(self,
                 combine_channels=False,
                 map_returns_natural=False,
                 grid_quantization_mode="db_mean"):
        """
        Args:
            - `combine_channels`: if True, the output of `generate_map_tensor`
            will be a 1 x grid.num_points_x x grid.num_points_y tensor, where
            the entry [0,j,k] is the aggregate power rx from all sources at grid
            point (j,k). Else, the output will be a num_sources x
            grid.num_points_x x grid.num_points_y tensor, where the entry
            [i,j,k] is the power rx from source i at grid point (j,k).

            - `map_returns_natural`: the value of the `return_natural` property
              of the generated Maps. 

            - `grid_quantization_mode`: can be 'db_mean', 'db_median',
              'natural_mean', 'natural_median'. It determines how the
              measurements are quantized to the grid points.
        """

        self.combine_channels = combine_channels
        self.grid_quantization_mode = grid_quantization_mode
        self.map_returns_natural = map_returns_natural

    def generate_map_tensor(self):
        """If self.combine_channels == False:

        returns a num_sources x grid.num_points_x x grid.num_points_y
        tensor, where the entry [i,j,k] is the power rx from source i
        at grid point (j,k).

        `m_meta_data`, a Ny x Nx matrix whose [i,j] element is 1 if
        the (i, j)th grid point is inside the building.

        Else:

        returns a 1 x grid.num_points_x x grid.num_points_y tensor,
        where the entry [0,j,k] is the aggregate power rx from all
        sources at grid point (j,k).

        `m_meta_data`, a Ny x Nx matrix whose [i,j] element is 1 if
        the (i, j)th grid point is inside the building.
        """

        source_maps, m_meta_data = self._generate_map_tensor()

        if self.combine_channels:
            return natural_to_dB(
                np.sum(dB_to_natural(source_maps),
                       axis=0)[None, :, :]), m_meta_data
        else:
            return source_maps, m_meta_data

    def _generate_map_tensor(self):
        raise NotImplementedError

    def generate_map(self):
        """
        Returns: `patch_map`: an object of the class `Map()` that contains the
        measurement locations and measurements which lie inside the randomly
        selected rectangle (or patch).
        
        """

        rmap = self._generate_map()
        rmap.return_natural = self.map_returns_natural
        return rmap

    def _generate_map(self) -> Map:
        """        
        To be overridden.         

        TODO: implement this by invoking the method `generate_map_tensor`. Bring
        the necessary properties from RealDataMapGenerator to MapGenerator.

        """
        raise NotImplementedError


class MapGeneratorFromFiles(MapGenerator):

    file_extension = ".pickle"

    def __init__(
            self,
            z_coord=None,
            patch_side_len=None,  # in meters
            gridpoint_spacing=None,  # in meters
            l_file_num=None,
            folder=None,  #
            *args,
            **kwargs):
        """
        The data can be stored in standard or in grid form. 
        
        Args:

        - l_file_num: it can be either:
        
            + list of indices of the files in the folder `folder` that are used
                to draw patches. 

        - folder:  path to the folder containing the data files. 

        - z_coord: If not None, the z-coordinate of the measurement locations is
          set to this value. This is useful to approximate the measurement
          locations as belonging to the same plane.

        - patch_side_len: 
        
            * If the files are in standard form, this is a vector with two
              entries that indicates side lengths of the patch (or rectangle)
              that is randomly selected, the first for the x-axis and the second
              for the y-axis. If it is a scalar, it is assumed that both entries
              are equal. If None, all measurements are returned.

            * If the files are in grid form, this argument should be None. 

        - `gridpoint_spacing`: Vector with two entries, the first for the x-axis
        and the second for the y-axis. If it is a scalar, it is assumed that
        both entries are equal. It is used to create a grid for a large map. 

            + If gridpoint_spacing is None, then the patches are obtained by
              drawing a rectangle of dimensions specified by patch_side_len
              uniformly at random inside the smallest rectangle that contains
              all the measurement locations.

            + If gridpoint_spacing is not None, then the origin [x0,y0] of the
              patch is selected uniformly at random among grid points. Then the
              measurements whose x-coordinate is between
              x0-gridpoint_spacing[0]/2 and x0-gridpoint_spacing[0]/2 +
              patch_side_len[0] and whose y coordinate is between
              y0-gridpoint_spacing[1]/2 and y0-gridpoint_spacing[1]/2 +
              patch_side_len[1] are selected. The vector [x0,y0] is then
              subtracted from the selected measurement locations, effectively
              translating the patch so that [x0,y0] becomes the origin. As a
              consequence, if the original measurement locations lie on the grid
              points of the entire area, the returned measurement locations will
              also lie on the grid points of a grid constructed for the patch
              using the same grid spacing.
        
       
        """

        assert folder, "The folder path is missing"
        super().__init__(*args, **kwargs)

        if folder[-1] not in ["/", "\\"]:
            if "\\" in folder:
                folder += "\\"
            else:
                folder += "/"
        self.folder = folder

        self.l_file_num = l_file_num

        if gridpoint_spacing is not None:
            assert patch_side_len is not None, "The patch_len must be provided if the gridpoint_spacing is provided."

        if patch_side_len is not None:
            # broadcast patch_side_len to a vector
            if np.isscalar(patch_side_len):
                patch_side_len = np.array([patch_side_len, patch_side_len])
        self.patch_side_len = patch_side_len

        if gridpoint_spacing is not None:
            # broadcast gridpoint_spacing to a vector
            if np.isscalar(gridpoint_spacing):
                gridpoint_spacing = np.array(
                    [gridpoint_spacing, gridpoint_spacing])
        self.gridpoint_spacing = gridpoint_spacing

        self.z_coord = z_coord

        # Create a vector with the names of the files inside the folder
        # `self.folder` sorted in alphabetical order.
        self.l_file_names = sorted([
            f for f in os.listdir(self.folder)
            if f.endswith(self.file_extension)
        ])

        self.z_coord = z_coord

    def _generate_map(self) -> Map:
        """
        Returns the measurements that lie inside a randomly selected rectangle.
        If self.patch_side_len is None, all measurements are returned. 

        Returns: `patch_map`: an object of the class `Map()` that contains the
        measurement locations and measurements which lie inside the randomly
        selected rectangle (or patch).
        """

        # read the data from file
        ind_file = np.random.choice(
            self.l_file_num,
            size=1,  # self.num_tx_per_channel
            replace=False)[0] if self.l_file_num is not None else None
        d_data = self._get_data_from_file(ind_file)

        if "m_locs_sf" in d_data:
            # The data is in standard form
            m_large_map_measurement_locs, m_large_map_measurements, m_tx_loc = (
                d_data["m_locs_sf"], d_data['m_meas_sf'],
                d_data.get("m_tx_locs", None))
            return self._generate_map_from_sf(m_large_map_measurement_locs,
                                              m_large_map_measurements,
                                              m_tx_loc)
        else:
            # The data is in grid form
            return self._generate_map_from_gf(
                lt_meas_gf=d_data["lt_meas_gf"],
                l_grid_spacing=d_data["l_grid_spacing"])

    def _generate_map_from_gf(self, lt_meas_gf, l_grid_spacing):

        # Choose a random map
        ind_map = np.random.choice(range(len(lt_meas_gf)))
        t_meas_gf = lt_meas_gf[ind_map]
        grid_spacing = l_grid_spacing[ind_map]

        return Map(grid=RectangularGrid(gridpoint_spacing=grid_spacing,
                                        num_points_x=t_meas_gf.shape[2],
                                        num_points_y=t_meas_gf.shape[1],
                                        height=self.z_coord),
                   t_meas_gf=t_meas_gf)

    def _generate_map_from_sf(self,
                              m_large_map_measurement_locs,
                              m_large_map_measurements,
                              m_tx_loc=None):

        def get_coordinates_and_meas_inside_rectangle(
                v_blc_coords, v_trc_coords, m_large_map_measurement_locs,
                m_large_map_meas):
            """This function returns arrays containing measurement coordinates and measurements
            that lie inside the rectangle whose bottom-left-coordinate = v_origin and
            top-right-coordinate = v_top_right_coord. 

            Args:
                -`v_origin`: is an origin of the patch or rectangle that contains x,y coordinates
                -`v_top_right_coord`: contains x, y coordinates of the top-right corner of the
                                    patch or rectangle
                -`m_large_map_measurement_locs`: num_measurements x 3 measurement locations in the large map
                -`m_large_map_meas`: num_measurements x 1 measurements in the large map

            Returns:
                -`m_patch_map_measurement_locs`: num_measurements x 3 measurement locations that lie inside
                                            the rectangle or patch.
                -`m_patch_map_meas`: num_measurements x 1 measurements corresponding to
                                    `m_patch_map_measurement_locs`
            """

            m_patch_map_measurement_locs = np.zeros(
                (0, 3))  # num_measurements_so_far x 3
            m_patch_map_meas = np.zeros((0, 1))  # num_measurements_so_far x 1

            for v_meas_locs, v_meas in zip(m_large_map_measurement_locs,
                                           m_large_map_meas):

                if ((v_blc_coords[0] <= v_meas_locs[0] <= v_trc_coords[0]) and
                    (v_blc_coords[1] <= v_meas_locs[1] <= v_trc_coords[1])):
                    m_patch_map_measurement_locs = np.vstack(
                        (m_patch_map_measurement_locs, v_meas_locs))
                    m_patch_map_meas = np.vstack((m_patch_map_meas, v_meas))

            return m_patch_map_measurement_locs, m_patch_map_meas

        if self.patch_side_len is None:
            # return all measurements
            m_patch_map_meas_locs, m_patch_map_meas = m_large_map_measurement_locs, m_large_map_measurements
        else:
            # Obtain the measurement locations and measurements that lie inside
            # the randomly selected rectangle (or patch)

            # 1. Find the limits of the region
            x_coord_max_large_map = max(m_large_map_measurement_locs[:, 0])
            x_coord_min_large_map = min(m_large_map_measurement_locs[:, 0])
            y_coord_max_large_map = max(m_large_map_measurement_locs[:, 1])
            y_coord_min_large_map = min(m_large_map_measurement_locs[:, 1])
            assert (
                x_coord_max_large_map - x_coord_min_large_map
            ) > self.patch_side_len[0] and (
                y_coord_max_large_map - y_coord_min_large_map
            ) > self.patch_side_len[
                1], "The side length of" "the patch map is greater" "than the region side length."

            # 2. Find limits of the region that must contain the bottom-left corner of the patch
            v_x_lim_blc = [
                x_coord_min_large_map,
                x_coord_max_large_map - self.patch_side_len[0]
            ]
            v_y_lim_blc = [
                y_coord_min_large_map,
                y_coord_max_large_map - self.patch_side_len[1]
            ]

            # 3. Select the bottom-left corner of the patch map
            if self.gridpoint_spacing is None:
                # select the origin of the patch map uniformly at random
                # across the region
                v_blc_coords = [
                    np.random.uniform(low=v_x_lim_blc[0], high=v_x_lim_blc[1]),
                    np.random.uniform(low=v_y_lim_blc[0], high=v_y_lim_blc[1])
                ]
                v_shift = v_blc_coords
            else:
                # select the origin of the patch map uniformly at random
                # across the grid points inside the aforementioned limits

                v_blc_gridpt_inds = [
                    np.random.randint(low=0,
                                      high=int(
                                          np.floor(
                                              (v_x_lim_blc[1]) /
                                              self.gridpoint_spacing[0])) + 1),
                    np.random.randint(low=0,
                                      high=int(
                                          np.floor(
                                              (v_y_lim_blc[1]) /
                                              self.gridpoint_spacing[1])) + 1)
                ]
                v_blc_coords = self.gridpoint_spacing * np.array(
                    v_blc_gridpt_inds) - self.gridpoint_spacing / 2

                v_shift = v_blc_coords + self.gridpoint_spacing / 2

            # 4. get the top-right corner of the patch map
            v_trc_coords = v_blc_coords + self.patch_side_len

            # 5. Get the measurement locations and measurements that lie inside the rectangle
            m_patch_map_meas_locs, m_patch_map_meas = get_coordinates_and_meas_inside_rectangle(
                v_blc_coords=v_blc_coords,
                v_trc_coords=v_trc_coords,
                m_large_map_measurement_locs=m_large_map_measurement_locs,
                m_large_map_meas=m_large_map_measurements)

            # 6. Translate the measurement locations so that the desired point goes to the origin
            m_patch_map_meas_locs = m_patch_map_meas_locs - np.array(
                [v_shift[0], v_shift[1], 0])
            if m_tx_loc is not None:
                m_tx_loc = m_tx_loc - np.array([v_shift[0], v_shift[1], 0])

        if self.z_coord is not None:
            m_patch_map_meas_locs[:, 2] = self.z_coord

        if self.gridpoint_spacing is not None:
            assert (
                self.patch_side_len is not None
            ), "The patch_side_len must be provided if the gridpoint_spacing is provided."
            # TODO: seems that the np.round below is not correct. Somebody must
            # have written it by mistake.
            grid = RectangularGrid(
                gridpoint_spacing=self.gridpoint_spacing,
                num_points_x=int(
                    np.round(self.patch_side_len[0] /
                             self.gridpoint_spacing[0])),
                num_points_y=int(
                    np.round(self.patch_side_len[1] /
                             self.gridpoint_spacing[1])),
                height=self.z_coord if self.z_coord is not None else 0)
        else:
            grid = None

        patch_map = Map(m_meas_locs_sf=m_patch_map_meas_locs,
                        m_meas_sf=m_patch_map_meas,
                        grid=grid,
                        grid_quantization_mode=self.grid_quantization_mode,
                        m_tx_loc=m_tx_loc)

        return patch_map

    def _get_data_from_file(self, ind_file):
        """ 
    
        Returns:

        If it returns data in standard form, it returns a dict with the
        following keys:
        
            - `m_locs_sf`: num_measurements x 3 matrix containing
                3D measurement locations

            - `m_meas_sf`: num_measurements x 1 matrix containing
                measurements at corresponding measurement locations.

            - `m_tx_locs`: num_tx x 3 matrix containing 3D transmitter
                locations if this information is available in the file. If not,
                it is None.

        If it returns data in grid form, it returns a dict with the
        following keys:

            - "lt_meas_gf": list of num_maps tensors, each one of size num_ch x
                num_rows x num_cos

            - "l_grid_spacing": list of num_maps grid spacing (numbers or tuples) in
                meters.

        """
        raise NotImplementedError

    def get_max_patch_side_len(self, verbose=False):
        """Returns the maximum side length of the patches that can be drawn from
        the data files. 

        """

        max_side_len = np.inf

        for ind_file in self.l_file_num:

            m_large_map_measurement_locs, m_large_map_measurements, m_tx_loc = self._get_data_from_file(
                ind_file)

            x_coord_max_large_map = max(m_large_map_measurement_locs[:, 0])
            x_coord_min_large_map = min(m_large_map_measurement_locs[:, 0])
            y_coord_max_large_map = max(m_large_map_measurement_locs[:, 1])
            y_coord_min_large_map = min(m_large_map_measurement_locs[:, 1])

            len_x = x_coord_max_large_map - x_coord_min_large_map
            len_y = y_coord_max_large_map - y_coord_min_large_map

            max_side_len = min(max_side_len, len_x, len_y)

            if verbose:
                print(
                    f"File {ind_file}: Size of the region: {len_x} x {len_y}")

        return max_side_len
