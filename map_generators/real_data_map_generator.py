import os
import pickle

import numpy as np
#from util.communications import dbm_to_natural, natural_to_dbm, dbm_to_db, db_to_natural, natural_to_db
import pandas as pd
from sklearn.utils import shuffle

from ..map_estimators.kriging_estimator import BatchKrigingEstimator
from ..map_generators.map import Map
from ..map_generators.map_generator import MapGenerator, MapGeneratorFromFiles
from ..map_generators.grid import RectangularGrid

building_threshold = -200  # Threshold in dBm to determine building locations


class RealDataMapGenerator(MapGeneratorFromFiles):
    """This class generates a map from real data.
    
    The provided folder contains RME dataset files, one for each index. 
    
    The indices must be in the range [0, 999]. Each index corresponds to a file
    named `{:03d}-*.pickle`. 

    The files are saved in pickle format and can be either in standard or grid
    form: 

    - Standard form: The file contains a dictionary with keys (choice of key
      names is legacy; consider updating to a more consistent naming):

        - "m_meas_loc": num_measurements x 3 matrix containing 3D measurement
          locations
        
        - "m_dB_power": num_measurements x num_ch matrix containing measurements
          at corresponding measurement locations.

        - "m_tx_loc" [optional]: num_tx x 3 matrix containing 3D transmitter
          locations if this information is available in the file. If not, it is
          None.


    - Grid form: The file contains a dictionary with keys:

        - "lt_meas_gf": list of num_maps tensors, each one of size num_ch x
          num_rows x num_cos

        - "l_grid_spacing": list of num_maps grid spacing (numbers or tuples) in
          meters.
     
       """

    def _get_data_from_file(self, ind_file):
        """ 
        If `ind_file` is not None, it returns all measurements and measurement
        locations in the file with index `ind_file` (see the docstring of
        self.__init__). If None, it returns all measurements and measurement
        locations in the only file in the folder `self.folder` whose name starts
        with `rme_dataset`.

        See the parent for more details.
        """

        def file_ind_to_path(ind):
            """If `ind` is not None, it returns the path to the file with index
            `ind`. If None, it returns the path to the only file that starts with
            `rme_dataset`."""

            start = "{:03d}-".format(ind) if ind is not None else "rme_dataset"
            l_file_names_this_ind = [
                file_name for file_name in self.l_file_names
                if file_name.startswith(start)
            ]
            assert len(
                l_file_names_this_ind
            ) == 1, f"No file in the folder {self.folder} starts with {start}. Either {ind} is out of bounds or the naming of the files is incorrect. Please check the docstring of {self.__class__.__name__}.__init__."

            return os.path.join(self.folder, l_file_names_this_ind[0])

        def read_from_file(ind):
            with open(file_ind_to_path(ind), 'rb') as f:
                d_data = pickle.load(f)
            return d_data

        d_data = read_from_file(ind_file)

        if "m_meas_loc" in d_data:
            # The file is in standard form
            return {
                "m_locs_sf": d_data["m_meas_loc"],
                "m_meas_sf": d_data["m_dB_power"],
                "m_tx_loc":
                d_data["m_tx_loc"] if "m_tx_loc" in d_data else None
            }

        elif "lt_meas_gf" in d_data:
            # The file is in grid form
            return {
                "lt_meas_gf": d_data["lt_meas_gf"],
                "l_grid_spacing": d_data["l_grid_spacing"]
            }
        else:
            raise ValueError(
                f"The file {file_ind_to_path(ind_file)} does not contain "
                "measurements in standard or grid form. Please check the "
                "docstring of this class for more details.")
