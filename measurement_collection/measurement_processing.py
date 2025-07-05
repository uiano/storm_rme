import math
import os
import shutil
import pickle
from math import cos, radians, sin, sqrt

import numpy as np
import pandas as pd
import pyproj

from gsim.gfigure import GFigure

# Ellipsoid parameters: semi major axis in metres, reciprocal flattening.
GRS80 = 6378137, 298.257222100882711
WGS84 = 6378137, 298.257223563


class Processor():

    conf_filename = "parameters.conf"
    subband_est_filename = "subband_estimates.csv"
    wb_est_filename = "wb_estimates.pickle"
    telemetry_filename = "telemetry.csv"
    rme_dataset_filename_base = "rme_dataset-{file_key}.pickle"

    def __init__(self, folder, thresh=1e-5):
        """
        
        Args:
            - thresh: all entries of m_abs_channel below this value are
                assumed to be not defined. 
        
        """
        if folder[-1] not in ["/", "\\"]:
            if "\\" in folder:
                folder += "\\"
            else:
                folder += "/"
        self.folder = folder
        self.params = self._load_parameters()
        self.thresh = thresh

    # ------- Processing steps -------

    def process_samples(self,
                        start_ind_point=None,
                        stop_ind_point=None,
                        matlab_args=None):
        """
        Reads the USRP samples in the files of the folder `samples` and creates
        a single CSV file with name "subband_estimates.csv" whose columns are:

        - ind_pt: index of the measurement point.

        - carrier_freq: carrier frequency of the measurement point.

        - db_snr: dB SNR of the measurement point.

        - channel_est (num_subcarriers columns) with the magnitude of the
          channel estimate at point ind_pt and carrier freq carrier_freq..

        TODO: determine the default stop_ind_point from the names of the files
        in the folder `samples`.
        
        """

        import matlab.engine

        # set parameters
        # If file `subband_est_filename` exists, issue a warning
        subband_est_filepath = self.rel_to_abs_path(self.subband_est_filename)
        if os.path.isfile(subband_est_filepath):
            print("#####  Warning: file %s already exists. #####\n"
                  "Estimates will be appended." % subband_est_filepath)

        # Start the MATLAB engine
        matlab_engine = matlab.engine.start_matlab()
        matlab_engine.addpath('measurement_collection/matlab/')

        # Call the MATLAB method
        # matlab_engine.get_subband_estimates(self.folder,
        #                                     start_ind_point,
        #                                     stop_ind_point,
        #                                     nargout=0)
        estr = f"""get_subband_estimates('{self.folder}'"""
        # TODO: use a function to generate the string
        if start_ind_point is not None:
            estr = estr + f",start_ind_point={start_ind_point}"
        if stop_ind_point is not None:
            estr = estr + f",stop_ind_point={stop_ind_point}"
        if matlab_args is not None:
            estr = estr + f",{matlab_args}"
        estr += ");"
        matlab_engine.eval(estr, nargout=0)

        # Close the MATLAB engine
        matlab_engine.quit()

    def process_subband_measurements(self, filter_calibration=True):
        """
        Reads the file containing the subband estimates produced by
        `process_samples` and constructs wideband estimates. 
        
        Produces a pickle file with a dict with fields:

            - 'v_wb_freq': num_freq vector with the frequencies of the wideband
              estimates

            - 'm_wb_est': num_points x num_freq matrix with the wideband
              estimates
            
            - 'm_counts': num_points x num_subcarriers matrix with the number of
              symbols used for each subband estimate. 

        """
        subband_bandwidth = self.params["samp_rate"]

        subband_est_filepath = self.rel_to_abs_path(self.subband_est_filename)
        d_data = self.read_meas_file(subband_est_filepath,
                                     read_matlab_csv_file=True)

        v_carrier_freq = d_data["carrier_freq"]
        v_ind_pts = d_data["ind_pts"].astype(int)
        m_abs_channel = np.abs(d_data["channel"])
        v_db_snr = d_data["db_snr"]

        if filter_calibration:
            m_abs_channel = self._compensate_tx_rx_filters(
                m_abs_channel, v_db_snr)

        v_wb_freq, m_wb_est, m_counts = self._get_wb_estimates(
            m_abs_channel, v_ind_pts, v_carrier_freq, subband_bandwidth)

        out_filepath = self.rel_to_abs_path(self.wb_est_filename)
        with open(out_filepath, "wb") as f:
            pickle.dump(
                {
                    'v_wb_freq': v_wb_freq,
                    'm_wb_est': m_wb_est,
                    'm_counts': m_counts
                }, f)

    def create_rme_dataset(self,
                           ind_pt_min=None,
                           ind_pt_max=None,
                           rotation=0,
                           default_translation=True,
                           translation=[0, 0, 0],
                           v_ylimits=None,
                           cartesian_type='UTM',
                           utm_grid_zone=32,
                           average_altitude=True,
                           take_dB=True,
                           v_tx_geo_cords=None):
        """
        Reads the file containing the wideband estimates produced by
        process_subband_measurements and creates a pickle file that stores a
        dict with fields:

        - 'm_meas_loc': num_pts x 3

        - 'm_tx_loc': 1 x 3 matrix with the location of the transmitter (optional,
          if v_tx_geo_cords is not None)

        If take_dB is True, 

        - 'm_dB_power': num_pts x 1

        else:
        
        - 'm_power': num_pts x 1

        The file name is rme_dataset-{key}.pickle, where `key` is the substring
        of the folder name after the first hyphen (-).

        Args:

        - `default_translation`: boolean. If True: the measurement locations are
          translated so that the minimum x and y coordinates are zero. 

        - 'translation': 3-vector. It is added to the measurement locations
          after rotation and the default translation.

        - `v_ylimits`: None or 2-vector. If a 2-vector, the measurements whose
          coordinates after rotation and translation are outside the interval
          [v_ylimits[0], v_ylimits[1]] are discarded.

        - `take_dB`: if True, the metric stored is the 10*log10 of the metrics
          in the input file.

        - ind_pt_min and ind_pt_max: [complete]

        -`cartesian_type`: a string that takes the value 'ECEF' (or
        'ECEF_pyproj') or 'UTM', which defines whether the converted Cartesian
        coordinates are in the Earth-centered, Earth-fixed coordinate system
        (ECEF) or Universal Transverse Mercator coordinate system (UTM) form.

        If cartesian_type='UTM' then `utm_grid_zone` must be provided.

        -`utm_grid_zone`: an utm zone number for a location indicated by
        longitude and latitude. It is the zone number of a location where
        measurements were collected. For the zone number see here
        https://www.dmap.co.uk/utmworld.htm.

        - `rotation_angle`: is a counter-clockwise angle measured with
        respect to the x-axis and it is in radians.

        - `average_altitude`: boolean. If True: every measurement coordinate
        has the same height = mean of `altitude` else: the height is not
        altered.           

        - `v_tx_geo_cords`: None or 3-vector. If not None, it is a vector whose
          entries are respectively the latitude, longitude, and altitude of the
          transmitter. 
    
        """

        # Read the wideband estimates
        wb_est_filepath = self.rel_to_abs_path(self.wb_est_filename)
        with open(wb_est_filepath, "rb") as f:
            d_wb_est = pickle.load(f)
        # Read the telemetry file
        telemetry_filepath = self.rel_to_abs_path(self.telemetry_filename)
        df_telemetry = pd.read_csv(telemetry_filepath, sep=',')

        # Select the subset of points indicated by ind_pt_min and ind_pt_max
        m_wb_est = d_wb_est["m_wb_est"]
        if ind_pt_min is None:
            ind_pt_min = 0
        if ind_pt_max is None:
            ind_pt_max = m_wb_est.shape[0]
        m_wb_est = m_wb_est[ind_pt_min:ind_pt_max, :]
        df_telemetry = df_telemetry.iloc[ind_pt_min:ind_pt_max, :]

        # Apply Parseval to obtain the power
        v_power = np.sum(m_wb_est**2, axis=1)

        # Get the measurement locations
        m_meas_loc, v_tx_loc = Processor.get_meas_locations(
            longitude=list(df_telemetry.start_long),
            latitude=list(df_telemetry.start_lat),
            altitude=list(df_telemetry.start_alt),
            rotation_angle=rotation,
            average_altitude=average_altitude,
            cartesian_type=cartesian_type,
            utm_grid_zone=utm_grid_zone,
            v_tx_geo_cords=v_tx_geo_cords)

        # Apply the default translation
        if default_translation:
            v_translation = np.array([
                np.min(m_meas_loc[:, 0], axis=0),
                np.min(m_meas_loc[:, 1], axis=0), 0
            ])
            m_meas_loc -= v_translation
            if v_tx_loc is not None:
                v_tx_loc -= v_translation

        # Apply the custom translation
        m_meas_loc += np.array(translation)
        if v_tx_loc is not None:
            v_tx_loc += np.array(translation)

        # Apply the y limits
        if v_ylimits is not None:
            v_ind = np.logical_and(m_meas_loc[:, 1] >= v_ylimits[0],
                                   m_meas_loc[:, 1] <= v_ylimits[1])
            m_meas_loc = m_meas_loc[v_ind, :]
            v_power = v_power[v_ind]

        m_power = v_power[:, None]

        # Save the dataset
        out_filepath = self.rel_to_abs_path(self.get_rme_dataset_filename())
        if not take_dB:
            d_out = {'m_power': m_power, 'm_meas_loc': m_meas_loc}
        else:
            d_out = {
                'm_dB_power': 10 * np.log10(m_power),
                'm_meas_loc': m_meas_loc
            }
        if v_tx_loc is not None:
            d_out['m_tx_loc'] = np.array([v_tx_loc])

        with open(out_filepath, "wb") as f:
            pickle.dump(d_out, f)

    @staticmethod
    def create_rme_dataset_from_gradiant_measurements(
            folder_in=None,
            folder_out=None,
            rotation_angle=0,
            v_shift=[0, 0, 0],
            cartesian_type='UTM',
            utm_zone_number=32,
            ind_first_file=0,
            num_col_groups=7,
            add_only_first_group_per_row=False):
        """
        Reads a folder containing excel files that contain measurements for
        multiple cells. It creates three folders, one per metric: 'rsrp',
        'rsrq', and 'rssi'. Each folder contains one pickle file for each cell.
        Each pickle file has name rme_dataset-{cell_id}.pickle and stores a dict
        with fields:

        - 'm_dB_power': num_pts x 1

        - 'm_meas_loc': num_pts x 3

        Args:

        - folder_in: path to the folder containing excel files that contain the
          measurements.

        - folder_out: path to the folder where the data sets will be saved.

        - ind_first_file: the files in each of the generated folder are sorted
          by decreasing size and numbered starting from ind_first_file.
                  
        - num_col_groups: number of groups of columns in each Excel files (one
          per cell)

        - add_only_first_group_per_row: boolean. If True: if multiple groups with the same pcid are found in a row, only the first one is added to the dataset. Else: all the groups are added.

        See create_rme_dataset for the remaining arguments.
        
        """
        assert folder_in != None, "folder_in must be provided"
        assert folder_out != None, "folder_out must be provided"

        l_metrics = ['rsrq', 'rsrp', 'rssi']
        # create a folder for each metric
        for metric in l_metrics:
            folder_metric = os.path.join(folder_out, metric)

            # delete the folder if it exists
            if os.path.exists(folder_metric):
                shutil.rmtree(folder_metric)

            os.makedirs(folder_metric, exist_ok=True)

        # TODO:

        # Read and concatenate the excel files

        # Process each row. For each row, process each group of columns. Create
        # a dict whose keys are the cell ids and whose values are vectors [rsrq,
        # rsrp, rssi].

        # Write the files
        # - - - - - - - - - - - -

        def split_into_groups(row):
            """Returns a list of dicts with keys: earfcn, pcid, tech, type, rsrq, rsrp, rssi, sinr 
            There is one list per cell."""

            def ind_key_to_col_name(ind, key):
                if ind == 0 and key == 'pcid':
                    return "serving_cell_cid"
                base_col_name = "serving_cell_{key}" if ind == 0 else "neighbour_cell_{key}_{ind}"
                return base_col_name.format(ind=ind, key=key)

            l_keys = ["earfcn", "pcid", "tech", "rsrq", "rsrp", "rssi", "sinr"]

            ld_groups = []
            for ind_col in range(num_col_groups):
                try:
                    d_now = {
                        key: row[ind_key_to_col_name(ind_col, key)]
                        for key in l_keys
                    }
                except KeyError:
                    # This happens when the column does not exist
                    continue
                if np.isnan(d_now["pcid"]):
                    continue
                ld_groups.append(d_now)

            return ld_groups

        def add_group_to_store(v_meas_loc, d_group):
            cell_id = int(d_group['pcid'])

            if cell_id not in d_store.keys():

                d_store[cell_id] = dict()
                for metric in l_metrics:
                    d_store[cell_id][metric] = {
                        'l_power': [],  #np.zeros((0, )),
                        'lv_meas_loc': []  #np.zeros((0, 3))
                    }

            # Concatenate the info in d_group to the vectors in d_store[cell_id][metric]
            for metric in l_metrics:
                d_store[cell_id][metric]['l_power'] += [d_group[metric]]
                d_store[cell_id][metric]['lv_meas_loc'] += [v_meas_loc]

        def write_d_store_to_files(d_store, folder_out):

            # loop through cell_ids in d_store
            for cell_id in d_store.keys():

                # loop through metrics
                for metric in l_metrics:
                    # create a file for each cell_id in metric folder
                    out_filepath = os.path.join(
                        folder_out, metric, f'rme_dataset-{cell_id}.pickle')
                    with open(out_filepath, "wb") as f:
                        pickle.dump(
                            {
                                'm_dB_power':
                                np.array(d_store[cell_id][metric]['l_power'])[:,None],
                                'm_meas_loc': # Fix this
                                np.array(
                                    d_store[cell_id][metric]['lv_meas_loc'])
                            }, f)

            pass

        def rename_pickle_files(metric):
            # rename the pickle files in folder_out/metric. The files are sorted
            # by decreasing size and their index is appended at the beginning.
            # This is to follow the format specified in real_data_map_generator.py
            #
            # Example: rme_dataset-132.pickle --> 023-rme_dataset-132.pickle

            folder_metric = os.path.join(folder_out, metric)
            l_files = os.listdir(folder_metric)

            # Keep only the pickle files
            l_files = [file for file in l_files if file.endswith(".pickle")]

            # Sort the files by decreasing size
            l_files.sort(
                key=lambda x: os.path.getsize(os.path.join(folder_metric, x)),
                reverse=True)

            # Rename the files
            for ind_file, file in enumerate(l_files):
                ind_file += ind_first_file
                os.rename(
                    os.path.join(folder_metric, file),
                    os.path.join(folder_metric, f"{ind_file:03d}-{file}"))

        # Collect all the rows in all excel files
        l_rows = []
        for file in os.listdir(folder_in):

            filename = folder_in + file
            print(filename)

            # there are some files that cannot be read, i.e. 0 bytes.
            try:
                df = pd.read_excel(filename)
            except ValueError:
                print(
                    "Skipping file ", filename,
                    " because its format cannot be determined by pd.read_excel."
                )
                continue

            # process every row
            for row in df.iloc:
                l_rows.append(row)

        # Convert df into a list of lists of dicts, one list of dicts per row
        lld_rows = [split_into_groups(row) for row in l_rows]
        # The corresponding locations:
        m_meas_loc = np.array(
            [[row['longitude'], row['latitude'], row['altitude']]
             for row in l_rows])
        # Convert the geodesic coordinates to cartesian coordinates
        m_meas_loc = Processor.get_meas_locations(
            longitude=m_meas_loc[:, 0],
            latitude=m_meas_loc[:, 1],
            altitude=m_meas_loc[:, 2],
            rotation_angle=rotation_angle,
            cartesian_type=cartesian_type,
            utm_grid_zone=utm_zone_number)[0]

        m_meas_loc += np.array(v_shift)

        d_store = dict()
        for ind_row, ld_row in enumerate(lld_rows):
            l_unique_pcid_per_ld_rows = []
            for d_group in ld_row:
                # add only the first group for the repeated pcid in the same row
                if add_only_first_group_per_row:
                    if d_group['pcid'] in l_unique_pcid_per_ld_rows:
                        continue
                    else:
                        l_unique_pcid_per_ld_rows.append(d_group['pcid'])

                add_group_to_store(m_meas_loc[ind_row], d_group)

        # Write d_store into files
        write_d_store_to_files(d_store, folder_out)

        # Rename the files
        for metric in l_metrics:
            rename_pickle_files(metric)

        print('Ending create_rme_dataset')

    # ------- IO functions --------

    @staticmethod
    def read_meas_file(file_name, read_matlab_csv_file=False):
        """
        Args:
            - file_name: string
                Name of the file to read.
            - read_matlab_csv_file: bool. True if the file was generated by matlab. Else, False.

        The format of the file is as follows:

            - 1 row per measurement. N rows. 

            - Col 0 --> file_ind_pt

            - Col 1 --> carrier frequency (central frequency of the subband)

            - Col 2 --> dB SNR of the measurement

            - Col 3 --> number of selected symbols used for channel estimation

            - Cols 4:M-1 --> channel coefficients
        
        The output is a dict with fields:

            - 'ind_pts' --> N vector
            
            - 'carrier_freq' --> N vector

            - 'db_snr' --> N vector

            - 'num_selected_symbols' --> N vector

            - 'channel' --> N x M matrix


        """
        if not read_matlab_csv_file:
            # read a csv file generated by python in GnuRadio
            data = pd.read_csv(file_name, sep=',', header=None)
        else:
            # Read the file as a string from a csv file generated by matlab
            data = pd.read_csv(file_name, dtype=str, header=None)
            # Convert the string representations to complex numbers
            # data = data.applymap(lambda x: complex(x.replace('i', 'j')))

            # # Convert the string representations obtained from
            # matlab to complex numbers that python can understand
            for column in data.columns:
                data[column] = np.array(
                    [complex(x.replace('i', 'j')) for x in data[column]])

        m_channel = np.vectorize(complex)(data.iloc[:, 4:])
        v_file_ind_pt = abs(data.iloc[:, 0].to_numpy())
        v_carrier_freq = abs(data.iloc[:, 1].to_numpy())
        v_db_snr = np.real(data.iloc[:, 2].to_numpy())
        v_num_sel_symbols = abs(data.iloc[:, 3].to_numpy())

        return {
            'ind_pts': v_file_ind_pt,
            'carrier_freq': v_carrier_freq,
            'channel': m_channel,
            'db_snr': v_db_snr,
            'num_selected_symbols': v_num_sel_symbols,
        }

    def read_wb_est(self):
        wb_est_filepath = self.rel_to_abs_path(self.wb_est_filename)
        with open(wb_est_filepath, "rb") as f:
            d_out = pickle.load(f)
        return d_out

        return v_wb_freq, m_wb_est, m_counts

    def read_nb_est(self):
        d_data = Processor.read_meas_file(self.rel_to_abs_path(
            self.subband_est_filename),
                                          read_matlab_csv_file=True)
        d_data["m_nb_est"] = np.abs(d_data["channel"])
        return d_data

    def read_rme_dataset(self):
        rme_dataset_filepath = self.rel_to_abs_path(
            self.get_rme_dataset_filename())
        with open(rme_dataset_filepath, "rb") as f:
            d_out = pickle.load(f)
        return d_out

    # ------- Estimation functions --------
    def _get_wb_estimates(self,
                          m_abs_channel,
                          v_ind_pts,
                          v_carrier_freq,
                          subband_bandwidth,
                          gaussian_smoothing_sigma=-1.):
        """
        Returns a num_points x num_subcarriers matrix of wideband estimates. 

        Args:
            m_abs_channel: num_measurements x num_subcarriers matrix of channel
            magnitude measurements. 

            v_ind_pts: num_measurements vector of indices of the measurement
            points. 

            v_carrier_freq: num_measurements vector of carrier frequencies.

            gaussian_smoothing_sigma: sigma of the gaussian smoothing filter. If
            -1, no smoothing is applied.

        Returns:
            v_freq_axis: num_subcarriers vector of frequencies.
        
            m_wb_est: num_points x num_subcarriers matrix of wideband

            m_counts: num_points x num_subcarriers matrix of counts.

        """
        num_points = int(np.max(v_ind_pts)) + 1
        num_measurements, num_subcarriers = m_abs_channel.shape
        subcarrier_spacing = subband_bandwidth / num_subcarriers
        first_carrier_freq = np.min(v_carrier_freq)
        last_carrier_freq = np.max(v_carrier_freq)

        def carrier_freq_to_ind_1st_col(carrier_freq):
            ind_col = (carrier_freq - first_carrier_freq) / subcarrier_spacing
            err = ind_col - int(np.round(ind_col))
            if err > 1e-3:
                print("Warning: err = ", err)
            return int(np.round(ind_col))

        m_wb_est = np.zeros(
            (num_points,
             carrier_freq_to_ind_1st_col(last_carrier_freq) + num_subcarriers))
        m_counts = np.zeros(
            (num_points,
             carrier_freq_to_ind_1st_col(last_carrier_freq) + num_subcarriers))

        for ind_measurement in range(num_measurements):
            ind_pt = v_ind_pts[ind_measurement]
            ind_col = carrier_freq_to_ind_1st_col(
                v_carrier_freq[ind_measurement])
            m_wb_est[ind_pt, ind_col:ind_col +
                     num_subcarriers] += m_abs_channel[ind_measurement]
            m_counts[ind_pt, ind_col:ind_col +
                     num_subcarriers] += (m_abs_channel[ind_measurement]
                                          > self.thresh)
        m_wb_est /= np.maximum(m_counts, 1)

        ######## Gaussian smoothing
        from scipy.ndimage import gaussian_filter1d

        def gaussian_smoothing(data, sigma):
            smoothed_signal = gaussian_filter1d(data, sigma)
            return smoothed_signal

        if gaussian_smoothing_sigma > 0:
            m_wb_est_smooth = np.zeros(m_wb_est.shape)
            for ind_pt in range(num_points):
                m_wb_est_smooth[ind_pt, :] = gaussian_smoothing(
                    m_wb_est[ind_pt, :], sigma=gaussian_smoothing_sigma)
            m_wb_est = m_wb_est_smooth

        ###################
        # # Savitzky-Golay smoothing
        # from scipy.signal import savgol_filter

        # def savitzky_golay_smoothing(data, window_size, order):
        #     smoothed_signal = savgol_filter(data, window_size, order)
        #     return smoothed_signal

        # m_wb_est_smooth = np.zeros(m_wb_est.shape)
        # for ind_pt in range(num_points):
        #     m_wb_est_smooth[ind_pt, :] = savitzky_golay_smoothing(
        #         m_wb_est[ind_pt, :], window_size=50, order=2)
        # m_wb_est = m_wb_est_smooth
        ###################

        v_wb_freq = np.arange(
            first_carrier_freq - num_subcarriers / 2 * subcarrier_spacing,
            last_carrier_freq + (num_subcarriers / 2) * subcarrier_spacing,
            subcarrier_spacing)
        return v_wb_freq, m_wb_est, m_counts

    def _compensate_tx_rx_filters(self, m_abs_channel, v_db_snr, num_best=100):
        """ 
        Compensates the effect of the TX and RX filters on the channel
        estimates.

        Args:
            - m_abs_channel: matrix with the magnitude of the channel
                estimates. Each row contains the measurements of a subband for
                one of the points. Each column is a subcarrier.

            - v_db_snr: vector whose n-th entry contains the SNR of
                m_abs_channel[n,:].

            - num_best: number of subbands to use for the estimation of the
                transfer function of the TX and RX filters combined.

        Returns:
            - m_abs_channel: matrix with the magnitude of the channel
                estimates where the effects of the tx. and rx. filters is
                compensated. 

        """

        # Normalize the subband estimates by their medians
        m_abs_channel_normalized = m_abs_channel / np.median(m_abs_channel,
                                                             axis=1)[:, None]
        v_var = np.var(m_abs_channel_normalized, axis=1)
        v_ind = np.argsort(v_var)[:num_best]
        m_abs_channel_normalized = m_abs_channel_normalized[v_ind, :]
        v_db_snr = v_db_snr[v_ind]

        # Estimation of the transfer function of the TX and RX filters combined
        v_freq_support = np.sum(m_abs_channel_normalized >= self.thresh,
                                axis=0) > 0
        v_snr = 10**(v_db_snr / 10)
        v_weights = v_snr / np.sum(v_snr)
        m_weights = v_weights[:, None] @ v_freq_support[None, :]
        v_transfer_func = np.sum(m_abs_channel_normalized * m_weights, axis=0)

        # Debugging
        if False:
            G = GFigure(num_subplot_rows=2)
            G.next_subplot(yaxis=m_abs_channel, title="All estimates")
            G.next_subplot(yaxis=m_abs_channel_normalized,
                           title="Normalized best estimates")
            G.next_subplot(yaxis=v_transfer_func,
                           title="Transfer function estimate")
            G.plot()
            G.show()

        # Compensate the effect of the TX and RX filters
        v_transfer_func[v_freq_support != True] = 1
        return m_abs_channel / v_transfer_func[None, :]

    # ------- Utility functions -------

    @staticmethod
    def get_meas_locations(longitude,
                           latitude,
                           altitude,
                           cartesian_type='ECEF',
                           utm_grid_zone=None,
                           average_altitude=True,
                           rotation_angle=0,
                           v_tx_geo_cords=None):
        """
        Args:
            -`longitude`, `latitude`, and `altitude`:  vectors of len
            num_measurements -`cartesian_type`: a string that takes the value
            'ECEF' (or 'ECEF_pyproj') or 'UTM',
                which defines whether the converted cartesian coordinate is in
                the Earth-centered, Earth-fixed coordinate system (ECEF) or
                Universal Transverse Mercator coordinate system (UTM) form.

                If cartesian_type='UTM' then `utm_grid_zone` must be provided.

            -`utm_grid_zone`: an utm zone number for a location indicated by
            longitude and latitude.
                            It is the zone number of a location where
                            measurements were collected.
                             For the zone number see here
                             https://www.dmap.co.uk/utmworld.htm.

            `rotation_angle`: is a counter-clockwise angle measured with respect
            to
                    x-axis and is in radian.

            - `average_altitude`: boolean
                        if True: every measurement coordinate has the same
                        height = mean of `altitude` else: the height is not
                        altered.

         - `v_tx_geo_cords`: None or 3-vector. If not None, it is a vector whose
          entries are respectively the latitude, longitude, and altitude of the
          transmitter. 
    
        Returns:
            -`m_measurement_locs`: num_measurements x 3 matrix whose i-th row
            contains a 3D coordinate

            - `v_tx_loc`: if `v_tx_geo_cords` is not None, this is a 3-vector
              with the location of the transmitter. Else, it is None. 
        """

        indic_nans = np.isnan(longitude) | np.isnan(latitude) | np.isnan(
            altitude)
        # The locations with NaNs are removed now and restored later
        longitude = np.array(longitude)[~indic_nans]
        latitude = np.array(latitude)[~indic_nans]
        altitude = np.array(altitude)[~indic_nans]

        # get cartesian coordinates
        l_x_coord, l_y_coord = geodesic_to_cartesian_coordinate(
            Longitude=longitude,
            Latitude=latitude,
            altitude=altitude,
            cartesian_type=cartesian_type,
            utm_grid_zone=utm_grid_zone)

        # set origin of the map to be (min_x, min_y)
        v_origin_coord = [min(l_x_coord), min(l_y_coord)]

        # translate and rotate
        l_x_transformed_coord, l_y_transformed_coord = translate_and_rotate(
            l_x=l_x_coord,
            l_y=l_y_coord,
            rotation_angle=rotation_angle,
            v_origin=v_origin_coord)

        # get z coordinate
        if average_altitude:
            large_map_height = np.mean(altitude)
            l_z_transformed_coord = list(
                np.ones(len(l_x_transformed_coord)) * large_map_height)
        else:
            l_z_transformed_coord = list(altitude)

        if v_tx_geo_cords is not None:
            l_tx_x_loc, l_tx_y_loc = geodesic_to_cartesian_coordinate(
                Latitude=[v_tx_geo_cords[0]],
                Longitude=[v_tx_geo_cords[1]],
                altitude=[v_tx_geo_cords[2]],
                cartesian_type=cartesian_type,
                utm_grid_zone=utm_grid_zone)

            l_tx_x_transformed_coord, l_tx_y_transformed_coord = translate_and_rotate(
                l_x=l_tx_x_loc,
                l_y=l_tx_y_loc,
                rotation_angle=rotation_angle,
                v_origin=v_origin_coord)

            v_tx_loc = [
                l_tx_x_transformed_coord[0], l_tx_y_transformed_coord[0],
                v_tx_geo_cords[2]
            ]
        else:
            v_tx_loc = None

        def restore_nans(l_coord):
            v_coord = np.array([np.nan] * len(indic_nans))
            v_coord[~indic_nans] = l_coord
            return v_coord

        # Restore nans
        l_x_transformed_coord = restore_nans(l_x_transformed_coord)
        l_y_transformed_coord = restore_nans(l_y_transformed_coord)
        l_z_transformed_coord = restore_nans(l_z_transformed_coord)

        # num_measurements x 3
        m_measurement_locs = np.array([
            l_x_transformed_coord, l_y_transformed_coord, l_z_transformed_coord
        ]).T

        return m_measurement_locs, v_tx_loc

    def get_rme_dataset_filename(self):
        # Get the folder name from the path self.folder
        if "\\" in self.folder:
            folder_name = self.folder.split("\\")[-2]
        else:
            folder_name = self.folder.split("/")[-2]
        if "-" not in folder_name:
            raise ValueError("Invalid folder name: " + folder_name +
                             ". It must contain a hyphen (-).")
        # Find the position of the first hyphen in `folder_name`
        ind_hyphen = folder_name.index("-")
        key = folder_name[ind_hyphen + 1:]

        return self.rme_dataset_filename_base.format(file_key=key)

    def rel_to_abs_path(self, rel_path):
        """
        Returns the absolute path of the relative path.

        Args:
            rel_path: relative path.

        Returns:
            abs_path: absolute path.
        """
        return os.path.join(self.folder, rel_path)

    def _load_parameters(self):
        """
        Loads the parameters from the parameters.txt file.

        Returns:
            dict_params: dictionary of parameters.
        """
        dict_params = {}
        conf_filepath = self.rel_to_abs_path(self.conf_filename)
        with open(conf_filepath, "r") as f:
            for line in f:
                if "=" in line:  # ignore lines that don't have '='
                    var_name, value = line.split("=")
                    dict_params[var_name] = eval(value)
        return dict_params

    def plot_wb_estimates(self,
                          v_wb_freq,
                          m_wb_est,
                          mode='contour3D',
                          num_max_pts=None):
        """
        Plots the wideband estimates.        

        Args:
            v_wb_freq: num_subcarriers vector of frequencies.

            m_wb_est: num_points x num_subcarriers matrix of wideband

            mode: '2D', 'imshow', 'contour3d', or 'surface'.

        """

        G = GFigure(num_subplot_rows=1)
        num_max_pts = np.minimum(
            num_max_pts,
            len(m_wb_est)) if num_max_pts is not None else len(m_wb_est)
        m_wb_est = m_wb_est[:num_max_pts]

        xaxis = v_wb_freq / 1e6
        if mode == '2D':
            G = GFigure(xlabel='Freq. [MHz]', ylabel='Averaged Magnitude')
            for ind_pt in range(len(m_wb_est)):
                G.add_curve(xaxis=xaxis,
                            yaxis=m_wb_est[ind_pt],
                            legend="ind_pt = " + str(ind_pt))

        else:
            G.next_subplot(xaxis=xaxis,
                           yaxis=np.arange(0, m_wb_est.shape[0]),
                           zaxis=m_wb_est,
                           xlabel='Freq. [MHz]',
                           ylabel='Point Index',
                           mode=mode)

        return G

    def plot_nb_estimates(self, d_nb_est, ind_pts=[0]):
        m_nb_est = d_nb_est["m_nb_est"]

        G = GFigure(xlabel='Freq. [MHz]', ylabel='Magnitude')

        def plot_estimates(m_nb_est):
            for ind_pt in ind_pts:
                for ind_carrier, carrier_freq in enumerate(
                        np.unique(d_nb_est["carrier_freq"])):
                    G.add_curve(
                        xaxis=Processor.get_subband_freq_axis(
                            carrier_freq, m_nb_est.shape[1],
                            self.params["samp_rate"]) / 1e6,
                        yaxis=m_nb_est[d_nb_est["ind_pts"] == ind_pt]
                        [ind_carrier],
                    )

        G.select_subplot(0, title="Uncalibrated estimates")
        plot_estimates(m_nb_est)

        G.select_subplot(1, title="Calibrated estimates")
        plot_estimates(
            self._compensate_tx_rx_filters(m_nb_est, d_nb_est["db_snr"]))

        return G

    def plot_num_sel_symb(self, d_nb_est, ind_pts=[0]):
        v_num_sel_symb = d_nb_est["num_selected_symbols"]
        G = GFigure(xlabel='Freq. (MHz)',
                    ylabel='No. of Selected Symbols for ch. Est.')

        for ind_pt in ind_pts:
            v_num_sel_sym_per_ind_pt = v_num_sel_symb[d_nb_est["ind_pts"] ==
                                                      ind_pt]

            G.add_curve(
                xaxis=(self.params["freq_start"] +
                       np.arange(0, self.params["num_carrier_freq"]) *
                       self.params["freq_step"]) /
                1e6,  #np.arange(0, len(v_num_sel_sym_per_ind_pt)),
                yaxis=v_num_sel_sym_per_ind_pt,
                legend="ind_pt = " + str(ind_pt),
            )

        return G

    def plot_rme_dataset_vs_distance(self, l_change_colors=[]):
        """
        Returns a GFigure object with a plot of power vs. distance to 
         
         - the first tx if the dataset contains the tx location
         - the first measurement point otherwise

        Args:
            l_change_colors: list of indices of the points where the color of
            the curve changes.
        """

        d_rme_dataset = self.read_rme_dataset()
        v_dB_power = d_rme_dataset["m_dB_power"][:, 0]
        m_meas_loc = d_rme_dataset["m_meas_loc"]

        if "m_tx_loc" in d_rme_dataset.keys():
            v_ref_loc = d_rme_dataset["m_tx_loc"][0]
        else:
            v_ref_loc = m_meas_loc[0]

        v_dist = np.linalg.norm(m_meas_loc - v_ref_loc, axis=1)

        def split_array(v, l_ind):
            return [v[i:j] for i, j in zip([0] + l_ind, l_ind + [None])]

        l_v_power = split_array(v_dB_power, l_change_colors)
        l_v_dist = split_array(v_dist, l_change_colors)
        l_v_inds = split_array(np.arange(len(v_dist)), l_change_colors)

        G = GFigure(xlabel='Distance [m]', ylabel='Power [dBX]')
        for ind_segment in range(len(l_v_power)):
            G.add_curve(
                xaxis=l_v_dist[ind_segment],
                yaxis=l_v_power[ind_segment],
                styles=['.'],
            )

        # Plot the distance vs. the point index
        G.next_subplot(xlabel='Point Index', ylabel='Distance [m]')
        for ind_segment in range(len(l_v_inds)):
            G.add_curve(
                xaxis=l_v_inds[ind_segment],
                yaxis=l_v_dist[ind_segment],
            )

        return G

    @staticmethod
    def median_decimation(v_wb_freq, m_wb_est, decimation_step):
        """
        Applies median_decimation_vec to every row of m_wb_est.
        
        """

        def median_decimation_vec(v, decimation_step):
            return np.median(v.reshape(-1, decimation_step), axis=1)

        num_freq_pts_out = int(np.floor(len(v_wb_freq) / decimation_step))
        v_wb_freq = v_wb_freq[:num_freq_pts_out * decimation_step]
        m_wb_est = m_wb_est[:, :num_freq_pts_out * decimation_step]

        v_wb_freq_out = v_wb_freq[int(np.round(decimation_step /
                                               2))::decimation_step]

        m_wb_est_out = np.apply_along_axis(
            lambda x: median_decimation_vec(x, decimation_step),
            axis=1,
            arr=m_wb_est)

        return v_wb_freq_out, m_wb_est_out

    @staticmethod
    def get_subband_freq_axis(carrier_freq, num_subcarriers,
                              subband_bandwidth):

        return carrier_freq + np.arange(
            -num_subcarriers / 2,
            num_subcarriers / 2) * subband_bandwidth / num_subcarriers

    def freq_to_block_inds(self, freq):
        """
        Returns a list of the indices of the blocks that contain 
        frequency `freq`.
        """
        bw = self.params["samp_rate"]
        v_freqs = self.params["freq_start"] + np.arange(
            self.params["num_carrier_freq"]) * self.params["freq_step"]
        return np.where(np.abs(freq - v_freqs) <= bw / 2)[0]


# ------- Coordinate Transformation Methods ------------------------


def translate_and_rotate(l_x, l_y, rotation_angle, v_origin):
    """
    This method performs the transformation (translation and then rotation)
    of the input Cartesian coordinates. The point `v_origin` becomes the (0,0).

    Args:
            `rotation_angle`: is a counter-clockwise angle measured with respect to
                    x-axis and is in radian.

    Returns: the lists containing the transformed form of the input coordinates
    """

    def rotate(x_coord, y_coord, rotation_angle):
        """
        Args:
            `rotation_angle`: is a counter clockwise angle measured with respect to
                    x-axis and is in radian.

        Returns:
            x_rotated, y_rotated coordinates rotated by `rotation_angle`
        """
        # translate
        # x_coord = x_coord - v_origin[0]
        # y_coord = y_coord - v_origin[1]

        # rotate
        x_rotated = x_coord * math.cos(rotation_angle) \
                    - y_coord * math.sin(rotation_angle)
        y_rotated = x_coord * math.sin(rotation_angle) \
                    + y_coord * math.cos(rotation_angle)

        return x_rotated, y_rotated

    l_x_transformed_coord = []
    l_y_transformed_coord = []

    for x_coord, y_coord in zip(l_x, l_y):
        # translate
        x_coord = x_coord - v_origin[0]
        y_coord = y_coord - v_origin[1]

        # rotate
        x_rotated, y_rotated = rotate(x_coord, y_coord, rotation_angle)
        # print("rotated", x_rotated, y_rotated)
        # x_rotated, y_rotated =x_coord, y_coord

        l_x_transformed_coord.append(x_rotated)
        l_y_transformed_coord.append(y_rotated)

    return l_x_transformed_coord, l_y_transformed_coord


def geodesic_to_cartesian_coordinate(Longitude,
                                     Latitude,
                                     altitude=None,
                                     cartesian_type='ECEF',
                                     utm_grid_zone=None):
    """
    For a geodesic to cartesian coordinate conversion,
    geodesic data follows the WGS84 data format.

    Args:
        `cartesian_type`: a string that takes the value 'ECEF'(or 'ECEF_pyproj') or 'UTM'
            which defines whether the converted cartesian coordinate
            is in the Earth-centered, Earth-fixed coordinate system (ECEF)
            or Universal Transverse Mercator coordinate system (UTM) form.

            If cartesian_type='UTM' then `utm_grid_zone` must be provided.

        -`utm_grid_zone`: an utm zone number for a location indicated by longitude and latitude.
                         For the zone number see here https://www.dmap.co.uk/utmworld.htm.

    Returns:
        `l_x`: a list containing x-coordinates of measurement locations
        `l_y`: a list containing y-coordinates of measurement locations

    """
    l_x = []
    l_y = []
    # l_rssi = []
    if altitude is None:
        altitude = np.zeros(len(Longitude), )

    if cartesian_type == "UTM":
        assert utm_grid_zone

    for long, lat, alt in zip(Longitude, Latitude, altitude):
        # print(f"{long}\t{lat}\t{rssi}")

        if cartesian_type == 'UTM':
            # convert GPS to UTM coordinates
            # x, y, _, _ = utm.from_latlon(long, lat, force_zone_number=29)

            project_to_utm = pyproj.Proj(proj='utm',
                                         zone=utm_grid_zone,
                                         ellps='WGS84',
                                         preserve_units=False)
            x, y = project_to_utm(long, lat, errcheck=True)

        elif cartesian_type == 'ECEF':
            # convert GPS to ECEF coordinates
            x, y, z = geodetic_to_geocentric(ellipsoid=WGS84,
                                             latitude=lat,
                                             longitude=long,
                                             height=alt)
        elif cartesian_type == 'ECEF_pyproj':
            x, y, z = geodetic_to_ecef_pyproj(latitude=lat,
                                              longitude=long,
                                              altitude=alt)
        else:
            raise ValueError

        x_transformed = x
        y_transformed = y

        l_x.append(x_transformed)
        l_y.append(y_transformed)
        # l_rssi.append(rssi)
        # v_point = [x - x_origin, y - y_origin]

    return l_x, l_y


def geodetic_to_geocentric(ellipsoid, latitude, longitude, height):
    """
    Returns geocentric (Cartesian) Coordinates x, y, z corresponding to
    the geodetic coordinates given by latitude and longitude (in
    degrees) and height above ellipsoid. The ellipsoid must be
    specified by a pair (semi-major axis, reciprocal flattening).

    """

    phi_lat = radians(latitude)
    lambda_long = radians(longitude)
    sin_phi = sin(phi_lat)
    semi_major_a, reciprocal_flattening_f = ellipsoid  # semi-major axis, reciprocal flattening
    eccentricity_e2 = 1 - (
        1 - 1 / reciprocal_flattening_f)**2  # eccentricity squared
    n = semi_major_a / sqrt(
        1 - eccentricity_e2 * sin_phi**2)  # prime vertical radius
    r = (n + height) * cos(phi_lat)  # perpendicular distance from z axis
    x = r * cos(lambda_long)
    y = r * sin(lambda_long)
    z = (n * (1 - eccentricity_e2) + height) * sin_phi
    return x, y, z


def geodetic_to_ecef_pyproj(latitude, longitude, altitude):
    """
    Returns geocentric (Cartesian) Coordinates x, y, z corresponding to
    the geodetic coordinates
    """
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    x, y, z = pyproj.transform(lla,
                               ecef,
                               longitude,
                               latitude,
                               altitude,
                               radians=False)

    return x, y, z
