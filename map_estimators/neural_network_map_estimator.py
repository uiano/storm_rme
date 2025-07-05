import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf
import tensorflow_probability as tfp
from IPython.core.debugger import set_trace
from keras import Model
from keras.layers import (Conv2D, Conv2DTranspose, Dense, Flatten, Input,
                          MaxPool2D, Reshape, UpSampling2D)
from scipy.spatial.distance import euclidean
from scipy.stats import norm

from ..map_estimators.map_estimator import MapEstimator
from ..map_generators.map import Map
from ..utilities import empty_array, np


class NeuralNetworkMapEstimator(MapEstimator):
    """
    Args:

    -  estimator is an object of class NeuralNetworkEstimator

    - `l_aux_map_estimators`: if not None, then the estimates obtained with the
      estimators on this list are appended to the input of the DNN estimator.
    """
    name_on_figs = "NeuralNetworkMapEstimator"
    estimation_fun_type = 's2g'

    def __init__(self, estimator=None, l_aux_map_estimators=None, **kwargs):

        # if estimator is None:
        #     self.estimator = FullyConvolutionalNeuralNetwork.\
        #         get_fully_conv_nn_arch(nn_arch_id=nn_arch_id)
        # else:
        assert estimator is not None
        self.estimator = estimator
        self.l_aux_map_estimators = l_aux_map_estimators
        super().__init__(**kwargs)
        if self.l_aux_map_estimators is not None:
            assert isinstance(self.l_aux_map_estimators, list)

    @property
    def m_building_meta_data_grid(self):
        """
        Returns:
        `_m_building_meta_data_grid`, Ny x Nx matrix whose (i,j) entry is 1 if the
        grid point is inside the building.
        """
        # if self.m_meta_data is None:
        #     print('Please provide the building meta_data.')
        # self._m_building_meta_data_grid = np.zeros((self.grid.num_points_y,
        #                                             self.grid.num_points_x))
        if self.m_building_meta_data is None and self._m_building_meta_data_grid is None:
            self._m_building_meta_data_grid = np.zeros(
                (self.grid.num_points_y, self.grid.num_points_x))

        if self._m_building_meta_data_grid is None and self.m_building_meta_data is not None:
            self._m_building_meta_data_grid = np.zeros(
                (self.grid.num_points_y, self.grid.num_points_x))
            for v_building_meta_data in self.m_building_meta_data:
                v_meta_data_inds = self.grid.nearest_gridpoint_inds(
                    v_building_meta_data)
                self._m_building_meta_data_grid[v_meta_data_inds[0],
                                                v_meta_data_inds[1]] = 1

        return self._m_building_meta_data_grid

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, grid):
        self._grid = grid
        if self.l_aux_map_estimators is not None:
            for aux_map_estimator in self.l_aux_map_estimators:
                aux_map_estimator.grid = grid

    def estimate_s2g(
        self,
        measurement_loc=None,
        measurements=None,
        building_meta_data=None,
        #  test_loc=None,
    ):
        """
        See the docstring of the parent class.
        """

        d_map_estimate = self.estimate_metric_per_channel(
            measurement_loc.T,
            measurements.T,
            building_meta_data=building_meta_data,
            f_power_est=self.estimate_power_one_channel,
            f_service_est=self.estimate_service_one_gaussian_channel)
        return d_map_estimate

    def estimate_power_one_channel(self, ind_channel, measurement_loc,
                                   measurements, building_meta_data):
        """
        Args:
            -`ind_channel`: an integer value corresponding to the channel(or source)
            - `measurements`: num_sources x num_measurements  matrix
                   with the measurements at each channel.
            - `measurement_locs`: 3 x num_measurements matrix with the
                3D locations of the measurements.

        Returns: Two num_points_y x num_points_x matrices of
                the estimated posterior mean and the posterior variance of ind_channel.

        """
        if building_meta_data:
            raise NotImplementedError

    #     if building_meta_data is None:
    #         building_meta_data = np.zeros(
    #             (self.grid.num_points_y, self.grid.num_points_x))

    #     # mask whose (j, k) entry is 1 if sampled taken, 0 if not taken,
    #     # and -1 if it is inside the building
    #     m_mask_with_meta_data = measurement_loc - building_meta_data

    #     m_estimated_pow_of_source, m_variance = self.estimate_power_one_channel_grid(
    #         ind_channel=ind_channel,
    #         t_measurements_grid=measurements,
    #         t_mask=m_mask_with_meta_data[None, :, :])

    #     return m_estimated_pow_of_source, m_variance

    # def estimate_power_one_channel_grid(self, ind_channel, t_measurements_grid,
    #                                     t_mask):
    #     """
    #         Args:
    #             - `t_measurements_grid`: num_sources x self.grid.num_points_y x self.grid.num_points_x
    #             - `t_mask`: self.grid.num_points_y x self.grid.num_points_x is a concatenation of
    #                     a sampling mask and building meta data whose value is:
    #                     1 at the grid point where the samples were taken,
    #                     -1 if the grid point is inside a building, 0 elsewhere.

    #         Returns:

    #             - `m_estimated_pow_of_source`: self.grid.num_points_y x self.grid.num_points_x matrix
    #              with the posterior mean of channel `ind_channel`.

    #             - `m_variance`: self.grid.num_points_y x self.grid.num_points_x matrix
    #              with the posterior variance of channel `ind_channel` if provided by
    #              self.estimator.__call__. Else, it is None.
    #     """
        rmap = Map(grid=self.grid,
                   m_meas_sf=measurements[ind_channel, :][:, None],
                   m_meas_locs_sf=measurement_loc.T)

        # get the mask and concatenate with the measurement
        m_mask = rmap.mask

        # stack along the first dimension for channels x Ny x Nx
        x_test = np.concatenate([rmap.t_meas_gf, m_mask[None, ...]], axis=0)

        if self.l_aux_map_estimators is not None:
            # append the estimates obtained with the auxiliary estimators
            # to x_test

            # l_aux_map_estimates = [
            #     aux_map_estimator.estimate_from_tensors(
            #         measurements=x_test[0, :, :][None, ...],
            #         measurement_locs=x_test[1, :, :])["t_power_map_estimate"]
            #     for aux_map_estimator in self.l_aux_map_estimators
            # ]
            # rmap = Map(
            #     grid=self.grid,
            #     #m_meas_locs_gf=x_test[1, :, :],
            #     # t_meas_gf=x_test[0, :, :][None, ...])
            #     m_meas_locs_sf=measurement_loc.T,
            #     m_meas_sf=measurements[ind_channel, :][:, None])

            l_aux_map_estimates = [
                aux_map_estimator.estimate(rmap)
                ["t_power_map_estimate"].t_meas_gf
                for aux_map_estimator in self.l_aux_map_estimators
            ]

            # concatenate the aux map estimates with x_test
            x_test = np.concatenate([x_test] + l_aux_map_estimates, axis=0)

        # make measured data of dim = 4 for compatibility with the training data set of NN
        x_test = x_test[tf.newaxis, ...].astype("float32")

        # shape (1, 2, Ny, Nx)
        prediction = self.estimator(x_test)
        mu_pred = prediction[:, 0, ...]

        m_estimated_pow_of_source = np.reshape(
            mu_pred[0], (self.grid.num_points_y, self.grid.num_points_x))

        if prediction.shape[1] > 1:
            sigma_pred = prediction[:, 1, ...]
            # (i,j)-th entry contains the posterior variance of the power at the (i,j)-th grid point
            m_variance = np.reshape(
                np.square(sigma_pred[0]),
                (self.grid.num_points_y, self.grid.num_points_x))
            # m_variance = np.where(m_mask == 1, 0.03, m_variance)

            if (m_variance < 0).any():
                raise ValueError("Negative variance")
            # m_uncertainty = np.zeros((self.grid.num_points_x, self.grid.num_points_y))
        else:
            m_variance = None

        return m_estimated_pow_of_source, m_variance


class NeuralNetworkEstimator(Model):

    @tf.function
    def call(self, x):
        """
        Args:

            `x`: 'NCHW' format batch tf dataset where

            x[:, 0, :, :] contains the measurements. Unobserved entries are
            np.nan.

            x[:, 1, :, :] contains the mask. 1 means observed location, 0 means
            unobserved location, and -1 means location inside a building.

            x[:, 2, :, :] contains the auxiliary estimates [optional]

        Returns:

            `output`: a tensor of format NCHW (C>=1) where 
        
                output[:, 0, :, :] is the posterior mean or power estimate.
              
                output[:,1,:,:] is the posterior std [optional]

        """

        raise NotImplementedError

    @tf.function
    def replace_nan_with_0(self, x):
        return tf.where(tf.math.is_nan(x), 0.0, x)

    @staticmethod
    def generate_dataset_for_nn(map_generator,
                                grid,
                                num_maps,
                                num_blocks_per_map,
                                num_meas_fraction_per_map=0.3,
                                l_aux_map_estimators=None):
        """
        This function generates a dataset for training the neural network. It
        generates num_maps maps and for each map, it generates
        num_blocks_per_map. Args:
            - map_generator: Map generator
            - grid: Grid
            - num_maps: number of maps to generate

            - num_blocks_per_map: number of blocks of num_meas_fraction_per_map
              observation that are drawn from each generated map.

            - num_meas_fraction_per_map: Fraction of measurements to get
              observations from a radio map. It can be a tuple of two numbers,
              where the first number is the minimum fraction and the second
              number is the maximum fraction. If it is a number, it is the
              fraction of measurements to get observations from a radio map.

            - l_aux_map_estimators [optional]: a list of object of class MapEstimator. 
        Returns:
            `x_data`: (num_maps * num_blocks_per_map, C , num_points_y,
            num_points_x) array, where 
            
                - if map_estimator==None, then C=2, where the first
                channel contains the measurements and the second channel
                contains the mask.

                - if map_estimator is not None, then C=3, where the first
                channel contains the measurements, the second channel
                contains the mask, and the third channel contains the
                map estimate obtained using map_estimator.

            `y_data`: (num_maps * num_blocks_per_map, 2, num_grid_point_y,
            num_grid_point_x) array, where the first channel is the true
            measurements and the second channel is the mask.
        """
        if l_aux_map_estimators is not None:
            assert isinstance(l_aux_map_estimators, list)

        l_true_maps = []
        l_input_maps = []

        def get_frac_training_meas(num_meas_fraction_per_map):

            if type(num_meas_fraction_per_map) == tuple:
                fraction_of_training_meas = round(
                    np.random.uniform(num_meas_fraction_per_map[0],
                                      num_meas_fraction_per_map[1]), 2)
            else:
                fraction_of_training_meas = num_meas_fraction_per_map
            return fraction_of_training_meas

        for ind_maps in range(num_maps):

            if num_maps > 50 and ind_maps and ind_maps % 50 == 0:
                print(f"{ind_maps} maps generated")

            attempts = 0
            while True:
                map_patch = map_generator.generate_map()
                attempts += 1

                if map_patch.m_meas_sf.shape[0] > 0:
                    print(f'Attempts to generate a map: {attempts}')
                    break

            map_patch.grid = grid

            for _ in range(num_blocks_per_map):

                map_obs = map_patch.get_obs(
                    sampling_mode="grid",
                    frac_obs=get_frac_training_meas(num_meas_fraction_per_map),
                )

                m_map_obs_with_mask = np.concatenate(
                    [map_obs.t_meas_gf, map_obs.mask[None, :, :]], axis=0)

                m_true_map_with_mask = np.concatenate(
                    [map_patch.t_meas_gf, map_patch.mask[None, :, :]], axis=0)

                # get the map estimate if l_aux_map_estimators is not None
                if l_aux_map_estimators is not None:

                    # use list comprehension to get the aux map estimates
                    l_aux_map_estimates = [
                        aux_map_estimator.estimate(map_obs)
                        ["t_power_map_estimate"].t_meas_gf
                        for aux_map_estimator in l_aux_map_estimators
                    ]

                    # concatenate the aux map estimates with m_map_obs_with_mask
                    m_map_obs_with_mask = np.concatenate(
                        [m_map_obs_with_mask] + l_aux_map_estimates, axis=0)

                    # m_map_estimate = aux_map_estimator.estimate_from_tensors(
                    #     measurements=map_obs.t_meas_gf,
                    #     measurement_locs=map_obs.mask)

                    # m_map_obs_with_mask = np.concatenate([
                    #     m_map_obs_with_mask,
                    #     m_map_estimate["t_power_map_estimate"]
                    # ],
                    #                                      axis=0)

                # append all maps
                l_true_maps.append(m_true_map_with_mask)
                l_input_maps.append(m_map_obs_with_mask)

        x_data = np.array(l_input_maps)
        y_data = np.array(l_true_maps)
        return x_data, y_data

    @staticmethod
    def get_augmented_data(input_data, output_data, l_augment_params=[None]):
        """
        Args:
            input_data: of shape num_samples_in x C x H x W
            output_data: of shape num_samples_in x C x H x W
            l_augment_params: a list of types of your choice for customizing the augmentation. It can be "horizontal_flip", "vertical_flip", "rotation", or "add_constant". If None, no augmentation is applied.

        Returns:
            augmented_input_data: of shape num_samples_out x C x H x W
            augmented_output_data: of shape num_samples_out x C x H x W
        """

        def custom_augmentation(input_item, output_item, augment_param=None):
            """
            Args:
                input_item: of shape num_samples x C x H x W
                output_item: of shape num_samples x C x H x W
                augment_params: a type of your choice for customizing the augmentation. It can be "horizontal_flip", "vertical_flip", "rotation", or "add_constant". If None, no augmentation is applied.
                
            Returns:
                input_item: of shape num_samples x C x H x W
                output_item: of shape num_samples x C x H x W    
                """

            if augment_param is not None:
                if augment_param == "horizontal_flip":
                    # Apply horizontal flipping
                    input_item = np.flip(input_item, axis=-1)
                    output_item = np.flip(output_item, axis=-1)

                # Apply vertical flipping
                elif augment_param == "vertical_flip":
                    input_item = np.flip(input_item, axis=-2)
                    output_item = np.flip(output_item, axis=-2)

                # Apply 90-degree rotation
                elif augment_param == "rotation":
                    input_item = np.rot90(input_item, k=1, axes=(-2, -1))
                    output_item = np.rot90(output_item, k=1, axes=(-2, -1))
                elif augment_param == "add_constant":
                    # add constant to 1st channel only
                    const = np.random.uniform(-10,
                                              10,
                                              size=(input_item.shape[0], 1, 1))
                    input_item[:, 0, ...] = input_item[:, 0, ...] + const
                    output_item[:, 0, ...] = output_item[:, 0, ...] + const

                else:
                    raise ValueError

            return input_item, output_item

        augmented_input_data = input_data
        augmented_output_data = output_data

        # append the augmented data
        for augment_param in l_augment_params:
            augmented_input, augmented_output = custom_augmentation(
                input_data, output_data, augment_param)

            augmented_input_data = np.concatenate(
                (augmented_input_data, augmented_input), axis=0)
            augmented_output_data = np.concatenate(
                (augmented_output_data, augmented_output), axis=0)

        return augmented_input_data, augmented_output_data

    def train(
        self,
        train_dataset=None,
        validation_dataset=None,
        epochs=10,
        loss="rmse",
        learning_rate=1e-4,
        save_weights_to=None,
        test_dataset=None,
        verbose=1,
    ):
        """

         If test_dataset is not None, the RMSE on data in test_dataset is
         printed after each epoch during training.

        Args:
            train_dataset: tf.data.Dataset object of the training data
            
            validation_dataset: tf.data.Dataset object of the validation data

            epochs: number of epochs

            loss: loss function. It can be "rmse".
            
            learning_rate: learning rate

            save_weights_to: path and base name of the file where the weights
            are saved. For example: "/home/user/weights/weights_". The eventual
            file name will be the result of appending some parameter values to
            this `save_weights_to`. 

            test_dataset: tf.data.Dataset object of the test data

        """

        assert save_weights_to is not None

        # rmse loss
        def rmse_loss(y_true, y_pred):
            """
            Args:
                -`y_true`: is N x 2 x H x W tensor, where y_true[:,0,...] is a
                true map and y_true[:,1,...] is a mask, where the (i,j)-th entry
                is 0 if the measuremenet is missing for the (i,j)-th pixel and 1
                otherwise.

                -`y_predict`: is N x 1 x H x W tensor where y_predict[:,0,...]
                is an estimated mean power.
            """
            # replace nan with 0
            m_mask = y_true[:, 1, ...]
            y_true = tf.where(m_mask == 0.0, 0.0, y_true[:, 0, ...])
            y_pred = y_pred[:, 0, ...]

            loss = tf.square(y_true - y_pred) * m_mask
            # reduce the loss over the masked entries
            loss = tf.reduce_sum(loss, axis=[1, 2]) / tf.reduce_sum(
                m_mask, axis=[1, 2])
            # take the mean over the batch
            loss = tf.reduce_mean(loss)

            return tf.sqrt(loss)

        weightfolder, base_name = self.get_weightfolder_basename(
            save_weights_to)
        weight_filepath = weightfolder + f"{base_name}_weight_file_epoch=" + "{epoch}"

        if loss == "rmse":
            loss = rmse_loss
        else:
            raise NotImplementedError

        # Compile the model
        self._compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=rmse_loss,
            metrics=[rmse_loss])

        l_callbacks = [
            tf.keras.callbacks.ModelCheckpoint(filepath=weight_filepath,
                                               save_weights_only=True,
                                               verbose=0,
                                               save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                             patience=10,
                                             mode='min',
                                             restore_best_weights=True)
        ]
        if test_dataset is not None:
            l_callbacks.append(TestCallBack(test_dataset))

        # Fit the model
        history = self._fit(x=train_dataset,
                            validation_data=validation_dataset,
                            epochs=epochs,
                            callbacks=l_callbacks,
                            verbose=verbose)

        # save the loss history
        lossfolder = weightfolder + f"{base_name}_loss/"
        os.makedirs(lossfolder, exist_ok=True)
        loss_file_path = lossfolder + f"{base_name}loss_history.pkl"
        with open(loss_file_path, 'wb') as outfile:
            pickle.dump(history.history, outfile)
        outfile.close()

        return history.history

    def plot_loss(self, d_history):
        from gsim.gfigure import GFigure

        # plot the loss and save the figure.
        l_epoch_number = np.arange(len(d_history['loss']))
        G = GFigure(xaxis=l_epoch_number,
                    yaxis=d_history[f"loss"],
                    xlabel="Epochs",
                    ylabel="Loss",
                    title="Loss vs. Epochs",
                    legend="Train RMSE loss",
                    styles="-D")
        G.add_curve(xaxis=l_epoch_number,
                    yaxis=d_history[f"val_loss"],
                    legend="Val. RMSE loss",
                    styles="-o")
        G.add_curve(xaxis=l_epoch_number,
                    yaxis=d_history[f"test_loss_rmse"],
                    legend="Test RMSE loss",
                    styles="-+")
        return G

    def get_weightfolder_basename(self, save_weights_to):
        """
            Args:
                save_weights_to: path and base name of the file where the weights are saved. For example: "/home/user/weights/weights_". The eventual file name will be the result of appending some parameter values to this `save_weights_to`.

            Returns:
                weightfolder: path of the folder where the weights are saved.
                base_name: base name of the file where the weights are saved.
            """
        # get weightfolder and base_name from save_weights_to

        weightfolder = os.path.dirname(save_weights_to) + "/"
        base_name = os.path.basename(save_weights_to)

        os.makedirs(weightfolder, exist_ok=True)

        return weightfolder, base_name


class TestCallBack(tf.keras.callbacks.Callback):

    def __init__(self, test_dataset):
        self.test_dataset = test_dataset

    def on_epoch_end(self, epoch, logs=None):
        losses = self.model.evaluate(self.test_dataset, verbose=0)
        logs['test_loss_rmse'] = losses[1]
