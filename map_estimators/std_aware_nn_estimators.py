import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from IPython.core.debugger import set_trace
from keras import Model, regularizers
from keras.layers import (BatchNormalization, Conv2D, Conv2DTranspose, Dense,
                          Dropout, Flatten, Input, MaxPool2D, MaxPooling2D,
                          Reshape, UpSampling2D, add, concatenate)
from ..map_estimators.neural_network_map_estimator import NeuralNetworkEstimator

from ..utilities import list_are_close

tfd = tfp.distributions
import os
import pickle

import keras.backend as K

# tf.config.run_functions_eagerly(True)


def loss_function_sigma(y_true, y_predict):
    """
            Args:
                -`y_true`: is N x 2 x H x W tensor, where y_true[:,0,...] is a
                true map and y_true[:,1,...] is a mask, where the (i,j)-th entry
                is 0 if the measuremenet is missing for the (i,j)-th pixel and 1
                otherwise.

                -`y_predict`: is N x 4 x H x W tensor where y_predict[:,0,...]
                is an estimated mean power, y_predict[:,1,...] is an estimated
                std. deviation, y_predict[:,2,...] is a sample_scaling tensor
                y_predict[:,3,...] is a mask

            """
    m_target = y_true[:, 0, ...]
    m_mask = y_true[:, 1, ...]
    m_target = tf.where(m_mask == 0.0, 0.0, m_target)

    m_mean = y_predict[:, 0, ...]
    m_std = y_predict[:, 1, ...]

    t_delta = tf.abs(m_target - m_mean)
    loss_sigma = tf.square(t_delta - m_std) * m_mask

    # reduce sum except for the batch dimension
    loss_sigma = tf.reduce_sum(loss_sigma, axis=-1)
    loss_sigma = tf.reduce_sum(loss_sigma, axis=-1)

    # get normalization factor in_sample and out_sample only.
    # It is equal to scaling factor * no. of meas loc +
    # (1-scaling factor) * no. of loc outside buildings
    num_obs = tf.reduce_sum(m_mask, axis=-1)
    num_obs = tf.reduce_sum(num_obs, axis=-1)
    loss_sigma = tf.divide(loss_sigma,
                           tf.math.maximum(num_obs, tf.constant([1.0])))
    # print(tf.math.maximum(sum_m_sample_scaling, tf.constant([1.0])))

    return loss_sigma


def loss_function_mean(y_true, y_predict):
    """
    Args:
        -`y_true`: is N x 2 x H x W tensor, where y_true[:,0,...] is a
        true map and y_true[:,1,...] is a mask, where the (i,j)-th entry
        is 0 if the measuremenet is missing for the (i,j)-th pixel and 1
        otherwise.

        -`y_predict`: is N x 4 x H x W tensor where y_predict[:,0,...]
        is an estimated mean power, y_predict[:,1,...] is an estimated
        std. deviation, y_predict[:,2,...] is a sample_scaling tensor
        y_predict[:,3,...] is a mask
    """

    m_target = y_true[:, 0, ...]
    m_mask = y_true[:, 1, ...]
    m_target = tf.where(m_mask == 0.0, 0.0, m_target)
    m_mean = y_predict[:, 0, ...]

    loss_mean = tf.square(m_target - m_mean) * m_mask

    # reduce sum except for the batch dimension
    loss_mean = tf.reduce_sum(loss_mean, axis=-1)
    loss_mean = tf.reduce_sum(loss_mean, axis=-1)

    # get normalization factor in_sample and out_sample only.
    # It is equal to scaling factor * no. of meas loc +
    # (1-scaling factor) * no. of loc outside buildings
    num_obs = tf.reduce_sum(m_mask, axis=-1)
    num_obs = tf.reduce_sum(num_obs, axis=-1)
    loss_mean = tf.divide(loss_mean,
                          tf.math.maximum(num_obs, tf.constant([1.0])))

    return loss_mean


def loss_function_mean_rmse(y_true, y_predict):
    """
    Args: see loss_function_mean
    """
    loss_mean = loss_function_mean(y_true, y_predict)
    return tf.math.sqrt(loss_mean)


def append_loss(dict_history, alpha, epochs, history):
    """
    Args:
        'dict_history': dictionary that contains the loss history (i.e., see
        StdAwareNnEstimator.fit())

        'alpha': alpha value for which the model is trained

        'epochs': number of epochs that the model is trained for

        'history': history object that contains the loss history for the model
    """
    vector_alpha = alpha * np.ones(shape=(epochs, ))
    vector_alpha.tolist()

    dict_history[f"train_mean_mse_loss"] += history.history[
        "loss_function_mean"]
    dict_history[f"train_sigma_mse_error"] += history.history[
        "loss_function_sigma"]
    dict_history[f"val_mean_mse_loss"] += history.history[
        "val_loss_function_mean"]
    dict_history[f"val_sigma_mse_error"] += history.history[
        "val_loss_function_sigma"]
    dict_history[f"train_loss"] += history.history["loss"]
    dict_history[f"val_loss"] += history.history["val_loss"]
    dict_history[f"alpha_vector"] += vector_alpha.tolist()

    return dict_history


class TestCallBack(tf.keras.callbacks.Callback):

    def __init__(self, test_dataset):
        self.test_dataset = test_dataset

    def on_epoch_end(self, epoch, logs=None):
        losses = self.model.evaluate(self.test_dataset, verbose=0)
        # logs['test_loss'] = losses[0]
        # logs['test_loss_function'] = losses[1]
        # logs['test_loss_function_mean'] = losses[2]
        # logs['test_loss_function_sigma'] = losses[3]
        logs['test_loss_function_mean_rmse'] = losses[4]
        # logs['test_accuracy'] = acc


class StdAwareNnEstimator(NeuralNetworkEstimator):

    def __init__(self, sample_scaling=0.5, meas_output_same=False):
        super().__init__()

        self.sample_scaling = sample_scaling
        self.meas_output_same = meas_output_same

    # a factory method that would instantiate the class given its name:
    @staticmethod
    def get_skip_conv_nn_arch(nn_arch_id):
        return globals()[f'StdAwareNnEstimatorArch' + nn_arch_id]()

    @tf.function
    def call(self, x):
        """
        See parent.
        """
        # Convert the input data into the`NHWC` format that is channels_last
        x = tf.transpose(x, perm=[0, 2, 3, 1])

        # replace the nan in the sampled map with 0
        x = super().replace_nan_with_0(x)

        sampled_map = x[..., 0][..., None]
        mask = x[..., 1][..., None]

        # comb_layer_output = self.combined_layers(x)
        mu = self.mean_layers(x)

        if self.meas_output_same:
            # Keep the estimated to be the same as measured value at
            # observed locations.
            mu = tf.where(mask == 1, sampled_map, mu)
            # mu[mask == 1] = sampled_map[mask == 1]

        input_to_sigma_layer = concatenate([x, mu], axis=-1)
        # print(input_to_sigma_layer.shape)
        # set_trace()
        sigma = self.std_deviation_layers(input_to_sigma_layer)
        # z = tf.concat((mu, sigma), axis=3)

        m_sample_scaling = tf.where(mask == 1, self.sample_scaling,
                                    1 - self.sample_scaling)
        m_sample_scaling = tf.where(mask == -1, 0.0, m_sample_scaling)

        # NHWC
        mean_n_sigma = tf.concat((mu, sigma), axis=-1)
        # mean_sigma_n_sample_scaling = tf.concat(
        #     (mean_n_sigma, m_sample_scaling), axis=-1)

        # mean_sigma_n_sample_scaling = tf.concat(
        #     (mean_sigma_n_sample_scaling, mask), axis=-1)

        #output = tf.transpose(mean_sigma_n_sample_scaling, perm=[0, 3, 1, 2])
        # print("output shape ", output.shape)

        # NCHW
        return tf.transpose(
            mean_n_sigma,
            perm=[0, 3, 1, 2])  #output  # tf.transpose(z, perm=[0, 3, 1, 2])

    def train(self,
              train_dataset=None,
              validation_dataset=None,
              l_alpha=[],
              l_epochs=[],
              loss_metric="Custom",
              l_learning_rate=[],
              save_weights_to=None,
              test_dataset=None):
        """
        This method train the model for different alpha values and returns a
        dict of the training loss. If alpha=0, the model learns only the posterior mean. If alpha=1, the model learns only the posterior standard deviation. Else, the model learns both posterior mean and standard deviation. For each i-th alpha value in l_alpha, the model is trained for l_epochs[i] epochs and with l_learning_rate[i] learning rate.
        
        If test_dataset is not None, the RMSE on data in test_dataset is printed
        after each epoch during training.

        Args:
            - `train_dataset`: a tf.data.Dataset object that contains training
              data. 

            - `validation_dataset`: a tf.data.Dataset object that contains
              validation data. It is of the shape

            - `l_alpha`: a list of alpha values for which the model is trained.
              If alpha=0, the model learns only the posterior mean. If alpha=1, the model learns only posterior standard deviation. Else, the model learns both posterior mean and standard deviation. For each i-th alpha value in l_alpha, the model is trained for l_epochs[i] epochs and with l_learning_rate[i] learning rate.

            - `l_epochs`: a list of epochs for which the model is trained.

            - `l_learning_rate`: a list of learning rates for which the model is
              trained.

            - `loss_metric`: a string that specifies the loss metric. If
              "Custom", the loss function is a weighted sum of mean and sigma
              losses. Else, it is not implemented.
            
            - `save_weights_to`: see parent.

            - `test_dataset`: a tf.data.Dataset object that contains test data.
              If not None, the RMSE on data in test_dataset is printed after
              each epoch during training.
            
        Returns:
            'dic_history': a dictionary that contians
                - "train_mean_mse_loss": a list of length sum(l_epochs) that
                contains a concatenated train loss for a subnetwork that learns
                posterior mean for all entries of l_alpha.

                - "train_sigma_mse_error": a list of length sum(l_epochs) that contains a concatenated train loss for a subnetwork
                that learns posterior standard deviation for all entries of
                l_alpha.

                - "val_mean_mse_loss": a list of length sum(l_epochs) that
                contains a concatenated validation loss for a subnetwork that
                learns posterior mean for all entries of l_alpha.

                - "val_sigma_mse_error": a list of length sum(l_epochs) that
                contains a concatenated validation loss for a subnetwork that
                learns posterior standard deviation for all entries of l_alpha.

                - "alpha_vector": a list of length sum(l_epochs) that
                contains a concatenated alpha values for all entries of l_alpha.

                - "train_loss": a list of length sum(l_epochs) that contains
                a concatenated total train loss (posterior mean train loss +
                posterior sigma train loss) for all entries of l_alpha.

                - "val_loss": a list of length sum(l_epochs) that contains a
                concatenated total validation loss (posterior mean val loss +
                posterior sigma val loss) for all entries of l_alpha.

        """

        dict_history = {
            "train_mean_mse_loss": [],
            "train_sigma_mse_error": [],
            "val_mean_mse_loss": [],
            "val_sigma_mse_error": [],
            "alpha_vector": [],
            "train_loss": [],
            "val_loss": [],
        }
        assert l_learning_rate != []
        assert l_epochs != []
        assert l_alpha != []
        assert save_weights_to is not None

        # check len of l_alpha, l_epochs and l_learning_rate are the same
        assert len(l_alpha) == len(l_epochs) == len(l_learning_rate)
        weightfolder, base_name = self.get_weightfolder_basename(
            save_weights_to)

        for alpha, epochs, learning_rate in zip(l_alpha, l_epochs,
                                                l_learning_rate):

            # Loss function
            def loss_function(y_true, y_predict):
                """
                Args: see loss_function_mean
                """
                loss_mean = loss_function_mean(y_true, y_predict)
                loss_sigma = loss_function_sigma(y_true, y_predict)
                loss = alpha * loss_sigma + (1 - alpha) * loss_mean

                return loss

            # loss metric
            if loss_metric == 'Custom':
                assert alpha is not None
                if alpha == 0:
                    # freeze combined lsayers and std. deviation layers
                    self.std_deviation_layers.trainable = False
                    self.mean_layers.trainable = True
                elif alpha == 1:
                    # freeze combined layers and mean layers
                    self.std_deviation_layers.trainable = True
                    self.mean_layers.trainable = False
                else:
                    self.std_deviation_layers.trainable = True
                    self.mean_layers.trainable = True

            else:
                raise ValueError

            weight_filepath = weightfolder + f"{base_name}_weights_alpha={alpha}" f"_samp_scale={self.sample_scaling}" f"_out_same={self.meas_output_same}" + "_epoch={epoch}"

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

            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

            self.compile(
                optimizer=optimizer,
                loss=loss_function,
                metrics=[
                    loss_function,
                    loss_function_mean,
                    loss_function_sigma,
                    loss_function_mean_rmse,
                ],
            )

            # train the model
            history = super().fit(x=train_dataset,
                                  epochs=epochs,
                                  validation_data=validation_dataset,
                                  callbacks=l_callbacks,
                                  verbose=1)

            # append the loss for each alpha value
            append_loss(dict_history=dict_history,
                        alpha=alpha,
                        epochs=epochs,
                        history=history)

            # save the loss history
            lossfolder = weightfolder + f"{base_name}_loss/"
            os.makedirs(lossfolder, exist_ok=True)
            loss_file_path = lossfolder + f"{base_name}_loss_history_alpha={alpha}_epochs={epochs}_samp_scale={self.sample_scaling}_out_same={self.meas_output_same}.pkl"
            with open(loss_file_path, 'wb') as outfile:
                pickle.dump(dict_history, outfile)
            outfile.close()

        return dict_history

    def plot_loss(self, d_history):
        """
        This function plots the loss function and returns the figure.
        """

        from gsim.gfigure import GFigure

        # plot the loss and save the figure.
        l_epoch_number = np.arange(len(d_history['train_mean_mse_loss']))
        G = GFigure(xaxis=l_epoch_number,
                    yaxis=d_history[f"train_mean_mse_loss"],
                    xlabel="Epochs",
                    ylabel="Loss",
                    title="Loss vs. Epochs",
                    legend="Train loss mean mse",
                    styles="-D")
        G.add_curve(xaxis=l_epoch_number,
                    yaxis=d_history[f"train_sigma_mse_error"],
                    legend="Train loss sigma error",
                    styles="-o")
        G.add_curve(xaxis=l_epoch_number,
                    yaxis=d_history[f"val_mean_mse_loss"],
                    legend="Val loss mean ",
                    styles="-+")
        G.add_curve(xaxis=l_epoch_number,
                    yaxis=d_history[f"val_sigma_mse_error"],
                    legend="Val loss sigma error",
                    styles="-x")
        G.add_curve(xaxis=l_epoch_number,
                    yaxis=d_history[f"train_loss"],
                    legend="Total train loss",
                    styles=":*")
        G.add_curve(xaxis=l_epoch_number,
                    yaxis=d_history[f"val_loss"],
                    legend="Total val loss",
                    styles=":+")

        return G


class SurveyingStdAwareNnEstimator(StdAwareNnEstimator):
    """
        Network in [shrestha2022surveying] 
    
       Skip convolutional network with two identical sub-networks for learning
        posterior mean and posterior standard deviation. Each sub-network has
        total 26 layers, 10 convolution, 10 transpose convolution, 3 max
        pooling, and 3 upsampling layers.
    """

    def __init__(self, load_weights_from=None, **kwargs):

        super(SurveyingStdAwareNnEstimator, self).__init__(**kwargs)
        self.regularizers = regularizers.l2(0.001)
        # self.combined_layers = self.CombinedLayers()
        self.mean_layers = self.MeanLayers()
        self.std_deviation_layers = self.StdDeviationlayers()
        if load_weights_from is not None:
            self.load_weights(load_weights_from).expect_partial()

    class MeanLayers(Model):

        def __init__(self):
            super().__init__()
            self.regularizers = None
            # self.regularizers = regularizers.l2(0.001)

            self.conv1 = Conv2D(filters=128,
                                kernel_size=4,
                                activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv2 = Conv2D(128,
                                3,
                                activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv3 = Conv2D(128,
                                3,
                                activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers)

            self.max_pool_1 = MaxPool2D(pool_size=(2, 2))

            self.conv4 = Conv2D(256,
                                4,
                                activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv5 = Conv2D(256,
                                4,
                                activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv6 = Conv2D(256,
                                4,
                                activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers)

            self.max_pool_2 = MaxPool2D(pool_size=(2, 2))

            self.conv1T = Conv2DTranspose(128,
                                          4,
                                          activation=tf.nn.leaky_relu,
                                          padding='same',
                                          kernel_regularizer=self.regularizers)

            self.conv2T = Conv2DTranspose(128,
                                          4,
                                          activation=tf.nn.leaky_relu,
                                          padding='same',
                                          kernel_regularizer=self.regularizers)

            self.conv3T = Conv2DTranspose(128,
                                          4,
                                          activation=tf.nn.leaky_relu,
                                          padding='same',
                                          kernel_regularizer=self.regularizers)

            self.up_sampling_1 = UpSampling2D(size=(2, 2))

            self.conv4T = Conv2DTranspose(256,
                                          4,
                                          activation=tf.nn.leaky_relu,
                                          padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv5T = Conv2DTranspose(256,
                                          4,
                                          activation=tf.nn.leaky_relu,
                                          padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv6T = Conv2DTranspose(256,
                                          4,
                                          activation=tf.nn.leaky_relu,
                                          padding='same',
                                          kernel_regularizer=self.regularizers)

            self.up_sampling_2 = UpSampling2D(size=(2, 2))

            self.conv7 = Conv2D(512,
                                4,
                                activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv7T = Conv2DTranspose(512,
                                          4,
                                          activation=tf.nn.leaky_relu,
                                          padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv8 = Conv2D(512,
                                4,
                                activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv8T = Conv2DTranspose(512,
                                          4,
                                          activation=tf.nn.leaky_relu,
                                          padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv9 = Conv2D(512,
                                4,
                                activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv9T = Conv2DTranspose(512,
                                          4,
                                          activation=tf.nn.leaky_relu,
                                          padding='same',
                                          kernel_regularizer=self.regularizers)

            self.max_pool_3 = MaxPool2D(pool_size=(2, 2))
            self.up_sampling_3 = UpSampling2D(size=(2, 2))

            self.conv10 = Conv2D(1024,
                                 4,
                                 activation=tf.nn.leaky_relu,
                                 padding='same',
                                 kernel_regularizer=self.regularizers)
            self.conv10T = Conv2DTranspose(
                1024,
                4,
                activation=tf.nn.leaky_relu,
                padding='same',
                kernel_regularizer=self.regularizers)

            self.conv_out_mu = Conv2DTranspose(
                1,
                4,
                activation=tf.nn.leaky_relu,
                padding='same',
                kernel_regularizer=self.regularizers)

        @tf.function
        def call(self, x):
            ## separate branch from this point
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.max_pool_1(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.conv6(x)
            x = self.max_pool_2(x)
            x = self.conv7(x)
            x = self.conv8(x)
            x = self.conv9(x)
            x = self.max_pool_3(x)
            x = self.conv10(x)
            x = self.conv10T(x)
            x = self.up_sampling_3(x)
            x = self.conv9T(x)
            x = self.conv8T(x)
            x = self.conv7T(x)
            x = self.up_sampling_2(x)
            x = self.conv6T(x)
            x = self.conv5T(x)
            x = self.conv4T(x)
            x = self.up_sampling_1(x)
            x = self.conv3T(x)
            x = self.conv2T(x)
            x = self.conv1T(x)

            mu = self.conv_out_mu(x)

            return mu

    class StdDeviationlayers(Model):

        def __init__(self):
            super().__init__()
            self.regularizers = None
            # self.regularizers = regularizers.l2(0.001)
            self.conv1 = Conv2D(filters=128,
                                kernel_size=4,
                                activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv2 = Conv2D(128,
                                3,
                                activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv3 = Conv2D(128,
                                3,
                                activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers)

            self.max_pool_1 = MaxPool2D(pool_size=(2, 2))

            self.conv4 = Conv2D(256,
                                4,
                                activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv5 = Conv2D(256,
                                4,
                                activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv6 = Conv2D(256,
                                4,
                                activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers)

            self.max_pool_2 = MaxPool2D(pool_size=(2, 2))

            self.conv1_1T = Conv2DTranspose(
                128,
                4,
                activation=tf.nn.leaky_relu,
                padding='same',
                kernel_regularizer=self.regularizers)

            self.conv2_1T = Conv2DTranspose(
                128,
                4,
                activation=tf.nn.leaky_relu,
                padding='same',
                kernel_regularizer=self.regularizers)

            self.conv3_1T = Conv2DTranspose(
                128,
                4,
                activation=tf.nn.leaky_relu,
                padding='same',
                kernel_regularizer=self.regularizers)

            self.up_sampling_1_1 = UpSampling2D(size=(2, 2))

            self.conv4_1T = Conv2DTranspose(
                256,
                4,
                activation=tf.nn.leaky_relu,
                padding='same',
                kernel_regularizer=self.regularizers)
            self.conv5_1T = Conv2DTranspose(
                256,
                4,
                activation=tf.nn.leaky_relu,
                padding='same',
                kernel_regularizer=self.regularizers)
            self.conv6_1T = Conv2DTranspose(
                256,
                4,
                activation=tf.nn.leaky_relu,
                padding='same',
                kernel_regularizer=self.regularizers)
            self.up_sampling_2_1 = UpSampling2D(size=(2, 2))

            self.conv7_1 = Conv2D(512,
                                  4,
                                  activation=tf.nn.leaky_relu,
                                  padding='same',
                                  kernel_regularizer=self.regularizers)
            self.conv7_1T = Conv2DTranspose(
                512,
                4,
                activation=tf.nn.leaky_relu,
                padding='same',
                kernel_regularizer=self.regularizers)
            self.conv8_1 = Conv2D(512,
                                  4,
                                  activation=tf.nn.leaky_relu,
                                  padding='same',
                                  kernel_regularizer=self.regularizers)
            self.conv8_1T = Conv2DTranspose(
                512,
                4,
                activation=tf.nn.leaky_relu,
                padding='same',
                kernel_regularizer=self.regularizers)
            self.conv9_1 = Conv2D(512,
                                  4,
                                  activation=tf.nn.leaky_relu,
                                  padding='same',
                                  kernel_regularizer=self.regularizers)
            self.conv9_1T = Conv2DTranspose(
                512,
                4,
                activation=tf.nn.leaky_relu,
                padding='same',
                kernel_regularizer=self.regularizers)

            self.max_pool_3_1 = MaxPool2D(pool_size=(2, 2))
            self.up_sampling_3_1 = UpSampling2D(size=(2, 2))

            self.conv10_1 = Conv2D(1024,
                                   4,
                                   activation=tf.nn.leaky_relu,
                                   padding='same',
                                   kernel_regularizer=self.regularizers)
            self.conv10_1T = Conv2DTranspose(
                1024,
                4,
                activation=tf.nn.leaky_relu,
                padding='same',
                kernel_regularizer=self.regularizers)

            self.conv_out_sigma = Conv2DTranspose(
                1,
                4,
                activation=tf.keras.activations.
                exponential,  #lambda x: tf.nn.elu(x) + 1,
                padding='same',
                kernel_regularizer=self.regularizers)

        @tf.function
        def call(self, y):
            ## separate branch from this point
            y = self.conv1(y)
            y = self.conv2(y)
            y = self.conv3(y)
            y = self.max_pool_1(y)
            y = self.conv4(y)
            y = self.conv5(y)
            y = self.conv6(y)
            y = self.max_pool_2(y)
            y = self.conv7_1(y)
            y = self.conv8_1(y)
            y = self.conv9_1(y)
            y = self.max_pool_3_1(y)
            y = self.conv10_1(y)
            y = self.conv10_1T(y)
            y = self.up_sampling_3_1(y)
            y = self.conv9_1T(y)
            y = self.conv8_1T(y)
            y = self.conv7_1T(y)
            y = self.up_sampling_2_1(y)
            y = self.conv6_1T(y)
            y = self.conv5_1T(y)
            y = self.conv4_1T(y)
            y = self.up_sampling_1_1(y)
            y = self.conv3_1T(y)
            y = self.conv2_1T(y)
            y = self.conv1_1T(y)

            sigma = self.conv_out_sigma(y)

            return sigma


class SurveyingStdAwareNnEstimatorV2(StdAwareNnEstimator):
    """ 
    Similar to SurveyingNnEstimator but with fewer layers and filters.

    Convolutional Neural Network with total 16 layers in each sub-network.
    """

    def __init__(self, load_weights_from=None, **kwargs):

        super(SurveyingStdAwareNnEstimatorV2, self).__init__()
        self.regularizers = regularizers.l2(0.001)
        self.mean_layers = self.MeanLayers()
        self.std_deviation_layers = self.StdDeviationlayers()

        if load_weights_from is not None:
            self.compile(optimizer=tf.keras.optimizers.Adam())
            self.load_weights(load_weights_from)

        pass

    class MeanLayers(Model):

        def __init__(self):
            super().__init__()
            self.regularizers = None
            # self.regularizers = regularizers.l2(0.001)

            self.conv1 = Conv2D(filters=64,
                                kernel_size=4,
                                activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv2 = Conv2D(64,
                                3,
                                activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers)

            self.max_pool_1 = MaxPool2D(pool_size=(2, 2))

            self.conv3 = Conv2D(128,
                                3,
                                activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv4 = Conv2D(128,
                                4,
                                activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers)

            self.max_pool_2 = MaxPool2D(pool_size=(2, 2))

            self.conv5 = Conv2D(256,
                                4,
                                activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv6 = Conv2D(256,
                                4,
                                activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv1_1T = Conv2DTranspose(
                64,
                4,
                activation=tf.nn.leaky_relu,
                padding='same',
                kernel_regularizer=self.regularizers)

            self.conv2_1T = Conv2DTranspose(
                64,
                4,
                activation=tf.nn.leaky_relu,
                padding='same',
                kernel_regularizer=self.regularizers)
            self.up_sampling_1_1 = UpSampling2D(size=(2, 2))
            self.conv3_1T = Conv2DTranspose(
                128,
                4,
                activation=tf.nn.leaky_relu,
                padding='same',
                kernel_regularizer=self.regularizers)

            self.conv4_1T = Conv2DTranspose(
                128,
                4,
                activation=tf.nn.leaky_relu,
                padding='same',
                kernel_regularizer=self.regularizers)
            self.conv5_1T = Conv2DTranspose(
                256,
                4,
                activation=tf.nn.leaky_relu,
                padding='same',
                kernel_regularizer=self.regularizers)
            self.up_sampling_2_1 = UpSampling2D(size=(2, 2))

            self.conv_out_mu = Conv2DTranspose(
                1,
                4,
                activation=tf.nn.leaky_relu,
                padding='same',
                kernel_regularizer=self.regularizers)

        @tf.function
        def call(self, y):
            ## separate branch from this point
            y = self.conv1(y)
            y = self.conv2(y)
            conv1 = y  # first copy
            y = self.max_pool_1(y)

            y = self.conv3(y)
            y = self.conv4(y)
            conv2 = y  # second copy
            y = self.max_pool_2(y)

            y = self.conv5(y)
            y = self.conv6(y)

            y = self.conv5_1T(y)
            y = self.up_sampling_1_1(y)

            y = concatenate([y, conv2], axis=-1)  # merge2

            y = self.conv4_1T(y)
            y = self.conv3_1T(y)
            y = self.up_sampling_2_1(y)

            y = concatenate([y, conv1], axis=-1)  # merge1
            y = self.conv2_1T(y)
            y = self.conv1_1T(y)

            mu = self.conv_out_mu(y)

            return mu

    class StdDeviationlayers(Model):

        def __init__(self):
            super().__init__()
            self.regularizers = None
            # self.regularizers = regularizers.l2(0.001)
            self.conv1 = Conv2D(filters=64,
                                kernel_size=4,
                                activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv2 = Conv2D(64,
                                3,
                                activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers)

            self.max_pool_1 = MaxPool2D(pool_size=(2, 2))

            self.conv3 = Conv2D(128,
                                3,
                                activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv4 = Conv2D(128,
                                4,
                                activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers)

            self.max_pool_2 = MaxPool2D(pool_size=(2, 2))

            self.conv5 = Conv2D(256,
                                4,
                                activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv6 = Conv2D(256,
                                4,
                                activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv1_1T = Conv2DTranspose(
                64,
                4,
                activation=tf.nn.leaky_relu,
                padding='same',
                kernel_regularizer=self.regularizers)

            self.conv2_1T = Conv2DTranspose(
                64,
                4,
                activation=tf.nn.leaky_relu,
                padding='same',
                kernel_regularizer=self.regularizers)
            self.up_sampling_1_1 = UpSampling2D(size=(2, 2))
            self.conv3_1T = Conv2DTranspose(
                128,
                4,
                activation=tf.nn.leaky_relu,
                padding='same',
                kernel_regularizer=self.regularizers)

            self.conv4_1T = Conv2DTranspose(
                128,
                4,
                activation=tf.nn.leaky_relu,
                padding='same',
                kernel_regularizer=self.regularizers)
            self.conv5_1T = Conv2DTranspose(
                256,
                4,
                activation=tf.nn.leaky_relu,
                padding='same',
                kernel_regularizer=self.regularizers)
            self.up_sampling_2_1 = UpSampling2D(size=(2, 2))

            self.conv_out_sigma = Conv2DTranspose(
                1,
                4,
                activation=tf.keras.activations.exponential,
                # lambda x: tf.nn.elu(x) + 1,
                padding='same',
                kernel_regularizer=self.regularizers)

        @tf.function
        def call(self, y):
            ## separate branch from this point
            y = self.conv1(y)
            y = self.conv2(y)
            conv1 = y  # first copy
            y = self.max_pool_1(y)

            y = self.conv3(y)
            y = self.conv4(y)
            conv2 = y  # second copy
            y = self.max_pool_2(y)

            y = self.conv5(y)
            y = self.conv6(y)

            y = self.conv5_1T(y)
            y = self.up_sampling_1_1(y)

            y = concatenate([y, conv2], axis=-1)  # merge2

            y = self.conv4_1T(y)
            y = self.conv3_1T(y)
            y = self.up_sampling_2_1(y)

            y = concatenate([y, conv1], axis=-1)  # merge1
            y = self.conv2_1T(y)
            y = self.conv1_1T(y)

            sigma = self.conv_out_sigma(y)

            return sigma


class UnetStdAwareNnEstimator(StdAwareNnEstimator):
    """
    Network in [krijestorac2020deeplearning] with tanh activations replaced with
    leaky relu activations.
    """

    def __init__(self,
                 load_weights_from=None,
                 learn_diff_from_second_aux_est=False,
                 **kwargs):

        super(UnetStdAwareNnEstimator, self).__init__(**kwargs)
        self.regularizers = regularizers.l2(0.001)
        # self.combined_layers = self.CombinedLayers()
        self.mean_layers = self.MeanLayers(
            learn_diff_from_second_aux_est=learn_diff_from_second_aux_est)
        self.std_deviation_layers = self.StdDeviationlayers()
        if load_weights_from is not None:
            self.load_weights(load_weights_from).expect_partial()

    class MeanLayers(Model):

        def __init__(self, learn_diff_from_second_aux_est=False):
            self.learn_diff_from_second_aux_est = learn_diff_from_second_aux_est
            super().__init__()
            self.conv_mean_1 = Conv2D(64,
                                      3,
                                      activation='leaky_relu',
                                      padding='same')
            self.conv_mean_2 = Conv2D(64,
                                      3,
                                      activation='leaky_relu',
                                      padding='same')
            self.maxpool_mean_1 = MaxPooling2D((2, 2), padding="same")
            self.conv_mean_3 = Conv2D(128,
                                      3,
                                      activation='leaky_relu',
                                      padding='same')
            self.conv_mean_4 = Conv2D(128,
                                      3,
                                      activation='leaky_relu',
                                      padding='same')
            self.maxpool_mean_2 = MaxPooling2D((2, 2), padding="same")
            self.conv_mean_5 = Conv2D(256,
                                      3,
                                      activation='leaky_relu',
                                      padding='same')
            self.conv_mean_6 = Conv2D(256,
                                      3,
                                      activation='leaky_relu',
                                      padding='same')
            self.maxpool_mean_3 = MaxPooling2D((2, 2), padding="same")
            self.conv_mean_15 = Conv2D(256,
                                       3,
                                       activation='leaky_relu',
                                       padding='same')
            self.conv_mean_16 = Conv2D(256,
                                       3,
                                       activation='leaky_relu',
                                       padding='same')
            self.upconv_mean_1 = Conv2DTranspose(256,
                                                 2,
                                                 strides=(2, 2),
                                                 padding='same')
            self.conv_mean_7 = Conv2D(256,
                                      3,
                                      activation='leaky_relu',
                                      padding='same')
            self.conv_mean_8 = Conv2D(128,
                                      3,
                                      activation='leaky_relu',
                                      padding='same')
            self.upconv_mean_2 = Conv2DTranspose(128,
                                                 2,
                                                 strides=(2, 2),
                                                 padding='same')
            self.conv_mean_9 = Conv2D(128,
                                      3,
                                      activation='leaky_relu',
                                      padding='same')
            self.conv_mean_10 = Conv2D(64,
                                       3,
                                       activation='leaky_relu',
                                       padding='same')
            self.upconv_mean_3 = Conv2DTranspose(64,
                                                 2,
                                                 strides=(2, 2),
                                                 padding='same')
            self.conv_mean_11 = Conv2D(64,
                                       3,
                                       activation='leaky_relu',
                                       padding='same')
            self.conv_mean_12 = Conv2D(32,
                                       3,
                                       activation='leaky_relu',
                                       padding='same')
            self.conv_mean_13 = Conv2D(8,
                                       3,
                                       activation='leaky_relu',
                                       padding='same')
            self.conv_mean_14 = Conv2D(
                1,
                3,
                # activation='leaky_relu',
                padding='same')

        @tf.function
        def call(self, x):
            mean = x
            mean = self.conv_mean_1(mean)
            mean = self.conv_mean_2(mean)
            mean1 = mean
            mean = self.maxpool_mean_1(mean)
            mean = self.conv_mean_3(mean)
            mean = self.conv_mean_4(mean)
            mean2 = mean
            mean = self.maxpool_mean_2(mean)
            mean = self.conv_mean_5(mean)
            mean = self.conv_mean_6(mean)
            mean3 = mean
            mean = self.maxpool_mean_3(mean)
            mean = self.conv_mean_15(mean)
            mean = self.conv_mean_16(mean)
            mean = self.upconv_mean_1(mean)
            mean = tf.concat([mean3, mean], axis=-1)
            mean = self.conv_mean_7(mean)
            mean = self.conv_mean_8(mean)
            mean = self.upconv_mean_2(mean)
            mean = tf.concat([mean2, mean], axis=-1)
            mean = self.conv_mean_9(mean)
            mean = self.conv_mean_10(mean)
            mean = self.upconv_mean_3(mean)
            mean = tf.concat([mean1, mean], axis=-1)
            mean = self.conv_mean_11(mean)
            mean = self.conv_mean_12(mean)
            mean = self.conv_mean_13(mean)
            mean = self.conv_mean_14(mean)
            if x.shape[-1] >= 4 and self.learn_diff_from_second_aux_est:
                mean += x[..., 3:4]
            return mean

    class StdDeviationlayers(Model):

        def __init__(self):
            super().__init__()
            self.conv_var_1 = Conv2D(64,
                                     3,
                                     activation='leaky_relu',
                                     padding='same')
            self.conv_var_2 = Conv2D(64,
                                     3,
                                     activation='leaky_relu',
                                     padding='same')
            self.maxpool_var_1 = MaxPooling2D((2, 2), padding="same")
            self.conv_var_3 = Conv2D(128,
                                     3,
                                     activation='leaky_relu',
                                     padding='same')
            self.conv_var_4 = Conv2D(128,
                                     3,
                                     activation='leaky_relu',
                                     padding='same')
            self.maxpool_var_2 = MaxPooling2D((2, 2), padding="same")
            self.conv_var_5 = Conv2D(256,
                                     3,
                                     activation='leaky_relu',
                                     padding='same')
            self.conv_var_6 = Conv2D(256,
                                     3,
                                     activation='leaky_relu',
                                     padding='same')
            self.maxpool_var_3 = MaxPooling2D((2, 2), padding="same")
            self.conv_var_15 = Conv2D(256,
                                      3,
                                      activation='leaky_relu',
                                      padding='same')
            self.conv_var_16 = Conv2D(256,
                                      3,
                                      activation='leaky_relu',
                                      padding='same')
            self.upconv_var_1 = Conv2DTranspose(256,
                                                2,
                                                strides=(2, 2),
                                                padding='same')
            self.conv_var_7 = Conv2D(256,
                                     3,
                                     activation='leaky_relu',
                                     padding='same')
            self.conv_var_8 = Conv2D(128,
                                     3,
                                     activation='leaky_relu',
                                     padding='same')
            self.upconv_var_2 = Conv2DTranspose(128,
                                                2,
                                                strides=(2, 2),
                                                padding='same')
            self.conv_var_9 = Conv2D(128,
                                     3,
                                     activation='leaky_relu',
                                     padding='same')
            self.conv_var_10 = Conv2D(64,
                                      3,
                                      activation='leaky_relu',
                                      padding='same')
            self.upconv_var_3 = Conv2DTranspose(64,
                                                2,
                                                strides=(2, 2),
                                                padding='same')
            self.conv_var_11 = Conv2D(64,
                                      3,
                                      activation='leaky_relu',
                                      padding='same')
            self.conv_var_12 = Conv2D(32,
                                      3,
                                      activation='leaky_relu',
                                      padding='same')
            self.conv_var_13 = Conv2D(8,
                                      3,
                                      activation='leaky_relu',
                                      padding='same')
            self.conv_var_14 = Conv2D(1,
                                      3,
                                      activation='exponential',
                                      padding='same')

        @tf.function
        def call(self, x):
            var = x
            var = self.conv_var_1(var)
            var = self.conv_var_2(var)
            var1 = var
            var = self.maxpool_var_1(var)
            var = self.conv_var_3(var)
            var = self.conv_var_4(var)
            var2 = var
            var = self.maxpool_var_2(var)
            var = self.conv_var_5(var)
            var = self.conv_var_6(var)
            var3 = var
            var = self.maxpool_var_3(var)
            var = self.conv_var_15(var)
            var = self.conv_var_16(var)
            var = self.upconv_var_1(var)
            var = tf.concat([var3, var], axis=-1)
            var = self.conv_var_7(var)
            var = self.conv_var_8(var)
            var = self.upconv_var_2(var)
            var = tf.concat([var2, var], axis=-1)
            var = self.conv_var_9(var)
            var = self.conv_var_10(var)
            var = self.upconv_var_3(var)
            var = tf.concat([var1, var], axis=-1)
            var = self.conv_var_11(var)
            var = self.conv_var_12(var)
            var = self.conv_var_13(var)
            var = self.conv_var_14(var)
            return var

    def import_weights_from_non_hybrid_estimator(self, estimator,
                                                 input_shape_1, input_shape_2):
        """
        Args:
            estimator: an object of class NeuralNetworkEstimator
            in
        """

        def expand_weights(weights):
            """
            Args:
                weights: list of length 2, where 0-th element is the weights of shape (m,n, 2, num_filters) and 1-th element is the bias of shape (num_filters,)

            Returns:
                new_weights: list of length 2, where 0-th element is the result of appending zeros to weights[0] so that the resulting shape is (m,n,5, num_filters). And new_weights[1] = weights[1]
            """
            original_shape = weights[0].shape

            new_weights = [
                tf.concat([
                    weights[0],
                    tf.zeros(shape=(original_shape[0], original_shape[1], 3,
                                    original_shape[3]))
                ],
                          axis=2), weights[1]
            ]

            return new_weights

        def update_subnet_weights(current_subnet, target_subnet):
            """
            Args:
                current_subnet: a subnet of the hybrid estimator with multiple layers where the weights are obtained.

                target_subnet: a subnet of the hybrid estimator with multiple layers where the weights are expanded and set.
            """

            for ind_layer in range(len(current_subnet.layers)):
                if ind_layer == 0:
                    new_weights = expand_weights(
                        target_subnet.layers[ind_layer].get_weights())

                else:
                    new_weights = target_subnet.layers[ind_layer].get_weights()

                current_subnet.layers[ind_layer].set_weights(new_weights)

        # forward pass to initialize the weights
        self(tf.zeros(shape=input_shape_1))
        estimator(tf.zeros(shape=input_shape_2))

        # update the weights of the mean and std deviation subnets
        update_subnet_weights(self.mean_layers, estimator.mean_layers)
        update_subnet_weights(self.std_deviation_layers,
                              estimator.std_deviation_layers)
