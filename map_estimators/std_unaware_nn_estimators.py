import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
import tensorflow.keras.utils as ku
import numpy as np
import sys
from tensorflow.keras.layers import (BatchNormalization, Conv2D,
                                     Conv2DTranspose, Dense, Dropout, Flatten,
                                     Input, MaxPool2D, MaxPooling2D, Reshape,
                                     UpSampling2D, add, concatenate)
from ..map_estimators.neural_network_map_estimator import NeuralNetworkEstimator

# tf.config.run_functions_eagerly(True)


class CompletionAutoencoderEstimator(NeuralNetworkEstimator):
    """
     Network in [teganya2020rme]. 
     
     The network is a 26 layer network: 13 for encoder and 13 for decoder (fully convolutional)
    """

    class NchwToNhwcInput(tf.keras.layers.Layer):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.permute = tf.keras.layers.Permute((2, 3, 1))

        def call(self, inputs):
            # Permute the dimensions of the input tensor from NCHW to NHWC format
            inputs = self.permute(inputs)

            # replace nan with 0
            inputs = tf.where(tf.math.is_nan(inputs), 0.0, inputs)

            return inputs

    def __init__(self,
                 height=32,
                 width=32,
                 c_len=16,
                 n_filters=32,
                 kernel_size=(3, 3),
                 conv_stride=1,
                 pool_size_str=2,
                 use_batch_norm=False,
                 n_channels=2,
                 load_weights_from=None,
                 print_model_summary=False):
        """
        Args:
            - height: height of the input image
            - width: width of the input image
            - c_len: length of the code or latent space
            - n_filters: number of filters in the convolutional layers
            - kernel_size: size of the kernel in the convolutional layers
            - conv_stride: stride of the convolutional layers
            - pool_size_str: size of the pooling layers
            - use_batch_norm: flag for using batch normalization
            - n_channels: number of channels in the input image
            - load_weights_from: path to the weights file to load
            - print_model_summary: flag for printing the model summary
        """

        super().__init__()
        self.height = height
        self.width = width
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.pool_size_str = pool_size_str
        self.use_batch_norm = use_batch_norm
        self.c_len = c_len
        self.n_channels = n_channels
        self.print_model_summary = print_model_summary

        self.autoencoder_model = self.build_autoencoder()

        # self.autoencoder_model.compile(optimizer=tf.optimizers.Adam(),
        #                                loss='mse',
        #                                sample_weight_mode='temporal')

        if load_weights_from is not None:
            self.autoencoder_model.load_weights(load_weights_from)

    # 26 layer network: 13 for encoder and 13 for decoder (fully convolutional)
    # def convolutional_autoencoder_8(self):
    def build_autoencoder(self):

        # First build the Encoder Model
        inputs = keras.layers.Input(shape=(
            self.n_channels,
            self.height,
            self.width,
        ),
                                    name='encoder_input')

        # Permute the dimensions of the input tensor from NCHW to NHWC format
        # and replace nan with 0
        x = self.NchwToNhwcInput()(inputs)

        # Stacking  Conv2D, Activations, and AveragePooling layers blocks
        n_layers_and_n_filters = [1, self.n_filters, self.n_filters]
        n_layers_and_n_filters2 = [self.n_filters] * 3

        for filters in n_layers_and_n_filters:
            if filters == 1:
                filters = self.n_filters
            if self.height == 64:  # Assume square inputs
                kernel_to_use = (5, 5)
            else:
                kernel_to_use = self.kernel_size
            x = keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_to_use,
                strides=self.conv_stride,
                kernel_initializer='he_normal',
                activation=keras.layers.LeakyReLU(alpha=0.3),
                padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        for filters in n_layers_and_n_filters2:
            x = keras.layers.Conv2D(
                filters=filters,
                kernel_size=self.kernel_size,
                strides=self.conv_stride,
                kernel_initializer='he_normal',
                activation=keras.layers.LeakyReLU(alpha=0.3),
                padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        for filters in n_layers_and_n_filters2:
            x = keras.layers.Conv2D(
                filters=filters,
                kernel_size=self.kernel_size,
                strides=self.conv_stride,
                kernel_initializer='he_normal',
                activation=keras.layers.LeakyReLU(alpha=0.3),
                padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.AveragePooling2D(pool_size=self.pool_size_str,
                                          strides=self.pool_size_str,
                                          padding='same')(x)

        shape_here = keras.backend.int_shape(x)

        x = keras.layers.Conv2D(filters=int(self.c_len /
                                            (shape_here[1] * shape_here[2])),
                                kernel_size=self.kernel_size,
                                strides=self.conv_stride,
                                kernel_initializer='he_normal',
                                activation=keras.layers.LeakyReLU(alpha=0.3),
                                padding='same')(x)
        if self.use_batch_norm:
            x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        # Shape information to use in the decoder
        latent = x
        shape = keras.backend.int_shape(latent)

        # Instantiate Encoder Model
        encoder = Model(inputs, latent, name='encoder')
        if self.print_model_summary:
            encoder.summary()

        # Build the Decoder Model
        latent_inputs = keras.layers.Input(shape=(shape[1], shape[2],
                                                  shape[3]),
                                           name='decoder_input')
        x = latent_inputs

        # Stacking Transposed Conv2D, Activations, and Upsampling  blocks
        x = keras.layers.Conv2DTranspose(
            filters=int(self.c_len / (shape_here[1] * shape_here[2])),
            kernel_size=self.kernel_size,
            strides=self.conv_stride,
            kernel_initializer='he_normal',
            activation=keras.layers.LeakyReLU(alpha=0.3),
            padding='same')(x)
        if self.use_batch_norm:
            x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        for filters in n_layers_and_n_filters2[::-1]:
            x = keras.layers.Conv2DTranspose(
                filters=filters,
                kernel_size=self.kernel_size,
                strides=self.conv_stride,
                kernel_initializer='he_normal',
                activation=keras.layers.LeakyReLU(alpha=0.3),
                padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        for filters in n_layers_and_n_filters2[::-1]:
            x = keras.layers.Conv2DTranspose(
                filters=filters,
                kernel_size=self.kernel_size,
                strides=self.conv_stride,
                kernel_initializer='he_normal',
                activation=keras.layers.LeakyReLU(alpha=0.3),
                padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        x = keras.layers.UpSampling2D(size=self.pool_size_str,
                                      interpolation='bilinear')(x)

        for filters in n_layers_and_n_filters[::-1]:
            if filters == 1:
                filters = 1
            if self.height == 64:  # Assume square inputs
                kernel_to_use = (5, 5)
            else:
                kernel_to_use = self.kernel_size
            x = keras.layers.Conv2DTranspose(
                filters=filters,
                kernel_size=kernel_to_use,
                strides=self.conv_stride,
                kernel_initializer='he_normal',
                activation=keras.layers.LeakyReLU(alpha=0.3),
                padding='same')(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(axis=3, scale=False)(x)

        outputs = x

        # bring channels to the second dimension
        outputs = tf.transpose(outputs, perm=[0, 3, 1, 2])

        # Instantiate Decoder Model
        decoder = Model(latent_inputs, outputs, name='decoder')
        if self.print_model_summary:
            decoder.summary()

        # Instantiate Autoencoder Model
        autoencoder = Model(inputs,
                            decoder(encoder(inputs)),
                            name='autoencoder')
        if self.print_model_summary:
            autoencoder.summary()

        # Plot the Autoencoder Model
        # plot_the_model(encoder, decoder, autoencoder)
        return autoencoder

    @tf.function
    def call(self, inputs):
        """
        Args:
            inputs: input tensor of shape (batch_size, n_channels, height, width)
        Returns:
            output tensor of shape (batch_size, 1, height, width)
        """
        return self.autoencoder_model(inputs)

    def _compile(self,
                 optimizer,
                 loss,
                 metrics=None,
                 sample_weight_mode='temporal',
                 **kwargs):
        self.autoencoder_model.compile(optimizer=optimizer,
                                       loss=loss,
                                       metrics=metrics,
                                       sample_weight_mode=sample_weight_mode,
                                       **kwargs)

    def _fit(self, x, validation_data, epochs, callbacks, verbose):

        history = self.autoencoder_model.fit(x=x,
                                             validation_data=validation_data,
                                             epochs=epochs,
                                             callbacks=callbacks,
                                             verbose=verbose)
        return history

    def load_weights(self, load_weights_from):
        self.autoencoder_model.load_weights(load_weights_from)

    def evaluate(self, test_ds):
        return self.autoencoder_model.evaluate(test_ds)


class RadioUnetEstimator(NeuralNetworkEstimator):
    """
    Network in [levie2019radiounet] two UNets.
    """

    def __init__(self, load_weights_from=None, **kwargs):
        super(RadioUnetEstimator, self).__init__(**kwargs)

        self.first_unet = self.FirstUnet()
        self.second_unet = self.SecondUnet()
        if load_weights_from is not None:
            self.load_weights(load_weights_from).expect_partial()

    def call(self, inputs):
        """
        Args:
            inputs: input tensor of shape (batch_size, n_channels, height, width)
        Returns:
            output tensor of shape (batch_size, 1, height, width)
        """
        # convert to NHWC
        inputs = tf.transpose(inputs, perm=[0, 2, 3, 1])

        # replace nan with 0
        inputs = self.replace_nan_with_0(inputs)

        output1 = self.first_unet(inputs)
        outputs = self.second_unet(tf.concat([output1, inputs], axis=-1))

        return tf.transpose(outputs, perm=[0, 3, 1, 2])  # NCHW

    def _compile(self,
                 optimizer,
                 loss,
                 metrics=None,
                 sample_weight_mode='temporal',
                 **kwargs):
        self.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            #sample_weight_mode=sample_weight_mode,
            **kwargs)

    def _fit(self, x, validation_data, epochs, callbacks, verbose):
        """ This method first trains the FirstUnet and then the SecondUnet of the RadioUnet.
        """

        # First train the first unet
        self.first_unet.trainable = True
        self.second_unet.trainable = False

        history = super().fit(x=x,
                              validation_data=validation_data,
                              epochs=epochs,
                              callbacks=callbacks,
                              verbose=verbose)

        # Then train the second unet
        self.first_unet.trainable = False
        self.second_unet.trainable = True
        history = super().fit(x=x,
                              validation_data=validation_data,
                              epochs=epochs,
                              callbacks=callbacks,
                              verbose=verbose)

        return history

    class FirstUnet(Model):

        def __init__(self):
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
            return mean

    class SecondUnet(Model):

        def __init__(self):
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
                #   activation='exponential',
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
            return mean


def plot_the_model(encoder, decoder, autoencoder):
    ku.plot_model(encoder,
                  to_file='Models/encoder_model.pdf',
                  show_shapes=True,
                  show_layer_names=True)
    ku.plot_model(decoder,
                  to_file='Models/decoder_model.pdf',
                  show_shapes=True,
                  show_layer_names=True)
    ku.plot_model(autoencoder,
                  to_file='Models/autoencoder_model.pdf',
                  show_shapes=True,
                  show_layer_names=True)
