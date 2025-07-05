try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.layers import Conv1D, Conv2D, \
        Dense, MaxPooling1D, Conv2DTranspose, Flatten, \
        Dropout, BatchNormalization
except ImportError:
    print("Tensorflow not installed. Some functions will not work.")
import os
import numpy as np


class SpaceAverageEstimator:

    class Dnn(keras.Model):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.custom_layers = []
            self.custom_layers.append(
                Conv1D(filters=32,
                       kernel_size=3,
                       activation=tf.nn.leaky_relu,
                       padding='same'))
            self.custom_layers.append(
                Conv1D(filters=32,
                       kernel_size=3,
                       activation=tf.nn.leaky_relu,
                       padding='same'))
            self.custom_layers.append(
                MaxPooling1D(pool_size=2))
            self.custom_layers.append(
                Conv1D(filters=64,
                       kernel_size=3,
                       activation=tf.nn.leaky_relu,
                       padding='same'))
            self.custom_layers.append(
                Conv1D(filters=64,
                       kernel_size=3,
                       activation=tf.nn.leaky_relu,
                       padding='same'))
            self.custom_layers.append(
                MaxPooling1D(pool_size=2))
            self.custom_layers.append(
                Conv1D(filters=128,
                       kernel_size=3,
                       activation=tf.nn.leaky_relu,
                       padding='same'))

            self.custom_layers.append(
                Flatten())
            self.custom_layers.append(
                Dense(512, activation=tf.nn.leaky_relu))
            self.custom_layers.append(
                Dense(128, activation=tf.nn.leaky_relu))
            self.custom_layers.append(
                Dense(1, activation='linear'))

        def call(self, x):
            for layer in self.custom_layers:
                x = layer(x)
            return x

    def __init__(
            self,
            *args,
            load_weights_from=None,
            **kwargs):
        """
            Args:

            - 'load_weights_from': If different from None, the weights of the
              DNN are loaded from the file indicated by this argument.
        """

        super().__init__(*args, **kwargs)

        self.dnn = SpaceAverageEstimator.Dnn()

        if load_weights_from is not None:
            self.load_model(weights_filepath=load_weights_from)

    def train(self,
            x_data,
            y_data,
            epochs=10,
            batch_size=64,
            learning_rate=1e-5,
            verbose=1,
            validation_split=0.2,
            save_weights_to_folder=None):
        """

        Args:
            `x_data` is num_training_examples x len_f_kernel x 1
            `y_data` is num_training_examples length vector

            save_weights_to_folder: if not None, then the weights are saved to
            the folder with path `save_weights_to_folder` after every epoch.

        This function returns the history of the training.

        """
        self.dnn.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mean_squared_error',
            )

        if save_weights_to_folder is not None:
            os.makedirs(save_weights_to_folder, exist_ok=True)
            l_callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=save_weights_to_folder +
                             "/checkpoint-best.hdf5",
                    save_weights_only=True,
                    monitor='val_loss',
                    mode='min',
                    save_best_only=True,
                    verbose=1,
                    # save_freq=1,
                ),
                tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 patience=10,
                                                 mode='min',
                                                 restore_best_weights=True)
            ]
        else:
            l_callbacks = []

        # Train the model
        print("#################Training starts ###################")
        history = self.dnn.fit(
            x_data,
            y_data,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=l_callbacks,
            validation_split=validation_split,
        )
        print("#################Training completed ###################")

        return history.history

    def predict(self, wb_est_across_freq):
        """
        Args:
            `wb_est_across_freq`: [batch_size x len_f_kernel x 1] or [len_f_kernel x 1]

        """

        if np.ndim(wb_est_across_freq) == 2:
            wb_est_across_freq = wb_est_across_freq[None, ...]

            return self.dnn.predict(wb_est_across_freq)
        elif np.ndim(wb_est_across_freq) == 3:

            return self.dnn.predict(wb_est_across_freq)
        else:
            raise ValueError

    def load_model(self, weights_filepath):
        if not os.path.exists(weights_filepath):
            raise FileExistsError

        self.dnn(tf.zeros((1, 25, 1)))

        self.dnn.load_weights(weights_filepath)