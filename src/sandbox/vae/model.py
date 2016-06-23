# coding: utf8
import os

from keras import backend as K
from keras import objectives
from keras.layers import Input, Dense, Lambda
from keras.models import Model

from ..util import logger
from ..util.file_util import create_basedir


class VAEModel:
    x = z_mean = decoder_h = decoder_mean = None
    vae = None
    _generator = _encoder = None

    def __init__(self, original_dim, intermediate_dim, latent_dim, batch_size, epsilon_std=0.01):
        self.original_dim = original_dim
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epsilon_std = epsilon_std

    def build(self):
        self.x = x = Input(batch_shape=(self.batch_size, self.original_dim))
        h = Dense(self.intermediate_dim, activation='relu')(x)
        self.z_mean = z_mean = Dense(self.latent_dim)(h)
        z_log_std = Dense(self.latent_dim)(h)

        def sampling(args):
            sz_mean, sz_log_std = args
            epsilon = K.random_normal(shape=(self.batch_size, self.latent_dim),
                                      mean=0., std=self.epsilon_std)
            return sz_mean + K.exp(sz_log_std) * epsilon

        # note that "output_shape" isn't necessary with the TensorFlow backend
        # so you could write `Lambda(sampling)([z_mean, z_log_std])`
        z = Lambda(sampling, output_shape=(self.latent_dim,))([z_mean, z_log_std])

        # we instantiate these layers separately so as to reuse them later
        self.decoder_h = decoder_h = Dense(self.intermediate_dim, activation='relu')
        self.decoder_mean = decoder_mean = Dense(self.original_dim, activation='sigmoid')
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)

        def vae_loss(vx, vx_decoded_mean):
            xent_loss = objectives.binary_crossentropy(vx, vx_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_std - K.square(z_mean) - K.exp(z_log_std), axis=-1)
            return xent_loss + kl_loss

        self.vae = vae = Model(x, x_decoded_mean)
        vae.compile(optimizer='rmsprop', loss=vae_loss)

    def fit(self, x_train, y_train, **kwargs):
        self.vae.fit(x_train, y_train, **kwargs)

    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = Model(self.x, self.z_mean)
        return self._encoder

    @property
    def generator(self):
        if self._generator is None:
            decoder_input = Input(shape=(self.latent_dim,))
            _h_decoded = self.decoder_h(decoder_input)
            _x_decoded_mean = self.decoder_mean(_h_decoded)
            self._generator = Model(decoder_input, _x_decoded_mean)
        return self._generator

    def save(self, file_path):
        if file_path is None:
            return
        logger.info("save weights to %s" % file_path)
        create_basedir(file_path)
        self.vae.save_weights(filepath=file_path, overwrite=True)

    def load(self, file_path):
        logger.info("load weights from %s" % file_path)
        self.vae.load_weights(filepath=file_path)

    def load_if_exists(self, file_path):
        if os.path.exists(file_path):
            self.load(file_path)
            return True
        return False

