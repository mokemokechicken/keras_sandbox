"""This script demonstrates how to build a variational autoencoder with Keras.

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114

https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py
"""

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist

from ..util.option import standard_option
from .model import VAEModel

args = standard_option().parse_args()

batch_size = 16
original_dim = 784
latent_dim = 2
intermediate_dim = 128
epsilon_std = 0.01
nb_epoch = 40 if args.epoch is None else args.epoch

# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


vae = VAEModel(original_dim, intermediate_dim, latent_dim, batch_size)
vae.build()
vae.load_if_exists(args.model)
vae.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        validation_data=(x_test, x_test))
vae.save(args.model)


# display a 2D plot of the digit classes in the latent space
x_test_encoded = vae.encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.savefig('%s_encode.png' % args.name)
# plt.show()

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# we will sample n points within [-15, 15] standard deviations
grid_x = np.linspace(-15, 15, n)
grid_y = np.linspace(-15, 15, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]]) * epsilon_std
        x_decoded = vae.generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.savefig('%s_generate.png' % args.name)
# plt.show()

