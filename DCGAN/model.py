"""
        Implementation of a CNN autoencoder
        """

import lasagne
import lasagne.layers as layers
from lasagne import nonlinearities


def build_generator(input_var=None, noise_size=100, nfilters=[512, 256, 128, 64, 3]):

    ###############################
    # Build Network Configuration #
    ###############################

    print ('... Building the generator')

    # Input of the network : shape = (batch_size, 100)
    network = layers.InputLayer(shape=(None, noise_size), input_var=input_var)

    # Reshape layer : shape = (batch_size, 100, 1, 1)
    network = layers.ReshapeLayer(network, (-1, noise_size, 1, 1))

    # Tranposed conv layer : shape = (batch_size, 512, 4, 4)
    network = layers.batch_norm(layers.TransposedConv2DLayer(network, num_filters=nfilters[0], filter_size=(4, 4),
                                                             stride=(1, 1)))

    # Tranposed conv layer : shape = (batch_size, 256, 8, 8)
    network = layers.batch_norm(layers.TransposedConv2DLayer(network, num_filters=nfilters[1], filter_size=(5, 5),
                                                             stride=(2, 2), crop=2, output_size=8))

    # Tranposed conv layer : shape = (batch_size, 128, 16, 16)
    network = layers.batch_norm(layers.TransposedConv2DLayer(network, num_filters=nfilters[2], filter_size=(5, 5),
                                                             stride=(2, 2), crop=2, output_size=16))

    # Tranposed conv layer : shape = (batch_size, 64, 32, 32)
    network = layers.batch_norm(layers.TransposedConv2DLayer(network, num_filters=nfilters[3], filter_size=(5, 5),
                                                             stride=(2, 2), crop=2, output_size=32))

    # Tranposed conv layer : shape = (batch_size, 3, 64, 64)
    network = layers.TransposedConv2DLayer(network, num_filters=nfilters[4], filter_size=5, stride=2, crop=2,
                                           output_size=64, nonlinearity=nonlinearities.sigmoid)

    return network


def build_discriminator(input_var=None, nfilters=[64, 128, 256, 512], input_channels=3):

    ###############################
    # Build Network Configuration #
    ###############################

    print ('... Building the discriminator')

    leaky = nonlinearities.LeakyRectify(0.2)

    # Input of the network : shape = (batch_size, 3, 64, 64)
    network = layers.InputLayer(shape=(None, input_channels, 64, 64), input_var=input_var)

    # Conv layer : shape = (batch_size, 64, 32, 32)
    network = layers.Conv2DLayer(network, num_filters=nfilters[0], filter_size=(5, 5), stride=2, pad=2,
                                 nonlinearity=leaky)

    # Conv layer : shape = (batch_size, 128, 16, 16)
    network = layers.batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=nfilters[1], filter_size=(5, 5),
                                                           stride=2, pad=2, nonlinearity=leaky))

    # Conv layer : shape = (batch_size, 256, 8, 8)
    network = layers.batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=nfilters[2], filter_size=(5, 5),
                                                           stride=2, pad=2, nonlinearity=leaky))

    # Conv layer : shape = (batch_size, 512, 4, 4)
    network = layers.batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=nfilters[3], filter_size=(5, 5),
                                                           stride=2, pad=2, nonlinearity=leaky))

    # Flatten layer :shape = (batch_size, 8192)
    network = lasagne.layers.FlattenLayer(network)

    # Dense layer :shape = (batch_size, 1)
    network = lasagne.layers.DenseLayer(network, 1, nonlinearity=lasagne.nonlinearities.sigmoid)

    return network