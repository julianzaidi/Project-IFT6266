"""
        Implementation of a CNN autoencoder
        """

import lasagne
import lasagne.layers as layers
from lasagne.nonlinearities import rectify
from lasagne.nonlinearities import sigmoid


#########################
# Layers implementation #
#########################

def initialize_parameters():
    W = lasagne.init.Normal()
    b = lasagne.init.Constant(0.)

    return [W, b]


class InputLayer(object):
    def __init__(self, shape, input_var=None):
        """
                Input of the Network

                """

        self.output = layers.InputLayer(shape, input_var=input_var)


class DenseLayer(object):
    def __init__(self, input, num_units, activation=rectify):
        """
                Typical hidden layer of a MLP: units are fully-connected

                NOTE : The non linearity used here is relu

                """

        self.input = input
        self.output = layers.DenseLayer(self.input, num_units, W=initialize_parameters()[0],
                                        b=initialize_parameters()[1],
                                        nonlinearity=activation)


class ConvLayer(object):
    def __init__(self, input, num_filters, filter_size, stride=(1, 1), pad=(0, 0), activation=rectify):
        """
                Allocate a ConvLayer with shared variable internal parameters

                """

        self.input = input
        self.output = layers.Conv2DLayer(self.input, num_filters, filter_size, stride=stride, pad=pad,
                                         W=initialize_parameters()[0], b=initialize_parameters()[1],
                                         nonlinearity=activation)


class PoolLayer(object):
    def __init__(self, input, poolsize=(2, 2), stride=None, padding=(0, 0), mode='max'):
        """
                Allocate a PoolLayer

                """

        self.input = input
        self.output = layers.Pool2DLayer(self.input, poolsize, stride=stride, pad=padding, ignore_border=True,
                                         mode=mode)


class UnpoolLayer(object):
    def __init__(self, input, scale=(2, 2)):
        """
                Allocate an UnpoolLayer

                """

        self.input = input
        self.output = layers.Upscale2DLayer(self.input, scale)


class TransposedConvLayer(object):
    def __init__(self, input, num_filters, filter_size, stride=(2, 2), padding=(0, 0), activation=rectify):
        """
                Allocate a TransposedConvLayer with shared variable internal parameters.

                """

        self.input = input
        self.output = layers.TransposedConv2DLayer(self.input, num_filters, filter_size, stride=stride, crop=padding,
                                                   W=initialize_parameters()[0], b=initialize_parameters()[1],
                                                   nonlinearity=activation)


def build_generator(input_var=None, nfilters=[1000, 500, 250, 3], filter_size=[3, 3, 3, 4]):

    ###############################
    # Build Network Configuration #
    ###############################

    print ('... Building the generator')

    # Input of the network : shape = (batch_size, 4096=4*32*32)
    input_layer = InputLayer(shape=(None, 4096), input_var=input_var)

    # Reshape layer : output.shape = (batch_size, 4096, 1, 1)
    reshape_layer = layers.ReshapeLayer(input_layer.output, (input_var.shape[0], 4096, 1, 1))

    # Tranposed conv layer : output.shape = (batch_size, 1000, 3, 3)
    transconv_layer1 = TransposedConvLayer(reshape_layer, num_filters=nfilters[0], filter_size=filter_size[0])

    # Tranposed conv layer : output.shape = (batch_size, 500, 7, 7)
    transconv_layer2 = TransposedConvLayer(transconv_layer1.output, num_filters=nfilters[1], filter_size=filter_size[1])

    # Tranposed conv layer : output.shape = (batch_size, 250, 15, 15)
    transconv_layer3 = TransposedConvLayer(transconv_layer2.output, num_filters=nfilters[2], filter_size=filter_size[2])

    # Tranposed conv layer : output.shape = (batch_size, 3, 32, 32)
    transconv_layer4 = TransposedConvLayer(transconv_layer3.output, num_filters=nfilters[3], filter_size=filter_size[3])

    return transconv_layer4.output


def build_discriminator(input_var=None, nfilters=[500, 250, 100, 50, 1], filter_size=[5, 5, 3, 3, 3], input_channels=3):

    ###############################
    # Build Network Configuration #
    ###############################

    print ('... Building the discriminator')

    # Input of the network : shape = (batch_size, 3, 64, 64)
    input_layer = InputLayer(shape=(None, input_channels, 64, 64), input_var=input_var)

    # Conv layer : output.shape = (batch_size, 500, 62, 62)
    conv_layer1 = ConvLayer(input_layer.output, num_filters=nfilters[0], filter_size=filter_size[0], pad=(1, 1))

    # Conv layer : output.shape = (batch_size, 250, 58, 58)
    conv_layer2 = ConvLayer(conv_layer1.output, num_filters=nfilters[1], filter_size=filter_size[1])

    # Pooling layer : output.shape = (batch_size, 250, 29, 29)
    pool_layer1 = PoolLayer(conv_layer2.output)

    # Conv layer : output.shape = (batch_size, 100, 27, 27)
    conv_layer3 = ConvLayer(pool_layer1.output, num_filters=nfilters[2], filter_size=filter_size[2])

    # Conv layer : output.shape = (batch_size, 50, 25, 25)
    conv_layer4 = ConvLayer(conv_layer3.output, num_filters=nfilters[3], filter_size=filter_size[3])

    # Pooling layer : output.shape = (batch_size, 50, 12, 12)
    pool_layer2 = PoolLayer(conv_layer4.output)

    # Conv layer : output.shape = (batch_size, 1, 4, 4)
    conv_layer5 = ConvLayer(pool_layer2.output, num_filters=nfilters[4], filter_size=filter_size[4], stride=(3, 3))

    # Pooling layer : output.shape = (batch_size, 1, 1, 1)
    pool_layer3 = PoolLayer(conv_layer5.output, poolsize=(4, 4))

    # Dense Layer : output.shape = (batch_size, 1)
    dense_layer = DenseLayer(layers.FlattenLayer(pool_layer3.output), num_units=1, activation=sigmoid)

    return dense_layer.output