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


def build_generator(input_var=None, nfilters=[1024, 512, 256, 128, 3], filter_size=[4, 2, 2, 2, 2]):

    ###############################
    # Build Network Configuration #
    ###############################

    print ('... Building the generator')

    # Input of the network : shape = (batch_size, 100)
    input_layer = InputLayer(shape=(None, 100), input_var=input_var)

    # Reshape layer : output.shape = (batch_size, 100, 1, 1)
    reshape_layer = layers.ReshapeLayer(input_layer.output, (input_var.shape[0], 100, 1, 1))

    # Tranposed conv layer : output.shape = (batch_size, 1024, 4, 4)
    transconv_layer1 = TransposedConvLayer(reshape_layer, num_filters=nfilters[0], filter_size=filter_size[0])

    # Tranposed conv layer : output.shape = (batch_size, 512, 8, 8)
    transconv_layer2 = TransposedConvLayer(transconv_layer1.output, num_filters=nfilters[1], filter_size=filter_size[1])

    # Tranposed conv layer : output.shape = (batch_size, 256, 16, 16)
    transconv_layer3 = TransposedConvLayer(transconv_layer2.output, num_filters=nfilters[2], filter_size=filter_size[2])

    # Tranposed conv layer : output.shape = (batch_size, 128, 32, 32)
    transconv_layer4 = TransposedConvLayer(transconv_layer3.output, num_filters=nfilters[3], filter_size=filter_size[3])

    # Tranposed conv layer : output.shape = (batch_size, 3, 64, 64)
    transconv_layer5 = TransposedConvLayer(transconv_layer4.output, num_filters=nfilters[4], filter_size=filter_size[4])

    return transconv_layer5.output


def build_discriminator(input_var=None, nfilters=[128, 256, 512, 1024, 100], filter_size=[2, 2, 2, 2, 4],
                        input_channels=3):

    ###############################
    # Build Network Configuration #
    ###############################

    print ('... Building the discriminator')

    # Input of the network : shape = (batch_size, 3, 64, 64)
    input_layer = InputLayer(shape=(None, input_channels, 64, 64), input_var=input_var)

    # Conv layer : output.shape = (batch_size, 128, 32, 32)
    conv_layer1 = ConvLayer(input_layer.output, num_filters=nfilters[0], filter_size=filter_size[0])

    # Conv layer : output.shape = (batch_size, 256, 16, 16)
    conv_layer2 = ConvLayer(conv_layer1.output, num_filters=nfilters[1], filter_size=filter_size[1])

    # Conv layer : output.shape = (batch_size, 512, 8, 8)
    conv_layer3 = ConvLayer(conv_layer2.output, num_filters=nfilters[2], filter_size=filter_size[2])

    # Conv layer : output.shape = (batch_size, 1024, 4, 4)
    conv_layer4 = ConvLayer(conv_layer3.output, num_filters=nfilters[3], filter_size=filter_size[3])

    # Conv layer : output.shape = (batch_size, 100, 1, 1)
    conv_layer5 = ConvLayer(conv_layer4.output, num_filters=nfilters[4], filter_size=filter_size[4])

    # Dense Layer : output.shape = (batch_size, 1)
    dense_layer = DenseLayer(layers.FlattenLayer(conv_layer5.output), num_units=1, activation=sigmoid)

    return dense_layer.output