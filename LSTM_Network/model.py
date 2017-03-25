"""
        Implementation of a LSTM network to learn the caption
        """

import lasagne
import lasagne.layers as layers
from lasagne.nonlinearities import rectify
from generate_batches import get_vocab_length


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


class EmbeddingLayer(object):
    def __init__(self, input, input_size, embedding_size):
        """
                Allocate an Embedding Layer.

                """

        self.input = input
        self.output = layers.EmbeddingLayer(self.input, input_size, embedding_size, W=initialize_parameters()[0])


class LSTMLayer(object):
    def __init__(self, input, n_hidden=500, grad_clip=100., only_return_final=True):

        self.input = input

        gate_parameters = layers.Gate(W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
                                      b=initialize_parameters()[1])

        cell_parameters = layers.Gate(W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
                                      W_cell=None, b=initialize_parameters()[1],
                                      nonlinearity=lasagne.nonlinearities.tanh)

        self.output = layers.LSTMLayer(self.input, n_hidden, ingate=gate_parameters, forgetgate=gate_parameters,
                                       cell=cell_parameters, outgate=gate_parameters, grad_clipping=grad_clip,
                                       only_return_final=only_return_final)


def build_model(input_var=None, nfilters=[50, 20, 3], filter_size=[7, 2, 4]):
    '''
            :param batch_size: number of images
            :param nb_caption: number of caption used per image
    '''


    ###############################
    # Build Network Configuration #
    ###############################

    print '... Building the model'

    # Input of the network : shape = (nb_caption, seq_length)
    input_layer = InputLayer(shape=(None, None), input_var=input_var)

    # Embedding layer : output.shape = (nb_caption, seq_length, embedding_size)
    vocab_length = get_vocab_length()
    embedded_layer = EmbeddingLayer(input_layer.output, vocab_length, embedding_size=400)

    # LSTM layer : output.shape = (nb_caption, n_hidden)
    lstm_layer = LSTMLayer(embedded_layer.output, n_hidden=500)

    # Dense Layer : output.shape = (nb_caption, 500)
    dense_layer1 = DenseLayer(lstm_layer.output, num_units=500)

    # Reshape layer : output.shape = (nb_caption, 500, 1, 1)
    reshape_layer = layers.ReshapeLayer(dense_layer1.output, (input_var.shape[0], 500, 1, 1))

    # Tranposed conv layer : output.shape = (nb_caption, 50, 7, 7)
    transconv_layer1 = TransposedConvLayer(reshape_layer, num_filters=nfilters[0], filter_size=filter_size[0])

    # Unpool layer : output.shape = (nb_caption, 50, 14, 14)
    unpool_layer1 = UnpoolLayer(transconv_layer1.output)

    # Tranposed conv layer : output.shape = (nb_caption, 20, 15, 15)
    transconv_layer2 = TransposedConvLayer(unpool_layer1.output, num_filters=nfilters[1], filter_size=filter_size[1],
                                           stride=(1, 1))

    # Tranposed conv layer : output.shape = (nb_caption, 3, 32, 32)
    transconv_layer3 = TransposedConvLayer(transconv_layer2.output, num_filters=nfilters[2], filter_size=filter_size[2])

    return transconv_layer3.output


