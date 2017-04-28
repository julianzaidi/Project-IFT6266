"""
        Implementation of a conditional DCGAN
        """

import sys
import lasagne
import theano.tensor as T
import lasagne.layers as layers
from lasagne import nonlinearities

sys.path.insert(0, '/home2/ift6ed67/Project-IFT6266/Save_Dataset')
from generate_caption_batches import get_vocab_length


def integrate_captions(input_var=T.imatrix()):
    '''
            :param batch_size: number of images
            :param nb_caption: number of caption used per image
    '''

    ###############################
    # Build Network Configuration #
    ###############################

    print ('... Integrating captions to the model')

    # Input of the network : shape = (nb_caption, seq_length)
    network = layers.InputLayer(shape=(None, None), input_var=input_var)

    # Embedding layer : shape = (nb_caption, seq_length, 400)
    vocab_length = get_vocab_length()
    network = layers.EmbeddingLayer(network, vocab_length, output_size=400)

    # LSTM layer : shape = (nb_caption, 500)
    gate_parameters = layers.Gate(W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
                                  b=lasagne.init.Constant(0.))

    cell_parameters = layers.Gate(W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
                                  W_cell=None, b=lasagne.init.Constant(0.), nonlinearity=nonlinearities.tanh)

    network = layers.LSTMLayer(network, num_units=500, ingate=gate_parameters, forgetgate=gate_parameters,
                               cell=cell_parameters, outgate=gate_parameters, grad_clipping=100.,
                               only_return_final=True)

    # Dense Layer : shape = (nb_caption, 500)
    network = layers.DenseLayer(network, num_units=500)

    # Reshape layer : shape = (nb_caption, 500, 1, 1)
    network = layers.ReshapeLayer(network, (-1, 500, 1, 1))

    return network


def build_context_encoder(input_var1=None, input_var2=T.imatrix(), nfilters=[64, 128, 256, 512, 3000, 512, 256, 128, 3],
                          input_channels=3):
    ###############################
    # Build Network Configuration #
    ###############################

    print ('... Building the generator')

    leaky = nonlinearities.LeakyRectify(0.2)

    # Input of the network : shape = (batch_size, 3, 64, 64)
    network = layers.InputLayer(shape=(None, input_channels, 64, 64), input_var=input_var1)

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

    # Conv layer : shape = (batch_size, 3000, 1, 1)
    network = layers.batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=nfilters[4], filter_size=(4, 4),
                                                           stride=2, nonlinearity=leaky))

    # Integrating captions to the model : shape = (nb_caption, 500, 1, 1)
    captions = integrate_captions(input_var=input_var2)  # nb_caption = batch_size

    # Concatenation : shape = (batch_size, 3500, 1, 1)
    network = layers.ConcatLayer([network, captions])

    # Tranposed conv layer : shape = (batch_size, 512, 4, 4)
    network = layers.batch_norm(layers.TransposedConv2DLayer(network, num_filters=nfilters[5], filter_size=(4, 4),
                                                             stride=(1, 1)))

    # Tranposed conv layer : shape = (batch_size, 256, 8, 8)
    network = layers.batch_norm(layers.TransposedConv2DLayer(network, num_filters=nfilters[6], filter_size=(5, 5),
                                                             stride=(2, 2), crop=2, output_size=8))

    # Tranposed conv layer : shape = (batch_size, 128, 16, 16)
    network = layers.batch_norm(layers.TransposedConv2DLayer(network, num_filters=nfilters[7], filter_size=(5, 5),
                                                             stride=(2, 2), crop=2, output_size=16))

    # Tranposed conv layer : shape = (batch_size, 3, 32, 32)
    network = layers.TransposedConv2DLayer(network, num_filters=nfilters[8], filter_size=5, stride=2, crop=2,
                                           output_size=32, nonlinearity=nonlinearities.sigmoid)

    return network


def build_discriminator(input_var=None, nfilters=[128, 256, 512], input_channels=3):
    ###############################
    # Build Network Configuration #
    ###############################

    print ('... Building the discriminator')

    leaky = nonlinearities.LeakyRectify(0.2)

    # Input of the network : shape = (batch_size, 3, 32, 32)
    network = layers.InputLayer(shape=(None, input_channels, 32, 32), input_var=input_var)

    # Conv layer : shape = (batch_size, 128, 16, 16)
    network = layers.Conv2DLayer(network, num_filters=nfilters[0], filter_size=(5, 5), stride=2, pad=2,
                                 nonlinearity=leaky)

    # Conv layer : shape = (batch_size, 256, 8, 8)
    network = layers.batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=nfilters[1], filter_size=(5, 5),
                                                           stride=2, pad=2, nonlinearity=leaky))

    # Conv layer : shape = (batch_size, 512, 4, 4)
    network = layers.batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=nfilters[2], filter_size=(5, 5),
                                                           stride=2, pad=2, nonlinearity=leaky))

    # Flatten layer :shape = (batch_size, 8192)
    network = lasagne.layers.FlattenLayer(network)

    # Dense layer :shape = (batch_size, 1)
    network = lasagne.layers.DenseLayer(network, 1, nonlinearity=lasagne.nonlinearities.sigmoid)

    return network