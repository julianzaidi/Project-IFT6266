import sys
import timeit
import theano
import lasagne
import numpy as np
import theano.tensor as T
import lasagne.layers as layers
# import lasagne.objectives as objectives

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as AA
from mpl_toolkits.axes_grid1 import host_subplot

from model import build_generator
from model import build_discriminator

sys.path.insert(0, '/home2/ift6ed67/Project-IFT6266/CNN_Autoencoder')
from utils import get_path
# from utils import save_images
from utils import get_image
from utils import shared_GPU_data
from utils import assemble
from utils import random_sample

theano.config.floatX = 'float32'


def train_model(learning_rate_dis=0.0009, learning_rate_gen=0.0005, n_epochs=5, batch_size=200):
    '''
            Function that compute the training of the model
            '''

    #######################
    # Loading the dataset #
    #######################

    print ('... Loading data')

    # Load the dataset on the CPU
    data_path = get_path()
    train_input_path = 'train_input_'
    train_target_path = 'train_target_'
    nb_train_batch = 8

    # Creating symbolic variables
    input_channel = 3
    max_height = 64
    max_width = 64
    # Shape = (5000, 3, 64, 64)
    image = shared_GPU_data(shape=(batch_size, input_channel, max_height, max_width))
    # Shape = (5000, 100)
    random_matrix = shared_GPU_data(shape=(batch_size, 100))

    ######################
    # Building the model #
    ######################

    # Symbolic variables
    x_gen = T.matrix('x_gen', dtype=theano.config.floatX)
    x = T.tensor4('x', dtype=theano.config.floatX)
    # index = T.lscalar()

    # Creation of the model
    generator = build_generator(input_var=x_gen)
    params_gen = layers.get_all_params(generator, trainable=True)
    output_gen = layers.get_output(generator, deterministic=True)
    discriminator = build_discriminator(input_var=x)
    params_dis = layers.get_all_params(discriminator, trainable=True)
    model_dis = layers.get_output(discriminator, inputs=output_gen, deterministic=True)
    real_dis = layers.get_output(discriminator, deterministic=True)
    loss_gen = -T.mean(T.log(model_dis))
    loss_dis = -T.mean(T.log(real_dis) + T.log(1 - model_dis))

    updates_gen = lasagne.updates.adam(loss_gen, params_gen, learning_rate=learning_rate_gen)
    updates_dis = lasagne.updates.adam(loss_dis, params_dis, learning_rate=learning_rate_dis)

    # Creation of theano functions
    train_dis = theano.function([], loss_dis, updates=updates_dis, allow_input_downcast=True,
                                givens={x: image, x_gen: random_matrix})

    train_gen = theano.function([], loss_gen, updates=updates_gen, allow_input_downcast=True,
                                givens={x_gen: random_matrix})

    pred_batch = 5
    predict_image = theano.function([], output_gen, allow_input_downcast=True,
                                    givens={x_gen: random_sample(size=(pred_batch, 100))})

    ###################
    # Train the model #
    ###################

    print('... Training')

    epoch = 0
    nb_train_dis = 10
    nb_train_gen = 1
    nb_batch = 10000 // batch_size
    nb_block = nb_batch // nb_train_dis

    start_time = timeit.default_timer()

    while (epoch < n_epochs):
        epoch = epoch + 1
        loss_dis = []
        loss_gen = []
        for i in range(nb_train_batch):
            print (i)
            # Shape = (10000, 3, 64, 64) & Shape = (10000, 3, 32, 32)
            input, target = get_image(data_path, train_input_path, train_target_path, str(i))
            # Shape = (10000, 3, 64, 64)
            input = assemble(input, target)
            # Shape = (10000, 100)
            sample = random_sample(size=(10000, 100))
            for j in range(nb_block):
                print (j)
                for index in range(nb_train_dis * j, nb_train_dis * (j + 1)):
                    print (index)
                    image.set_value(input[index * batch_size: (index + 1) * batch_size])
                    random_matrix.set_value(sample[index * batch_size: (index + 1) * batch_size])
                    loss = train_dis()
                    loss_dis.append(loss)
                for index in range(nb_train_gen * j, nb_train_gen * (j + 1)):
                    print (index)
                    random_matrix.set_value(sample[index * batch_size: (index + 1) * batch_size])
                    loss = train_gen()
                    loss_gen.append(loss)

        # Plot the learning curve
        ax1 = host_subplot(111, axes_class=AA.Axes)
        plt.subplots_adjust(right=0.75)
        ax2 = ax1.twiny()

        x1 = range(1, len(loss_dis) + 1)
        ax1.set_xlim([x1[0], x1[-1]])
        x2 = range(1, len(loss_gen) + 1)
        ax2.set_xlim([x2[0], x2[-1]])

        ax1.set_xlabel('mini batch (Discriminator)', color='g')
        ax2.set_xlabel('mini batch (Generator)', color='b')
        ax1.set_ylabel('Loss')

        ax1.plot(x1, loss_dis, 'g', label='Discriminator loss')
        ax2.plot(x2, loss_gen, 'b', label='Generator Loss')

        ax1.grid(True)
        ax1.legend()

        plt.savefig('Learning_curve_epoch' + str(epoch))

        if epoch % 5 == 0:
            # save the model and a bunch of generated pictures
            print ('... saving model and generated images')

            np.savez('discriminator_epoch' + str(epoch) + '.npz', *layers.get_all_param_values(discriminator))
            np.savez('generator_epoch' + str(epoch) + '.npz', *layers.get_all_param_values(generator))
            generated_images = predict_image()

            for i in range(pred_batch):
                plt.subplot(1, pred_batch, (i + 1))
                plt.axis('off')
                plt.imshow(generated_images[i, :, :, :].transpose(1, 2, 0))

            plt.savefig('generated_images_epoch' + str(epoch) + '.png', bbox_inches='tight')

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('The code ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    train_model()
