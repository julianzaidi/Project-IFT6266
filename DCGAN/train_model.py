import sys
import timeit
import theano
import lasagne
import numpy as np
import theano.tensor as T
import lasagne.layers as layers
import lasagne.objectives as objectives

from model import build_generator
from model import build_discriminator

sys.path.insert(0, '/home2/ift6ed67/Project-IFT6266/CNN_Autoencoder')
from utils import get_path
from utils import save_images
from utils import get_image
from utils import shared_GPU_data
from utils import assemble

theano.config.floatX = 'float32'


def train_model(learning_rate=0.0009, n_epochs=50, batch_size=200):
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
    valid_input_path = 'valid_input_'
    valid_target_path = 'valid_target_'
    nb_train_batch = 8
    nb_valid_batch = 4

    # Creating symbolic variables
    batch = 200
    max_size = 25
    input_channel = 3
    max_height = 64
    max_width = 64
    min_height = 32
    min_width = 32
    # Shape = (5000, 3, 64, 64)
    train_input = shared_GPU_data(shape=(batch * max_size, input_channel, max_height, max_width))
    valid_input = shared_GPU_data(shape=(batch * max_size, input_channel, max_height, max_width))
    # Shape = (5000, 3, 32, 32)
    train_target = shared_GPU_data(shape=(batch * max_size, input_channel, min_height, min_width))
    valid_target = shared_GPU_data(shape=(batch * max_size, input_channel, min_height, min_width))

    ######################
    # Building the model #
    ######################

    # Symbolic variables
    x_gen = T.matrix('x_gen', dtype=theano.config.floatX)
    x = T.tensor4('x', dtype=theano.config.floatX)
    index = T.lscalar()

    # Creation of the model
    generator = build_generator(input_var=x_gen)
    output_gen = layers.get_output(generator, deterministic=True)
    params_gen = layers.get_all_params(generator, trainable=True)
    discriminator = build_discriminator(input_var=x)
    output_dis = layers.get_output(discriminator, deterministic=True)
    params_dis = layers.get_all_params(discriminator, trainable=True)
    loss_gen = -T.mean(T.log(output_dis))
    loss_dis1 = -T.mean(T.log(output_dis))
    loss_dis2 = -T.mean(T.log(1 - output_dis))

    updates_gen = lasagne.updates.adam(loss_gen, params_gen, learning_rate=learning_rate)
    updates_dis1 = lasagne.updates.adam(loss_dis1, params_dis, learning_rate=learning_rate)
    updates_dis2 = lasagne.updates.adam(loss_dis2, params_dis, learning_rate=learning_rate)

    # Creation of theano functions
    train_dis1 = theano.function([index], loss_dis1, updates=updates_dis1, allow_input_downcast=True,
                                 givens={x: assemble(train_input[index * batch_size: (index + 1) * batch_size],
                                                     train_target[index * batch_size: (index + 1) * batch_size])})

    train_dis2 = theano.function([index], loss_dis2, updates=updates_dis2, allow_input_downcast=True,
                                 givens={x: assemble(train_input[index * batch_size: (index + 1) * batch_size],
                                                     output_gen),
                                         x_gen: np.random.rand(batch_size, 4096)})

    train_gen = theano.function([index], loss_gen, updates=updates_gen, allow_input_downcast=True,
                                givens={x: assemble(train_input[index * batch_size: (index + 1) * batch_size],
                                                    output_gen),
                                        x_gen: np.random.rand(batch_size, 4096)})

    valid_dis1 = theano.function([index], loss_dis1, allow_input_downcast=True,
                                 givens={x: assemble(valid_input[index * batch_size: (index + 1) * batch_size],
                                                     valid_target[index * batch_size: (index + 1) * batch_size])})

    valid_dis2 = theano.function([index], loss_dis2, allow_input_downcast=True,
                                 givens={x: assemble(valid_input[index * batch_size: (index + 1) * batch_size],
                                                     output_gen),
                                         x_gen: np.random.rand(batch_size, 4096)})

    valid_gen = theano.function([index], loss_gen, allow_input_downcast=True,
                                givens={x: assemble(valid_input[index * batch_size: (index + 1) * batch_size],
                                                    output_gen),
                                        x_gen: np.random.rand(batch_size, 4096)})

    idx = 50  # idx = index in this case
    pred_batch = 5
    predict_target = theano.function([index], output_gen, allow_input_downcast=True,
                                     givens={x_gen: np.random.rand(pred_batch, 4096)})

    ###################
    # Train the model #
    ###################

    print('... Training')

    best_validation_loss = np.inf
    best_iter = 0
    epoch = 0

    # Valid images chosen when a better model is found
    batch_verification = 0
    num_images = range(idx * pred_batch, (idx + 1) * pred_batch)

    start_time = timeit.default_timer()

    while (epoch < n_epochs):
        epoch = epoch + 1
        n_train_batches = 0
        for i in range(nb_train_batch):
            # Shape = (10000, 3, 64, 64) & Shape = (10000, 3, 32, 32)
            input, target = get_image(data_path, train_input_path, train_target_path, str(i))
            train_input.set_value(input[0: batch * max_size])
            train_target.set_value(target[0: batch * max_size])
            for j in range(max_size):
                cost = train_big_model(j)
                n_train_batches += 1
            train_input.set_value(input[batch * max_size:])
            train_target.set_value(target[batch * max_size:])
            for j in range(max_size):
                cost = train_big_model(j)
                n_train_batches += 1

        validation_losses = []
        for i in range(nb_valid_batch):
            # Shape = (10000, 3, 64, 64) & Shape = (10000, 3, 32, 32)
            input, target = get_image(data_path, valid_input_path, valid_target_path, str(i))
            valid_input.set_value(input[0: batch * max_size])
            valid_target.set_value(target[0: batch * max_size])
            for j in range(max_size):
                validation_losses.append(big_valid_loss(j))
            valid_input.set_value(input[batch * max_size:])
            valid_target.set_value(target[batch * max_size:])
            for j in range(max_size):
                validation_losses.append(big_valid_loss(j))

        this_validation_loss = np.mean(validation_losses)

        print('epoch %i, minibatch %i/%i, validation error %f %%' %
              (epoch, n_train_batches, n_train_batches, this_validation_loss * 100.))

        # if we got the best validation score until now
        if this_validation_loss < best_validation_loss:
            # save best validation score and iteration number
            best_validation_loss = this_validation_loss
            best_iter = epoch

            # save the model and a bunch of valid pictures
            print ('... saving model and valid images')

            np.savez('best_cnn_model.npz', *layers.get_all_param_values(model))
            # Shape = (10000, 3, 64, 64) & Shape = (10000, 3, 32, 32)
            input, target = get_image(data_path, valid_input_path, valid_target_path, str(batch_verification))
            small_valid_input.set_value(input[0: batch * min_valid_size])
            input = input[num_images]
            target = target[num_images]
            output = predict_target(idx)
            save_images(input=input, target=target, output=output, nbr_images=len(num_images), iteration=epoch)

    end_time = timeit.default_timer()

    print('Optimization complete.')
    print('Best validation score of %f %% obtained at epoch %i' %
          (best_validation_loss * 100., best_iter))
    print('The code ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    train_model()
