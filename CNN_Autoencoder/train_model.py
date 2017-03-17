import timeit
import theano
import lasagne
import numpy as np
import theano.tensor as T
import lasagne.layers as layers
import lasagne.objectives as objectives

from model import build_model
from utils import save_images
from utils import get_path

theano.config.floatX = 'float32'


def shared_GPU_data(shape, dtype=theano.config.floatX, borrow=True):
    '''
            Create theano shared variables to be load on the GPU
    '''
    return theano.shared(np.zeros(shape=shape, dtype=dtype), borrow=borrow)


def train_model(learning_rate=0.0002, n_epochs=20, batch_size=200, dataset='normalized_mscoco_dataset.npz'):
    '''
            Function that compute the training of the model
            '''

    #######################
    # Loading the dataset #
    #######################

    print ('... Loading data')

    # Load the dataset on the CPU
    data_path = get_path()
    dataset = np.load(data_path + dataset)

    train_input_data = dataset['train_input']  # Shape = (82611, 3, 64, 64)
    train_target_data = dataset['train_target']  # Shape = (82611, 3, 32, 32)
    valid_input_data = dataset['valid_input']  # Shape = (40438, 3, 64, 64)
    valid_target_data = dataset['valid_target']  # Shape = (40438, 3, 32, 32)

    # Creating symbolic variables
    max_size = 50
    min_train_size = 13
    min_valid_size = 2
    input_channel = 3
    max_height = 64
    max_width = 64
    min_height = 32
    min_width = 32
    nb_train_batch = 9
    nb_valid_batch = 5
    # Shape = (10000, 3, 64, 64)
    big_train_input = shared_GPU_data(shape=(batch_size * max_size, input_channel, max_height, max_width))
    big_valid_input = shared_GPU_data(shape=(batch_size * max_size, input_channel, max_height, max_width))
    # Shape = (10000, 3, 32, 32)
    big_train_target = shared_GPU_data(shape=(batch_size * max_size, input_channel, min_height, min_width))
    big_valid_target = shared_GPU_data(shape=(batch_size * max_size, input_channel, min_height, min_width))
    # Shape = (2600, 3, 64, 64)
    small_train_input = shared_GPU_data(shape=(batch_size * min_train_size, input_channel, max_height, max_width))
    # Shape = (2600, 3, 32, 32)
    small_train_target = shared_GPU_data(shape=(batch_size * min_train_size, input_channel, min_height, min_width))
    # Shape = (400, 3, 64, 64)
    small_valid_input = shared_GPU_data(shape=(batch_size * min_valid_size, input_channel, max_height, max_width))
    # Shape = (400, 3, 32, 32)
    small_valid_target = shared_GPU_data(shape=(batch_size * min_valid_size, input_channel, min_height, min_width))

    ###################
    # Building the model #
    ###################

    # Symbolic variables
    x = T.tensor4('x', dtype=theano.config.floatX)
    y = T.tensor4('y', dtype=theano.config.floatX)
    index = T.lscalar()

    # Creation of the model
    model = build_model(input_var=x)
    output = layers.get_output(model, deterministic=True)
    params = layers.get_all_params(model, trainable=True)
    loss = T.mean(objectives.squared_error(output, y))
    updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)

    # Creation of theano functions
    train_big_model = theano.function([index], loss, updates=updates, allow_input_downcast=True,
                                      givens={x: big_train_input[index * batch_size: (index + 1) * batch_size],
                                              y: big_train_target[index * batch_size: (index + 1) * batch_size]})

    train_small_model = theano.function([index], loss, updates=updates, allow_input_downcast=True,
                                        givens={x: small_train_input[index * batch_size: (index + 1) * batch_size],
                                                y: small_train_target[index * batch_size: (index + 1) * batch_size]})

    big_valid_loss = theano.function([index], loss, allow_input_downcast=True,
                                     givens={x: big_valid_input[index * batch_size: (index + 1) * batch_size],
                                             y: big_valid_target[index * batch_size: (index + 1) * batch_size]})

    small_valid_loss = theano.function([index], loss, allow_input_downcast=True,
                                       givens={x: small_valid_input[index * batch_size: (index + 1) * batch_size],
                                               y: small_valid_target[index * batch_size: (index + 1) * batch_size]})

    idx = 50  # idx = index in this case
    batch = 5
    predict_target = theano.function([index], output, allow_input_downcast=True,
                                     givens={x: big_valid_input[index * batch: (index + 1) * batch]})

    ###################
    # Train the model #
    ###################

    print('... Training')

    best_validation_loss = np.inf
    best_iter = 0
    epoch = 0

    # Valid images chosen when a better model is found
    num_images = range(idx * batch, (idx + 1) * batch)
    input = valid_input_data[num_images]
    target = valid_target_data[num_images]

    start_time = timeit.default_timer()

    while (epoch < n_epochs):
        epoch = epoch + 1
        n_train_batches = 0
        for i in range(nb_train_batch):
            if nb_train_batch == 8:
                small_train_input.set_value(train_input_data[batch_size * max_size * nb_train_batch:
                                            batch_size * (nb_train_batch * max_size + min_train_size)])
                small_train_target.set_value(train_target_data[batch_size * max_size * nb_train_batch:
                                             batch_size * (nb_train_batch * max_size + min_train_size)])
                for j in range(min_train_size):
                    train_small_model(j)
                    n_train_batches += 1
            else:
                big_train_input.set_value(train_input_data[batch_size * max_size * nb_train_batch:
                                          batch_size * max_size * (nb_train_batch + 1)])
                big_train_target.set_value(train_target_data[batch_size * max_size * nb_train_batch:
                                           batch_size * max_size * (nb_train_batch + 1)])
                for j in range(max_size):
                    train_big_model(j)
                    n_train_batches += 1

        validation_losses = []
        for i in range(nb_valid_batch):
            if nb_valid_batch == 4:
                small_valid_input.set_value(valid_input_data[batch_size * max_size * nb_valid_batch:
                                            batch_size * (nb_valid_batch * max_size + min_valid_size)])
                small_valid_target.set_value(valid_target_data[batch_size * max_size * nb_valid_batch:
                                             batch_size * (nb_valid_batch * max_size + min_valid_size)])
                for j in range(min_valid_size):
                    validation_losses.append(small_valid_loss(j))
            else:
                big_valid_input.set_value(valid_input_data[batch_size * max_size * nb_valid_batch:
                                                           batch_size * max_size * (nb_valid_batch + 1)])
                big_valid_target.set_value(valid_target_data[batch_size * max_size * nb_valid_batch:
                                           batch_size * max_size * (nb_valid_batch + 1)])
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

            np.savez('best_model.npz', *layers.get_all_param_values(model))
            big_valid_input.set_value(valid_input_data[0: 10000])
            output = predict_target(idx)
            save_images(input=input, target=target, output=output, nbr_images=len(num_images), iteration=epoch)

    end_time = timeit.default_timer()

    print('Optimization complete.')
    print('Best validation score of %f %% obtained at epoch %i' %
          (best_validation_loss * 100., best_iter))
    print('The code ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    train_model()
