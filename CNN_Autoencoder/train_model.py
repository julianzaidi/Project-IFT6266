import timeit
import theano
import lasagne
import numpy as np
import theano.tensor as T
import lasagne.layers as layers
import lasagne.objectives as objectives

#from model import build_model1
from model import build_model2
from utils import get_path
from utils import save_images
from utils import get_image
from utils import shared_GPU_data

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
    nb_train_batch = 9
    nb_valid_batch = 5

    # Creating symbolic variables
    batch = 200
    max_size = 25
    min_train_size = 13
    min_valid_size = 2
    input_channel = 3
    max_height = 64
    max_width = 64
    min_height = 32
    min_width = 32
    # Shape = (5000, 3, 64, 64)
    big_train_input = shared_GPU_data(shape=(batch * max_size, input_channel, max_height, max_width))
    big_valid_input = shared_GPU_data(shape=(batch * max_size, input_channel, max_height, max_width))
    # Shape = (5000, 3, 32, 32)
    big_train_target = shared_GPU_data(shape=(batch * max_size, input_channel, min_height, min_width))
    big_valid_target = shared_GPU_data(shape=(batch * max_size, input_channel, min_height, min_width))
    # Shape = (2600, 3, 64, 64)
    small_train_input = shared_GPU_data(shape=(batch * min_train_size, input_channel, max_height, max_width))
    # Shape = (2600, 3, 32, 32)
    small_train_target = shared_GPU_data(shape=(batch * min_train_size, input_channel, min_height, min_width))
    # Shape = (400, 3, 64, 64)
    small_valid_input = shared_GPU_data(shape=(batch * min_valid_size, input_channel, max_height, max_width))
    # Shape = (400, 3, 32, 32)
    small_valid_target = shared_GPU_data(shape=(batch * min_valid_size, input_channel, min_height, min_width))

    ###################
    # Building the model #
    ###################

    # Symbolic variables
    x = T.tensor4('x', dtype=theano.config.floatX)
    y = T.tensor4('y', dtype=theano.config.floatX)
    index = T.lscalar()

    # Creation of the model
    model = build_model2(input_var=x)
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
    pred_batch = 5
    predict_target = theano.function([index], output, allow_input_downcast=True,
                                     givens={x: small_valid_input[index * pred_batch: (index + 1) * pred_batch]})

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
            if i == (nb_train_batch - 1):
                # Shape = (2600, 3, 64, 64) & Shape = (2600, 3, 32, 32)
                input, target = get_image(data_path, train_input_path, train_target_path, str(i))
                small_train_input.set_value(input)
                small_train_target.set_value(target)
                for j in range(min_train_size):
                    cost = train_small_model(j)
                    n_train_batches += 1
            else:
                # Shape = (10000, 3, 64, 64) & Shape = (10000, 3, 32, 32)
                input, target = get_image(data_path, train_input_path, train_target_path, str(i))
                big_train_input.set_value(input[0: batch * max_size])
                big_train_target.set_value(target[0: batch * max_size])
                for j in range(max_size):
                    cost = train_big_model(j)
                    n_train_batches += 1
                big_train_input.set_value(input[batch * max_size:])
                big_train_target.set_value(target[batch * max_size:])
                for j in range(max_size):
                    cost = train_big_model(j)
                    n_train_batches += 1

        validation_losses = []
        for i in range(nb_valid_batch):
            if i == (nb_valid_batch - 1):
                # Shape = (400, 3, 64, 64) & Shape = (400, 3, 32, 32)
                input, target = get_image(data_path, valid_input_path, valid_target_path, str(i))
                small_valid_input.set_value(input)
                small_valid_target.set_value(target)
                for j in range(min_valid_size):
                    validation_losses.append(small_valid_loss(j))
            else:
                # Shape = (10000, 3, 64, 64) & Shape = (10000, 3, 32, 32)
                input, target = get_image(data_path, valid_input_path, valid_target_path, str(i))
                big_valid_input.set_value(input[0: batch * max_size])
                big_valid_target.set_value(target[0: batch * max_size])
                for j in range(max_size):
                    validation_losses.append(big_valid_loss(j))
                big_valid_input.set_value(input[batch * max_size:])
                big_valid_target.set_value(target[batch * max_size:])
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
