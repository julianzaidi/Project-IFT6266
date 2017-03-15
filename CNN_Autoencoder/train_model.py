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


def get_train_data(path, input_path, target_path, iteration):
    train_input = np.load(path + input_path + iteration + '.npy')
    train_target = np.load(path + target_path + iteration + '.npy')

    return train_input, train_target


def get_valid_data(path, input_path, target_path, iteration):
    valid_input = np.load(path + input_path + iteration + '.npy')
    valid_target = np.load(path + target_path + iteration + '.npy')

    return valid_input, valid_target


def train_model(learning_rate=0.01, n_epochs=3, batch_size=200):
    '''
                    Function that compute the training of the model
                    '''

    ###########################################
    # Writting what we know about the dataset #
    ###########################################

    image_path = get_path()
    train_input_path = 'train_input_'
    train_target_path = 'train_target_'
    valid_input_path = 'valid_input_'
    valid_target_path = 'valid_target_'
    train_save = 59  # Total number of training files
    valid_save = 26  # Total number of validation files

    ###################
    # Building the model #
    ###################

    # Symbolic variables
    x = T.tensor4('x', dtype=theano.config.floatX)
    y = T.tensor4('y', dtype=theano.config.floatX)

    # Creation of the model
    model = build_model(input_var=x)
    output = layers.get_output(model, deterministic=True)
    params = layers.get_all_params(model, trainable=True)
    loss = T.mean(objectives.squared_error(output, y))
    updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)

    # Creation of theano functions
    train_model = theano.function([x, y], loss, updates=updates, allow_input_downcast=True)
    valid_loss = theano.function([x, y], loss, allow_input_downcast=True)
    predict_target = theano.function([x], output, allow_input_downcast=True)

    ###################
    # Train the model #
    ###################

    print('... Training')

    best_validation_loss = np.inf
    best_iter = 0
    epoch = 0

    # Valid images chosen when a better model is found
    idx = 50
    batch = 5
    batch_verification = 0
    num_images = range(idx * batch, (idx + 1) * batch)

    input, target = get_valid_data(image_path, valid_input_path, valid_target_path, str(batch_verification))
    input = input[idx * batch: (idx + 1) * batch]
    target = target[idx * batch: (idx + 1) * batch]

    start_time = timeit.default_timer()

    while (epoch < n_epochs):
        epoch = epoch + 1
        n_train_batches = 0
        for i in range(train_save):
            train_input, train_target = get_train_data(image_path, train_input_path, train_target_path, str(i))
            for j in range(train_input.shape[0] // batch_size):
                train_model(train_input[j * batch_size: (j + 1) * batch_size],
                            train_target[j * batch_size: (j + 1) * batch_size])
                n_train_batches += 1

        validation_losses = []
        for i in range(valid_save):
            valid_input, valid_target = get_valid_data(image_path, valid_input_path, valid_target_path, str(i))
            validation_losses.append(valid_loss(valid_input, valid_target))

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
            output = predict_target(input)
            save_images(input=input, target=target, output=output, nbr_images=len(num_images), iteration=epoch)

    end_time = timeit.default_timer()

    print('Optimization complete.')
    print('Best validation score of %f %% obtained at epoch %i' %
          (best_validation_loss * 100., best_iter))
    print('The code ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    train_model()
