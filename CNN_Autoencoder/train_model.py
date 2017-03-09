import timeit
import theano
import lasagne
import numpy as np
import theano.tensor as T
import lasagne.layers as layers
import lasagne.objectives as objectives

from model import build_model
from utils import save_images

theano.config.floatX = 'float32'


def train_model(learning_rate=0.01, n_epochs=200, batch_size=200, dataset='normalized_mscoco_dataset.npz'):
    '''
                    Function that compute the training of the model
                    '''

    #######################
    # Loading the dataset #
    #######################

    print '... Loading data'

    dataset = np.load(dataset)

    train_input = dataset['train_input']
    train_target = dataset['train_target']
    valid_input = dataset['valid_input']
    valid_target = dataset['valid_target']

    n_train_batches = train_input.size[0] // batch_size
    n_valid_batches = valid_input.size[0] // batch_size

    ###################
    # Build the model #
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
    train_model = theano.function([index], loss, updates=updates,
                                  givens={x: train_input[index * batch_size: (index + 1) * batch_size],
                                          y: train_target[index * batch_size: (index + 1) * batch_size]})

    valid_loss = theano.function([index], loss, givens={x: valid_input[index * batch_size: (index + 1) * batch_size],
                                                        y: valid_target[index * batch_size: (index + 1) * batch_size]})

    idx = 50  # idx = index in this case
    batch = 5
    predict_target = theano.function([index], output,
                                     givens={x: valid_input[index * batch: (index + 1) * batch],
                                             y: valid_target[index * batch: (index + 1) * batch]})

    ###################
    # Train the model #
    ###################

    print('... training')

    # early-stopping parameters
    #patience = 20000  # look as this many minibatches regardless = 48 epochs
    patience = 2  # look as this many minibatches regardless = 48 epochs
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    # go through this many minibatches before checking the
    # network on the validation set. In this case we check
    # at every epoch because n_train_batches = 413
    validation_frequency = min(n_train_batches, patience // 2)

    best_validation_loss = np.inf
    best_iter = 0

    save = 0
    num_images = range(idx * batch, (idx + 1) * batch)

    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index
            cost = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:
                # compute loss on validation set
                validation_losses = [valid_loss(i) for i in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # save the model and a bunch of valid pictures
                    print '... saving model and valid images'

                    np.savez('best_model.npz', *layers.get_all_param_values(model))

                    save += 1
                    input = valid_input[num_images, :, :, :]
                    target = valid_target[num_images, :, :, :]
                    output = predict_target(idx)
                    save_images(input=input, target=target, output=output, nbr_images=len(num_images), iteration=save)

            if patience <= (iter + 1):
                done_looping = True
                break

    end_time = timeit.default_timer()

    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i' %
          (best_validation_loss * 100., best_iter + 1))
    print('The code ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    train_model()
