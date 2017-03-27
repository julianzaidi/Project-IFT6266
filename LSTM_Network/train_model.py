import sys
import timeit
import theano
import lasagne
import numpy as np
import theano.tensor as T
import lasagne.layers as layers
import lasagne.objectives as objectives

from model import build_model
sys.path.insert(0, '/home2/ift6ed67/Project-IFT6266/CNN_Autoencoder')
from utils import get_path
from utils import save_images
from utils import get_image
from utils import get_caption

theano.config.floatX = 'float32'


def train_model(learning_rate=0.0009, n_epochs=1, nb_caption='max'):
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
    train_caption_path = 'train_caption_'
    valid_input_path = 'valid_input_'
    valid_target_path = 'valid_target_'
    valid_caption_path = 'valid_caption_'
    batch_size = 10000
    size_max = 6000
    nb_train_batch = 9
    nb_valid_batch = 5

    ###################
    # Building the model #
    ###################

    # Symbolic variables
    x = T.imatrix('x')
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

    print ('... Training')

    best_validation_loss = np.inf
    best_iter = 0
    epoch = 0

    # Valid images chosen when a better model is found
    batch_verification = 0
    num_images = 2

    start_time = timeit.default_timer()

    while (epoch < n_epochs):
        epoch = epoch + 1
        n_train_batches = 0
        for i in range(nb_train_batch):
            input, target = get_image(data_path, train_input_path, train_target_path, str(i))
            caption = get_caption(data_path, train_caption_path, str(i), str(nb_caption))
            for j in range(len(caption)):
                # Build the target according to the caption
                image = caption[j][:, -1]
                caption_target = target[image - i * batch_size]
                print (caption[j][:, :-1].shape)
                print (caption_target.shape)
                if caption[j][:, :-1].shape[0] > size_max:
                    print ('split')
                    caption_split = caption[j][:size_max, :-1]
                    caption_target_split = caption_target[:size_max]
                    print (caption_split.shape)
                    print (caption_target_split.shape)
                    train_model(caption_split, caption_target_split)
                    caption_split = caption[j][size_max:, :-1]
                    caption_target_split = caption_target[size_max:]
                    print (caption_split.shape)
                    print (caption_target_split.shape)
                    train_model(caption_split, caption_target_split)
                else:
                    train_model(caption[j][:, :-1], caption_target)
                    n_train_batches += 1

        validation_losses = []
        for i in range(nb_valid_batch):
            input, target = get_image(data_path, valid_input_path, valid_target_path, str(i))
            caption = get_caption(data_path, valid_caption_path, str(i), str(nb_caption))
            for j in range(len(caption)):
                # Build the target according to the caption
                image = caption[j][:, -1]
                caption_target = target[image - i * batch_size]
                if caption[j][:, :-1].shape[0] > size_max:
                    caption_split = caption[j][:size_max, :-1]
                    caption_target_split = caption_target[:size_max]
                    validation_losses.append(valid_loss(caption_split, caption_target_split))
                    caption_split = caption[j][:size_max, :-1]
                    caption_target_split = caption_target[:size_max]
                    validation_losses.append(valid_loss(caption_split, caption_target_split))
                else:
                    validation_losses.append(valid_loss(caption[j][:, :-1], caption_target))

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

            np.savez('best_lstm_model.npz', *layers.get_all_param_values(model))

            input, target = get_image(data_path, valid_input_path, valid_target_path, str(batch_verification))
            caption = get_caption(data_path, valid_caption_path, str(batch_verification), str(nb_caption))
            image = caption[0][0:2, -1]
            caption_input = input[image]
            caption_target = target[image]
            output = predict_target(caption[0][0:2, :-1])
            save_images(input=caption_input, target=caption_target, output=output, nbr_images=num_images,
                        iteration=epoch)

    end_time = timeit.default_timer()

    print('Optimization complete.')
    print('Best validation score of %f %% obtained at epoch %i' %
          (best_validation_loss * 100., best_iter))
    print('The code ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    train_model()
