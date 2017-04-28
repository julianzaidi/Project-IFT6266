import sys
import random
import timeit
import theano
import lasagne
import numpy as np
import theano.tensor as T
import lasagne.layers as layers
import lasagne.objectives as objectives

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as AA
from mpl_toolkits.axes_grid1 import host_subplot

from model import build_context_encoder
from model import build_discriminator

sys.path.insert(0, '/home2/ift6ed67/Project-IFT6266/CNN_Autoencoder')
from utils import get_path
from utils import assemble
from utils import get_image
from utils import get_caption

theano.config.floatX = 'float32'


def rolling_average(list, max_iter=100):
    y = []
    for i in range(len(list)):
        if i < max_iter:
            y.append(np.mean(list[:i + 1]))
        else:
            y.append(np.mean(list[i - max_iter:i + 1]))

    return y


def train_model(learning_rate_dis=0.0004, learning_rate_model=0.0004, n_epochs=1, batch_size=20, nb_caption='max'):
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
    nb_train_batch = 2
    #nb_train_batch = 8


    ######################
    # Building the model #
    ######################

    # Symbolic variables
    # Shape = (_, 3, 64, 64)
    x = T.tensor4('x', dtype=theano.config.floatX)
    # Shape = (_, 3, 32, 32)
    y = T.tensor4('y', dtype=theano.config.floatX)
    # Shape = (_, 3, 32, 32)
    z = T.tensor4('x', dtype=theano.config.floatX)
    # Shape = (_, seq_length)
    w = T.imatrix('captions')

    # Creation of the model
    model = build_context_encoder(input_var1=x, input_var2=w)
    discriminator = build_discriminator(input_var=None)

    fake_image = layers.get_output(model)
    fake_image_det = layers.get_output(model, deterministic=True)
    prob_real = layers.get_output(discriminator, inputs=y)
    prob_fake = layers.get_output(discriminator, inputs=fake_image)

    params_model = layers.get_all_params(model, trainable=True)
    params_dis = layers.get_all_params(discriminator, trainable=True)

    loss_real = -T.mean(T.log(prob_real))
    loss_fake = -T.mean(T.log(1 - prob_fake))
    loss_dis = 0.005 * (loss_real + loss_fake)

    loss_gen = -T.mean(T.log(prob_fake))
    recons_error = T.mean(objectives.squared_error(fake_image, z))
    loss_model = 0.005 * loss_gen + 0.995 * recons_error

    updates_dis = lasagne.updates.adam(loss_dis, params_dis, learning_rate=learning_rate_dis, beta1=0.5)
    updates_model = lasagne.updates.adam(loss_model, params_model, learning_rate=learning_rate_model, beta1=0.5)

    # Creation of theano functions
    train_dis = theano.function([x, y, w], loss_dis, updates=updates_dis, allow_input_downcast=True)

    train_model = theano.function([x, z, w], loss_model, updates=updates_model, allow_input_downcast=True)

    predict_image = theano.function([x, w], fake_image_det, allow_input_downcast=True)

    ###################
    # Train the model #
    ###################

    print('... Training')

    epoch = 0
    nb_train_dis = 25
    nb_train_gen = 10
    nb_batch = 10000 // batch_size
    nb_block = nb_batch // nb_train_dis
    loss_dis = []
    loss_model = []

    idx = [0, 1, 2, 4, 5]

    start_time = timeit.default_timer()

    while (epoch < n_epochs):
        epoch = epoch + 1
        for i in range(nb_train_batch):
            print (i)
            # Shape = (10000, 3, 64, 64) & Shape = (10000, 3, 32, 32)
            contour, center = get_image(data_path, train_input_path, train_target_path, str(i))
            # List of captions of different sequence length
            caption = get_caption(data_path, train_caption_path, str(i), str(nb_caption))
            # List of size nb_train_dis
            list = [k % len(caption) for k in range(nb_train_dis)]
            for j in range(nb_block):
                print (j)
                for index in range(nb_train_dis * j, nb_train_dis * (j + 1)):
                    print (index)
                    train_caption = caption[list[index % nb_train_dis]]
                    if train_caption.shape[0] >= batch_size:
                        random_idx = random.sample(range(0, train_caption.shape[0]), batch_size)
                    else:
                        random_idx = random.sample(range(0, train_caption.shape[0]), train_caption.shape[0])
                    input = contour[train_caption[random_idx, -1] - i * 10000]
                    target = center[train_caption[random_idx, -1] - i * 10000]
                    train_caption = train_caption[random_idx, :-1]
                    if index == 13:
                        print (input.shape)
                        print (target.shape)
                        print (train_caption.shape)
                    loss = train_dis(input, target, train_caption)
                    loss_dis.append(loss)
                for index in range(nb_train_gen * j, nb_train_gen * (j + 1)):
                    print (index)
                    rand_nb = random.randint(0, len(list) - 1)
                    train_caption = caption[rand_nb]
                    if train_caption.shape[0] >= batch_size:
                        random_idx = random.sample(range(0, train_caption.shape[0]), batch_size)
                    else:
                        random_idx = random.sample(range(0, train_caption.shape[0]), train_caption.shape[0])
                    input = contour[train_caption[random_idx, -1] - i * 10000]
                    target = center[train_caption[random_idx, -1] - i * 10000]
                    train_caption = train_caption[random_idx, :-1]
                    if index == 6:
                        print (input.shape)
                        print (target.shape)
                        print (train_caption.shape)
                    loss = train_model(input, target, train_caption)
                    loss_model.append(loss)

        if epoch % 1 == 0:
            # save the model and a bunch of generated pictures
            print ('... saving model and generated images')

            np.savez('discriminator_epoch' + str(epoch) + '.npz', *layers.get_all_param_values(discriminator))
            np.savez('context_encoder_epoch' + str(epoch) + '.npz', *layers.get_all_param_values(model))
            np.save('loss_dis', loss_dis)
            np.save('loss_gen', loss_model)

            contour, center = get_image(data_path, valid_input_path, valid_target_path, str(0))
            caption = get_caption(data_path, valid_caption_path, str(0), str(nb_caption))
            valid_caption = caption[4][idx]
            input = contour[valid_caption[:, -1]]
            print (valid_caption.shape)
            print (input.shape)

            generated_centers = predict_image(input, valid_caption[:, :-1])
            generated_images = assemble(input, generated_centers)

            for k in range(len(idx)):
                plt.subplot(1, len(idx), (k + 1))
                plt.axis('off')
                plt.imshow(generated_images[k, :, :, :].transpose(1, 2, 0))

            plt.savefig('generated_images_epoch' + str(epoch) + '.png', bbox_inches='tight')

    end_time = timeit.default_timer()

    # Plot the learning curve
    ax1 = host_subplot(111, axes_class=AA.Axes)
    plt.subplots_adjust(right=0.75)
    ax2 = ax1.twiny()

    x1 = range(1, len(loss_dis) + 1)
    ax1.set_xlim([x1[0], x1[-1]])
    x2 = range(1, len(loss_model) + 1)
    ax2.set_xlim([x2[0], x2[-1]])

    ax1.set_xlabel('training iteration (Discriminator)', color='g')
    ax2.set_xlabel('training iteration (Context encoder)', color='b')
    ax1.set_ylabel('Loss')

    ax1.plot(x1, rolling_average(loss_dis), 'g', label='Discriminator loss')
    ax2.plot(x2, rolling_average(loss_model), 'b', label='Context encoder Loss')

    ax1.grid(True)
    ax1.legend()

    plt.savefig('Learning_curve')

    print('Optimization complete.')
    print('The code ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    train_model()
