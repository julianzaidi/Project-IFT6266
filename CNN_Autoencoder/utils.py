'''
                Useful functions to use throughout the project
                '''

import theano
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import _pickle as pkl
# import PIL.Image as Image

theano.config.floatX = 'float32'


def save_obj(obj, name, extension='.pkl'):
    with open(name + extension, 'wb') as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)


def load_obj(name, extension='.pkl'):
    with open(name + extension, 'rb') as f:
        return pkl.load(f)


def get_path():
    data_path = "/home2/ift6ed67/Data/"

    return data_path


def get_image(path, input_path, target_path, iteration):
    input = np.load(path + input_path + iteration + '.npy')
    target = np.load(path + target_path + iteration + '.npy')

    return input, target


def get_caption(path, caption_path, iteration, nb_caption):
    caption = np.load(path + caption_path + iteration + '_caption=' + nb_caption + '.npy', encoding='latin1')

    return caption


def shared_GPU_data(shape, dtype=theano.config.floatX, borrow=True):
    '''
            Create theano shared variables to be load on the GPU
    '''
    return theano.shared(np.zeros(shape=shape, dtype=dtype), borrow=borrow)


def random_sample(size=None, dtype=theano.config.floatX):

    sample = np.random.normal(size=size)
    sample = sample.astype(dtype)

    return sample


def rolling_average(list, max_iter=100):
    y = []
    for i in range(len(list)):
        if i < max_iter:
            y.append(np.mean(list[:i + 1]))
        else:
            y.append(np.mean(list[i - max_iter:i + 1]))

    return y


def assemble(input, target):
    '''
                Assemble the input with the target

                :type input : numpy array of dimension 4
                :param input : shape = (n_batch, 3, 64, 64)

                :type target : numpy array of dimension 4
                :param target : shape = (n_batch, 3, 32, 32)
                '''

    assembling = np.copy(input)
    center = (int(np.floor(input.shape[2] / 2.)), int(np.floor(input.shape[3] / 2.)))
    assembling[:, :, center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16] = target

    return assembling  # Shape = (n_batch, 3, 64, 64)


def save_images(input, target, output, nbr_images, iteration):
    '''
                    Save a bunch of images to see the performance of the
                    model on the validation set at some periods of training
                    '''

    model_assembling = assemble(input, output)

    # Save true assembling
    if iteration == 1:
        true_assembling = assemble(input, target)
        for i in range(nbr_images):
            plt.subplot(1, nbr_images, (i + 1))
            plt.axis('off')
            plt.imshow(true_assembling[i, :, :, :].transpose(1, 2, 0))

        plt.savefig('epoch' + str(iteration) + '_valid_set.png', bbox_inches='tight')
        # Image.open('epoch' + str(iteration) + '_valid_set.png').save('epoch' + str(iteration) + '_valid_set.jpg',
        #                                                            'JPEG')

    # Save model assembling
    for i in range(nbr_images):
        plt.subplot(1, nbr_images, (i + 1))
        plt.axis('off')
        plt.imshow(model_assembling[i, :, :, :].transpose(1, 2, 0))

    plt.savefig('epoch' + str(iteration) + '_best_model.png', bbox_inches='tight')
    # Image.open('epoch' + str(iteration) + '_best_model.png').save('epoch' + str(iteration) + '_best_model.jpg', 'JPEG')
