import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image

'''
                Useful functions to use throughout the project
                '''


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
        Image.open('epoch' + str(iteration) + '_valid_set.png').save('epoch' + str(iteration) + '_valid_set.jpg', 'JPEG')

    # Save model assembling
    for i in range(nbr_images):
        plt.subplot(1, nbr_images, (i + 1))
        plt.axis('off')
        plt.imshow(model_assembling[i, :, :, :].transpose(1, 2, 0))

    plt.savefig('epoch' + str(iteration) + '_best_model.png', bbox_inches='tight')
    Image.open('epoch' + str(iteration) + '_best_model.png').save('epoch' + str(iteration) + '_best_model.jpg', 'JPEG')


def get_path():
    path = "/home2/ift6ed67/Data/"
    return path
