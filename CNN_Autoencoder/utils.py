import pylab
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

    true_assembling = assemble(input, target)
    model_assembling = assemble(input, output)

    # Save true assembling
    for i in range(nbr_images):
        pylab.subplot(1, nbr_images, (i + 1))
        pylab.axis('off')
        pylab.imshow(true_assembling[i, :, :, :].transpose(1, 2, 0))

    pylab.savefig('save' + str(iteration) + '_valid_set.png', bbox_inches='tight')
    Image.open('save' + str(iteration) + '_valid_set.png').save('save' + str(iteration) + '_valid_set.jpg', 'JPEG')

    # Save model assembling
    for i in range(nbr_images):
        pylab.subplot(1, nbr_images, (i + 1))
        pylab.axis('off')
        pylab.imshow(model_assembling[i, :, :, :].transpose(1, 2, 0))

    pylab.savefig('save' + str(iteration) + '_best_model.png', bbox_inches='tight')
    Image.open('save' + str(iteration) + '_best_model.png').save('save' + str(iteration) + '_best_model.jpg', 'JPEG')


def get_path():
    path = "/Users/Julian/Desktop/Cours/Polytechnique_Montreal/05_Hiver_2017/IFT6266_Deep_Learning/Project/Data"
    return path