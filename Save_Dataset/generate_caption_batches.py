import sys
import numpy as np

sys.path.insert(0, '/Users/Julian/Desktop/Cours/Polytechnique_Montreal/05_Hiver_2017/IFT6266_Deep_Learning/Project/'
                   'Project_IFT6266_GitHub/cnn_autoencoder')
from utils import get_path
from utils import load_obj


def get_vocab_length():
    path = get_path()
    vocabulary = load_obj(path + 'my_vocabulary', extension='.txt')

    return len(vocabulary)


def get_length(caption_dict, nb_caption, idx, num_images=10000):
    '''
                Determine the length of each caption
                :param caption_dict: has to be train_caption or valid_caption
                :param nb_caption: the number of caption used per image
                '''

    length = []
    nbr_caption = nb_caption
    assert len(caption_dict) is 82611 or 40438

    if len(caption_dict) == 82611:
        if idx == 8:
            images = range(idx * num_images, idx * num_images + 2600)
            for i in images:
                if nb_caption == 'max':
                    nbr_caption = len(caption_dict[i])
                for j in range(nbr_caption):
                    size = len(caption_dict[i][j].split())
                    length.append(size)
        else:
            images = range(idx * num_images, (idx + 1) * num_images)
            for i in images:
                if nb_caption == 'max':
                    nbr_caption = len(caption_dict[i])
                for j in range(nbr_caption):
                    size = len(caption_dict[i][j].split())
                    length.append(size)

    if len(caption_dict) == 40438:
        if idx == 4:
            images = range(idx * num_images, idx * num_images + 400)
            for i in images:
                if nb_caption == 'max':
                    nbr_caption = len(caption_dict[i])
                for j in range(nbr_caption):
                    size = len(caption_dict[i][j].split())
                    length.append(size)
        else:
            images = range(idx * num_images, (idx + 1) * num_images)
            for i in images:
                if nb_caption == 'max':
                    nbr_caption = len(caption_dict[i])
                for j in range(nbr_caption):
                    size = len(caption_dict[i][j].split())
                    length.append(size)

    return length, images


def same_size(caption_dict, nb_caption, idx):
    '''
            Count the number of captions of the same size
            :param nb_caption: the number of caption used per image
            '''

    length, images = get_length(caption_dict, nb_caption, idx)
    min_length = min(length)
    max_length = max(length)

    count = {}
    for i in range(min_length, max_length + 1):
        count[i] = 0
        for j in range(len(length)):
            if length[j] == i:
                count[i] += 1

    return count, images


def generate_batches(nb_caption):
    '''
            Generate batches to store captions of the same size together
            :param nb_caption: the number of caption used per image
    '''

    path = get_path()

    train_caption = load_obj(path + 'train_caption')
    valid_caption = load_obj(path + 'valid_caption')
    vocabulary = load_obj(path + 'my_vocabulary', extension='.txt')

    nbr_caption = nb_caption

    for idx in range(9):
        train_mini_batches = []
        train_count, images = same_size(train_caption, nb_caption, idx)
        train_sizes = train_count.keys()
        for i in train_sizes:
            if train_count[i] != 0:
                iterator = 0
                batch = np.zeros((train_count[i], i + 1))
                for j in images:
                    if nb_caption == 'max':
                        nbr_caption = len(train_caption[j])
                    for k in range(nbr_caption):
                        size = len(train_caption[j][k].split())
                        if size == i:
                            index = []
                            for l in range(size):
                                index.append(vocabulary.index(train_caption[j][k].split()[l]))
                            index.append(j)  # To keep track of the association between the caption and the image
                            batch[iterator, :] = index
                            iterator += 1

                batch = batch.astype('int32')
                train_mini_batches.append(batch)
        np.save('train_caption_' + str(idx) +'_caption=' + str(nb_caption), train_mini_batches)

    for idx in range(5):
        valid_mini_batches = []
        valid_count, images = same_size(valid_caption, nb_caption, idx)
        valid_sizes = valid_count.keys()
        for i in valid_sizes:
            if valid_count[i] != 0:
                iterator = 0
                batch = np.zeros((valid_count[i], i + 1))
                for j in images:
                    if nb_caption == 'max':
                        nbr_caption = len(valid_caption[j])
                    for k in range(nbr_caption):
                        size = len(valid_caption[j][k].split())
                        if size == i:
                            index = []
                            for l in range(size):
                                index.append(vocabulary.index(valid_caption[j][k].split()[l]))
                            index.append(j)  # To keep track of the association between the caption and the image
                            batch[iterator, :] = index
                            iterator += 1

                batch = batch.astype('int32')
                valid_mini_batches.append(batch)
        np.save('valid_caption_' + str(idx) +'_caption=' + str(nb_caption), valid_mini_batches)


if __name__ == '__main__':
    nb_caption = 1
    generate_batches(nb_caption)
    nb_caption = 'max'
    generate_batches(nb_caption)