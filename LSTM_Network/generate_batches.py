import sys
import numpy as np
sys.path.insert(0, '/Users/Julian/Desktop/Cours/Polytechnique_Montreal/05_Hiver_2017/IFT6266_Deep_Learning/Project/'
                   'Project_IFT6266_GitHub/cnn_autoencoder')
from utils import get_path, load_obj


def get_length(caption_dict, nb_caption):
    '''
            Determine the length of each caption
            :param caption_dict: has to be train_caption or valid_caption
            :param nb_caption: the number of caption used per image
            '''

    length = []
    nbr_caption = nb_caption
    assert len(caption_dict) is 82611 or 40438

    if len(caption_dict) == 82611:
        for i in range(len(caption_dict) - 11):
            if nb_caption == 'max':
                nbr_caption = len(caption_dict[i])
            for j in range(nbr_caption):
                size = len(caption_dict[i][j].split())
                length.append(size)

    elif len(caption_dict) == 40438:
        for i in range(len(caption_dict) - 38):
            if nb_caption == 'max':
                nbr_caption = len(caption_dict[i])
            for j in range(nbr_caption):
                size = len(caption_dict[i][j].split())
                length.append(size)

    return length


def same_size(caption_dict, nb_caption):
    '''
            Count the number of captions of the same size
            :param nb_caption: the number of caption used per image
            '''

    length = get_length(caption_dict, nb_caption)
    min_length = min(length)
    max_length = max(length)

    count = {}
    for i in range(min_length, max_length + 1):
        count[i] = 0
        for j in range(len(length)):
            if length[j] == i:
                count[i] += 1

    return count


def generate_batches(nb_caption):
    '''
            Generate batches to store captions of the same size together
            :param nb_caption: the number of caption used per image
    '''

    path = get_path()

    train_caption = load_obj(path + 'train_caption')
    valid_caption = load_obj(path + 'valid_caption')
    vocabulary = load_obj(path + 'my_vocabulary', extension='.txt')

    train_mini_batches = []
    train_count = same_size(train_caption, nb_caption)
    train_sizes = train_count.keys()

    valid_mini_batches = []
    valid_count = same_size(valid_caption, nb_caption)
    valid_sizes = valid_count.keys()

    nbr_caption = nb_caption

    for i in train_sizes:
        if train_count[i] != 0:
            iterator = 0
            batch = np.zeros((train_count[i], i + 1))
            for j in range(len(train_caption)):
                if j >= 82600:
                    pass
                else:
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

            idx = np.argsort(batch[:, -1])
            batch = batch[idx]
            batch = batch.astype('int32')
            train_mini_batches.append(batch)

    for i in valid_sizes:
        if valid_count[i] != 0:
            iterator = 0
            batch = np.zeros((valid_count[i], i + 1))
            for j in range(len(valid_caption)):
                if j >= 40400:
                    pass
                else:
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

            idx = np.argsort(batch[:, -1])
            batch = batch[idx]
            batch = batch.astype('int32')
            valid_mini_batches.append(batch)

    return train_mini_batches, valid_mini_batches