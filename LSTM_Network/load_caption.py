import os
import glob
import theano
import numpy as np
import cPickle as pkl
import PIL.Image as Image
from utils import get_path


def load_caption(train="train_caption.pkl", valid="valid_caption.pkl"):
    path = get_path()

    caption_path = train
    train_path = os.path.join(path, caption_path)

    with open(train_path) as fd:
        train_caption = pkl.load(fd)  # Keys from 0 to 82610

    caption_path = valid
    valid_path = os.path.join(path, caption_path)

    with open(valid_path) as fd:
        valid_caption = pkl.load(fd)  # Keys from 0 to 40437

    return train_caption, valid_caption


def ordered_list(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def reshape(sentence):
    sentence = sentence.lower().replace('.', '').replace(',', '').replace(';', '').replace(':', '').replace('"', ''). \
        replace('...', '').replace('/', '').replace('\'', '')
    return sentence


def get_caption():
    train_caption, valid_caption = load_caption()

    vocabulary = []
    # train_vocabulary = []
    # valid_vocabulary = []
    nb_train_images = len(train_caption)
    nb_valid_images = len(valid_caption)

    for i in range(nb_train_images):
        nb_caption = len(train_caption[i])
        for j in range(nb_caption):
            train_caption[i][j] = reshape(train_caption[i][j])
            split = train_caption[i][j].split()
            nb_words = len(split)
            for k in range(nb_words):
                # train_vocabulary.append(split[k])
                vocabulary.append(split[k])

    # train_vocabulary = list(set(train_vocabulary))
    # train_vocabulary = ordered_list(train_vocabulary)

    for i in range(nb_valid_images):
        nb_caption = len(valid_caption[i])
        for j in range(nb_caption):
            valid_caption[i][j] = reshape(valid_caption[i][j])
            split = valid_caption[i][j].split()
            nb_words = len(split)
            for k in range(nb_words):
                # valid_vocabulary.append(split[k])
                vocabulary.append(split[k])

    # valid_vocabulary = list(set(valid_vocabulary))
    # valid_vocabulary = ordered_list(valid_vocabulary)
    vocabulary = ordered_list(vocabulary)

    return train_caption, valid_caption, vocabulary


# train_caption = load_caption()
# print len(train_caption)  # 82611 = nbr images
# print train_caption[0]  # Captions for the first image
# print len(train_caption[0])  # Number of captions for the first image
# print train_caption[0][0]  # First caption of the first image
# print train_caption[0][0].split()  # List of words for the first caption of the first image
# print len(train_caption[0][0].split())  # Number of words for the first caption of the image
# print train_caption[0][0].split()[0] # First word of the first caption of the first image

#train_caption, valid_caption, vocabulary = get_caption()
#print train_caption[82400]
# print train_caption[0]
# print train_caption[6]
# print valid_caption[0]
# print len(train_vocabulary)
# print len(valid_vocabulary)
# print len(vocabulary)
# for i in range(10):
#    print vocabulary[i]
# for i in range(10):
# for i in range(10):
#    print train_vocabulary[i]
# for i in range(10):
#    print valid_vocabulary[i]
# print (train_vocabulary.index('food'))

# train_caption, valid_caption = load_caption()
# print train_caption[0][0]
# train_caption[0][0] = reshape(train_caption[0][0])
# print train_caption[0][0]

# sentence1 = 'BonJour JULIAN , ZA " IDI'
# print sentence1
# print len(sentence1.split())
# sentence1 = reshape(sentence1)
# print sentence1
# print len(sentence1.split())


def get_length(caption_dict, idx, nb_caption, batch_size):
    '''
            Find the shortest sentence in the batch
            :param nb_caption: the number of caption used per image
            '''

    min_length = np.inf
    nbr_caption = nb_caption

    for i in range(idx * batch_size, (idx + 1) * batch_size):
        if nb_caption == 'max':
            nbr_caption = len(caption_dict[i])
        for j in range(nbr_caption):
            length = len(caption_dict[i][j].split())
            if length < min_length:
                min_length = length

    return min_length


def get_number(caption_dict, idx, nb_caption, batch_size):
    '''
                Find the number of total caption in the batch
                :param nb_caption: the number of caption used per image
                '''

    tot_caption = 0

    if nb_caption == 'max':
        for i in range(idx * batch_size, (idx + 1) * batch_size):
            tot_caption += len(caption_dict[i])
    else:
        tot_caption = batch_size * nb_caption

    return tot_caption


def get_batch(caption_dict, vocabulary_list, idx, nb_caption, batch_size):
    '''
                    Compute the mini-batch
                    :param nb_caption: the number of caption used per image
                    '''

    length = get_length(caption_dict, idx, nb_caption, batch_size)
    tot_caption = get_number(caption_dict, idx, nb_caption, batch_size)
    batch = np.zeros((tot_caption, length))
    nbr_caption = nb_caption
    iterator = 0

    for i in range(idx * batch_size, (idx + 1) * batch_size):
        if nb_caption == 'max':
            nbr_caption = len(caption_dict[i])
        for j in range(nbr_caption):
            caption = [caption_dict[i][j].split()[k] for k in range(length)]
            index = []
            for l in range(length):
                index.append(vocabulary_list.index(caption[l]))
            batch[iterator, :] = index
            iterator += 1

    batch = batch.astype('int32')

    return batch


def get_batches(batch_size=200, nb_caption='max'):
    '''
            Create matrix mini-batches for the input of the network
            :param batch_size: Number of images
            :param nb_caption: the number of caption used per image
            '''

    train_caption, valid_caption, vocabulary = get_caption()

    nb_train_images = len(train_caption)
    nb_valid_images = len(valid_caption)
    train_mini_batches = []
    valid_mini_batches = []

    n_train_batches = nb_train_images // batch_size  # 413 mini-batch of 200 images
    n_valid_batches = nb_valid_images // batch_size  # 202 mini-batch of 200 images

    for i in range(n_train_batches):
        mini_batch = get_batch(train_caption, vocabulary, i, nb_caption, batch_size)
        train_mini_batches.append(mini_batch)

    for j in range(n_valid_batches):
        mini_batch = get_batch(valid_caption, vocabulary, j, nb_caption, batch_size)
        valid_mini_batches.append(mini_batch)

    return train_mini_batches, valid_mini_batches


train_caption, valid_caption, vocabulary = get_caption()

print train_caption[0]
print len(vocabulary)
print vocabulary[:50]


train_mini_batches, valid_mini_batches = get_batches()

print len(train_mini_batches)
print len(valid_mini_batches)
print train_mini_batches[0].shape
print train_mini_batches[-1].shape
print valid_mini_batches[0].shape
print valid_mini_batches[-1].shape

print train_mini_batches[0][0,:]
print train_mini_batches[0][1,:]
print valid_mini_batches[-1][-1,:]
