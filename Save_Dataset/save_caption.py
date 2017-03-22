import os
import cPickle as pkl


def get_path():
    path = "/Users/Julian/Desktop/Cours/Polytechnique_Montreal/05_Hiver_2017/IFT6266_Deep_Learning/Project/Data/" \
           "Useful_Data/"
    return path


def load_caption(train="train_caption.pkl", valid="valid_caption.pkl", words="worddict.pkl"):
    '''
            Load captions for the training and validation set
            Load the vocabulary dictionary
            '''
    path = get_path()

    caption_path = train
    train_path = os.path.join(path, caption_path)

    with open(train_path) as fd:
        train_caption = pkl.load(fd)  # Keys from 0 to 82610

    caption_path = valid
    valid_path = os.path.join(path, caption_path)

    with open(valid_path) as fd:
        valid_caption = pkl.load(fd)  # Keys from 0 to 40437

    vocab = words
    vocabulary_path = os.path.join(path, vocab)

    with open(vocabulary_path) as fd:
        vocabulary = pkl.load(fd)  # Dictionary of the different words composing the caption

    return train_caption, valid_caption, vocabulary


def ordered_list(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def reshape(sentence):
    sentence = sentence.lower().replace('.', '').replace(',', '').replace(';', '').replace(':', '').replace('"', ''). \
        replace('...', '').replace('/', '').replace('\'', '')
    return sentence


def save_obj(obj, name, extension='.pkl'):
    with open(name + extension, 'wb') as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)


def load_obj(name, extension='.pkl'):
    with open(name + extension, 'rb') as f:
        return pkl.load(f)


def save_caption():
    '''
            Reshape the caption to lower case without punctuation
            Reshape the vocabulary dictionary and set it in a new vocabulary list
            '''
    train_caption, valid_caption, vocab = load_caption()

    vocabulary = []
    nb_train_images = len(train_caption)
    nb_valid_images = len(valid_caption)

    for i in range(nb_train_images):
        nb_caption = len(train_caption[i])
        for j in range(nb_caption):
            train_caption[i][j] = reshape(train_caption[i][j])

    for i in range(nb_valid_images):
        nb_caption = len(valid_caption[i])
        for j in range(nb_caption):
            valid_caption[i][j] = reshape(valid_caption[i][j])

    for i in range(len(vocab)):
        word = reshape(vocab.items()[i][0])
        if len(word) != 0:
            vocabulary.append(word)

    vocabulary = ordered_list(vocabulary)

    save_obj(train_caption, 'train_caption')
    save_obj(valid_caption, 'valid_caption')
    save_obj(vocabulary, 'vocabulary', extension='.txt')


if __name__=='__main__':
    save_caption()

    #train_caption = load_obj('train_caption')
    #valid_caption = load_obj('valid_caption')
    #vocabulary = load_obj('vocabulary', extension='.txt')

    #print len(train_caption)
    #print train_caption[0]
    #print len(valid_caption)
    #print valid_caption[0]
    #print len(vocabulary)
    #print vocabulary[:30]