import os
import glob
import theano
import numpy as np
import cPickle as pkl
import PIL.Image as Image

theano.config.floatX = 'float32'


def count_color_images(mscoco="inpainting/", train="train2014", valid="val2014"):
    '''
                iterator that count how many color images there are
                in the training and validation set
                '''

    train_data_path = os.path.join(mscoco, train)
    valid_data_path = os.path.join(mscoco, valid)

    train_images = glob.glob(train_data_path + "/*.jpg")
    valid_images = glob.glob(valid_data_path + "/*.jpg")

    num_train = len(train_images)
    num_valid = len(valid_images)
    color_train = 0
    color_valid = 0

    for i, img_path in enumerate(train_images):
        img = Image.open(img_path)
        img_array = np.array(img)

        if len(img_array.shape) == 3:
            color_train += 1
        else:
            pass

    for i, img_path in enumerate(valid_images):
        img = Image.open(img_path)
        img_array = np.array(img)

        if len(img_array.shape) == 3:
            color_valid += 1
        else:
            pass

    # 82782 train examples
    print num_train, '\n'
    # 82611 color train examples
    print color_train, '\n'
    # 40504 valid examples
    print num_valid, '\n'
    # 40438 color valid examples
    print color_valid, '\n'

    return [color_train, color_valid]


def load_dataset(mscoco="inpainting/", train="train2014", valid="val2014",
                 caption_path="dict_key_imgID_value_caps_train_and_valid.pkl", normalize=True):
    '''
            Load the training and validation set
            Here we exclude black and white pictures
            '''

    train_data_path = os.path.join(mscoco, train)
    valid_data_path = os.path.join(mscoco, valid)
    caption_path = os.path.join(mscoco, caption_path)

    with open(caption_path) as fd:
        caption_dict = pkl.load(fd)

    train_images = glob.glob(train_data_path + "/*.jpg")
    valid_images = glob.glob(valid_data_path + "/*.jpg")

    [num_train, num_valid] = count_color_images()

    train_input = np.zeros((num_train, 3, 64, 64), dtype=theano.config.floatX)
    valid_input = np.zeros((num_valid, 3, 64, 64), dtype=theano.config.floatX)

    train_target = np.zeros((num_train, 3, 32, 32), dtype=theano.config.floatX)
    valid_target = np.zeros((num_valid, 3, 32, 32), dtype=theano.config.floatX)

    train_caption = {}
    valid_caption = {}

    j = 0
    k = 0

    for i, img_path in enumerate(train_images):
        img = Image.open(img_path)

        if normalize:
            img_array = np.asarray(img, dtype=theano.config.floatX) / 256.
        else:
            img_array = np.asarray(img, dtype=theano.config.floatX)

        cap_id = os.path.basename(img_path)[:-4]

        ### Get input/target from the images
        center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
        if len(img_array.shape) == 3:
            input = np.copy(img_array)
            input[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = 0
            target = img_array[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :]

            train_input[j, :, :, :] = input.transpose(2, 0, 1)
            train_target[j, :, :, :] = target.transpose(2, 0, 1)
            train_caption[j] = caption_dict[cap_id]

            j += 1
        else:
            # Black and white
            pass

    for i, img_path in enumerate(valid_images):
        img = Image.open(img_path)

        if normalize:
            img_array = np.asarray(img, dtype=theano.config.floatX) / 256.
        else:
            img_array = np.asarray(img, dtype=theano.config.floatX)

        cap_id = os.path.basename(img_path)[:-4]

        ### Get input/target from the images
        center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
        if len(img_array.shape) == 3:
            input = np.copy(img_array)
            input[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = 0
            target = img_array[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :]

            valid_input[k, :, :, :] = input.transpose(2, 0, 1)
            valid_target[k, :, :, :] = target.transpose(2, 0, 1)
            valid_caption[k] = caption_dict[cap_id]

            k += 1
        else:
            # Black and white
            pass

    return [train_input, train_target, train_caption, valid_input, valid_target, valid_caption]


def save_dataset(normalize=True, caption=True):
    '''
            Save the training and validation set + dictionaries
            '''

    [train_input, train_target, train_caption,
     valid_input, valid_target, valid_caption] = load_dataset(normalize=normalize)

    if normalize:
        np.savez('normalized_mscoco_dataset', train_input=train_input, train_target=train_target,
                 valid_input=valid_input, valid_target=valid_target)
    else:
        np.savez('mscoco_dataset', train_input=train_input, train_target=train_target,
                 valid_input=valid_input, valid_target=valid_target)

    if caption:
        with open('train_caption' + '.pkl', 'wb') as f:
            pkl.dump(train_caption, f)

        with open('valid_caption' + '.pkl', 'wb') as f:
            pkl.dump(valid_caption, f)


if __name__ == '__main__':
    save_dataset()
