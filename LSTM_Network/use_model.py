import sys
import theano
import numpy as np
import theano.tensor as T
import lasagne.layers as layers
import matplotlib.pyplot as plt

from model import build_model
sys.path.insert(0, '/home2/ift6ed67/Project-IFT6266/CNN_Autoencoder')
from utils import get_path
from utils import load_obj
from utils import get_image
from utils import assemble
from utils import get_caption


#######################
# Loading the dataset #
#######################

data_path = get_path()
valid_input_path = 'valid_input_'
valid_target_path = 'valid_target_'
valid_caption_path = 'valid_caption_'
vocabulary_path = 'my_vocabulary'
batch_size = 5
batch = 0


######################
# Building the model #
######################

# Symbolic variables
x = T.imatrix('x')

# Creating the model
model = build_model(input_var=x)
with np.load(data_path + 'best_lstm_model.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
layers.set_all_param_values(model, param_values)
output = layers.get_output(model, deterministic=True)

# Creating theano function
predict_target = theano.function([x], output, allow_input_downcast=True,)


######################
# Predict the target #
######################

valid_input, valid_target = get_image(data_path, valid_input_path, valid_target_path, str(batch))
valid_caption = get_caption(data_path, valid_caption_path, str(batch), nb_caption=str(1))
vocabulary = load_obj(data_path + vocabulary_path, extension='.txt')

images = valid_caption[3][:batch_size, -1]
caption_batch = valid_caption[3][:batch_size, :-1]
input_batch = valid_input[images]
target_batch = valid_target[images]

for i in range(batch_size):
    sentence = [vocabulary[caption_batch[i, j]] for j in range(caption_batch.shape[1])]
    print sentence

output = predict_target(caption_batch)
true_assembling = assemble(input_batch, target_batch)
model_assembling = assemble(input_batch, output)

for j in range(batch_size):
    plt.subplot(1, batch_size, (j + 1))
    plt.axis('off')
    plt.imshow(true_assembling[j, :, :, :].transpose(1, 2, 0))

plt.savefig('true_output_caption=max.png', bbox_inches='tight')

for j in range(batch_size):
    plt.subplot(1, batch_size, (j + 1))
    plt.axis('off')
    plt.imshow(model_assembling[j, :, :, :].transpose(1, 2, 0))

plt.savefig('model_output_caption=max.png', bbox_inches='tight')


