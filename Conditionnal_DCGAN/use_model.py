import sys
import theano
import numpy as np
import theano.tensor as T
import lasagne.layers as layers
import matplotlib.pyplot as plt

from model import build_context_encoder
sys.path.insert(0, '/home2/ift6ed67/Project-IFT6266/CNN_Autoencoder')
from utils import get_path
from utils import get_image
from utils import assemble

theano.config.floatX = 'float32'

#######################
# Loading the dataset #
#######################

data_path = get_path()
valid_input_path = 'valid_input_'
valid_target_path = 'valid_target_'
nb_valid_batch = 4
batch_size = 5


######################
# Building the model #
######################

# Symbolic variables
x = T.tensor4('x', dtype=theano.config.floatX)

# Creating the model
model = build_context_encoder(input_var=x)
with np.load(data_path + 'context_encoder_epoch40.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
layers.set_all_param_values(model, param_values)
output = layers.get_output(model, deterministic=True)

# Creating theano function
predict_target = theano.function([x], output, allow_input_downcast=True,)


######################
# Predict the target #
######################

for i in range(nb_valid_batch):
    input, target = get_image(data_path, valid_input_path, valid_target_path, str(i))
    for idx in range(10):
        true_input = input[idx * 1000: idx * 1000 + batch_size]
        true_target = target[idx * 1000: idx * 1000 + batch_size]
        output = predict_target(true_input)

        true_assembling = assemble(true_input, true_target)
        model_assembling = assemble(true_input, output)

        for j in range(batch_size):
            plt.subplot(1, batch_size, (j + 1))
            plt.axis('off')
            plt.imshow(true_assembling[j, :, :, :].transpose(1, 2, 0))

        plt.savefig(str(i) + 'true_output_' + str(idx * 1000) + '_to_' + str(idx * 1000 + batch_size - 1) + '.png',
                    bbox_inches='tight')

        for j in range(batch_size):
            plt.subplot(1, batch_size, (j + 1))
            plt.axis('off')
            plt.imshow(model_assembling[j, :, :, :].transpose(1, 2, 0))

        plt.savefig(str(i) + 'model_output_' + str(idx * 1000) + '_to_' + str(idx * 1000 + batch_size - 1) + '.png',
                    bbox_inches='tight')



