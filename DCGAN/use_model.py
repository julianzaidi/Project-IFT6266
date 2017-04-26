import theano
import numpy as np
import theano.tensor as T
import lasagne.layers as layers
import matplotlib.pyplot as plt

from model import build_generator

theano.config.floatX = 'float32'


def random_sample(size=None, dtype=theano.config.floatX):
    sample = np.random.normal(size=size)
    sample = sample.astype(dtype)

    return sample


######################
# Building the model #
######################

# Symbolic variables
x_gen = T.matrix('x_gen', dtype=theano.config.floatX)

# Creating the model
generator = build_generator(input_var=x_gen)
with np.load('generator_epoch36.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
layers.set_all_param_values(generator, param_values)
output = layers.get_output(generator, deterministic=True)

# Creating theano function
predict_target = theano.function([x_gen], output, allow_input_downcast=True, )


######################
# Predict the target #
######################

sample = random_sample(size=(5, 100))
image = predict_target(sample)

for k in range(5):
    plt.subplot(1, 5, (k + 1))
    plt.axis('off')
    plt.imshow(image[k, :, :, :].transpose(1, 2, 0))

plt.savefig('test.png', bbox_inches='tight')