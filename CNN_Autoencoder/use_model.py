import theano
import numpy as np
import theano.tensor as T
import lasagne.layers as layers

path ='path/'

model = np.load(path + 'best_model.npz')