import h5py
import lasagne
import numpy as np
import theano
import theano.tensor as T

def dnn():
  input_X  = T.tensor4('input',dtype=theano.config.floatX)
  target   = T.matrix('target',dtype=theano.config.floatX)

  l_in = lasagne.layers.InputLayer(shape=(None, 1, 1, 89),
                           input_var=input_X)

  l_hid1 = lasagne.layers.DenseLayer(
                           l_in, num_units=100,
                           nonlinearity=lasagne.nonlinearities.sigmoid,
                           W = lasagne.init.GlorotUniform())

  network = lasagne.layers.DenseLayer(
                           l_hid1, num_units=1,
                           nonlinearity=lasagne.nonlinearities.sigmoid)

  prediction = lasagne.layers.get_output(network)
  loss       = lasagne.objectives.squared_error(prediction, target).mean()

  params  = lasagne.layers.get_all_params(network, trainable=True)
  updates = lasagne.updates.nesterov_momentum(loss, params, \
                                    learning_rate=1e-3, momentum=0.9)

  train = theano.function(
                          inputs  = [ input_X, target ],
                          outputs = [ prediction, loss ],
                          updates = updates,
                          allow_input_downcast = True
                          )



  val_prediction = lasagne.layers.get_output(network, deterministic=True)
  val_loss       = lasagne.objectives.squared_error(val_prediction, target).mean()
  val = theano.function(
                          inputs  = [ input_X, target ],
                          outputs = [ val_prediction, val_loss ],
                          allow_input_downcast = True
                          )

  test_prediction = lasagne.layers.get_output(network, deterministic=True)
  test = theano.function(
                          inputs  = [ input_X ],
                          outputs = test_prediction,
                          allow_input_downcast = True
                          )

  return network, train, val, test
