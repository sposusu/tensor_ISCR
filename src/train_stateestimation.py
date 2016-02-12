import pdb


import h5py
import lasagne
import numpy as np
import theano
import theano.tensor as T

from IR.stateestimation import dnn


#################
#   Data Path   #
#################
se_prefix = '../Data/stateestimation/'
fname = se_prefix + 'features_89.h5'
weights_file = se_prefix + 'dnn.npz'

##################
#   Parameters   #
##################
batch_size = 256
num_epoch = 10

split = 0.2

save_weights = True

def load_dataset(split=split):
  h5f = h5py.File(fname,'r')
  features = h5f['features'][:]
  MAPs = h5f['MAPs'][:]
  h5f.close()

  N = int(features.shape[0] * split)

  X_train = features[:-N]
  y_train = MAPs[:-N]

  X_test = features[-N:]
  y_test = MAPs[-N:]

  return (X_train, y_train), (X_test, y_test)

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
  assert len(inputs) == len(targets)

  if shuffle:
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)

  for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
    if shuffle:
      excerpt = indices[start_idx:start_idx + batchsize]
    else:
      excerpt = slice(start_idx, start_idx + batchsize)
    yield inputs[excerpt], targets[excerpt]

def main():
  print 'Loading Datasets...'

  (X_train, y_train), (X_test, y_test) = load_dataset(split)

  print 'Compiling Network...'
  network, train_fn, val_fn, test_fn = dnn()

  print '[ State Estimation Neural Network Offline Training ]'
  for epoch in range(num_epoch):
    epoch_loss = 0.
    epoch_norm_one_err = 0.
    count = 1.
    for inputs,targets in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
      inputs = inputs.reshape(batch_size,1,1,inputs.shape[1])
      targets = targets.reshape(batch_size,1)
      predictions,loss = train_fn(inputs, targets)
      epoch_loss += loss
      batch_norm_one_err = np.absolute(targets-predictions).mean()
      epoch_norm_one_err += 1./count*(batch_norm_one_err - epoch_norm_one_err)
      count += 1.
    print 'Trained {0}/{1} epochs, loss: {2}, average norm_one_error: {3}'\
          .format(epoch,num_epoch,epoch_loss,epoch_norm_one_err)
  print 'Done Training!'

  print
  print '[ State Estimation Neural Network Offline Testing ]'
  test_loss = 0.
  test_norm_one_err = 0.
  count = 1.
  for inputs, targets in iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
    inputs = inputs.reshape(batch_size,1,1,inputs.shape[1])
    targets = targets.reshape(batch_size,1)
    predictions, loss = val_fn(inputs, targets)
    test_loss += loss
    batch_norm_one_err = np.absolute(targets-predictions).mean()
    test_norm_one_err += 1./count*(batch_norm_one_err - test_norm_one_err)
    count += 1.
  print 'Testing results: loss: {0}, average norm_one_error: {1}'\
        .format(test_loss * 1/split, test_norm_one_err)

  if save_weights:
    # Optionally, you could now dump the network weights to a file like this:
    np.savez( weights_file, *lasagne.layers.get_all_param_values(network) )
    #
    # And load them again later on like this:
    #with np.load( weights_file ) as f:
    #  param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    #  lasagne.layers.set_all_param_values(network, param_values)
    #test_saveweights()

def test_saveweights():

  (X_train, y_train), (X_test, y_test) = load_dataset(split)

  network, train_fn, val_fn, test_fn = dnn()

  with np.load( weights_file ) as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

  print '[ State Estimation Neural Network Offline Testing ]'
  test_loss = 0.
  test_norm_one_err = 0.
  count = 1.
  for inputs, targets in iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
    inputs = inputs.reshape(batch_size,1,1,inputs.shape[1])
    targets = targets.reshape(batch_size,1)
    predictions, loss = val_fn(inputs, targets)
    test_loss += loss
    batch_norm_one_err = np.absolute(targets-predictions).mean()
    test_norm_one_err += 1./count*(batch_norm_one_err - test_norm_one_err)
    count += 1.
  print 'Testing results: loss: {0}, average norm_one_error: {1}'\
        .format(test_loss * 1/split, test_norm_one_err)

if __name__ == "__main__":
    main()
