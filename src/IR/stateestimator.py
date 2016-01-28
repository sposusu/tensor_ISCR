import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

class StateEstimater(object):
    def __init__(self,training_method = 'online'):
        self.approximator = Approximator()

    def online_training(self):
        pass

    def offline_training(self):
        pass

def Approximator():
    model = Sequential()
    model.add(Dense(64, input_dim=89, init='uniform', activation='tanh'))
    model.add(Dense(1, init='uniform', activation='softmax'))
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model
