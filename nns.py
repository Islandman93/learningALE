import theano
import theano.tensor as T
import lasagne
import numpy as np

class CNN:
    def __init__(self):
        self.l_in = lasagne.layers.InputLayer((None, 4, 105, 80))
        self.l_hid1 = lasagne.layers.Conv2DLayer(self.l_in, 16, (8, 8))
        self.l_pool1 = lasagne.layers.MaxPool2DLayer(self.l_hid1, 2)
        self.l_hid2 = lasagne.layers.Conv2DLayer(self.l_pool1, 32, (4, 4))
        self.l_pool2 = lasagne.layers.MaxPool2DLayer(self.l_hid2, 2)
        # output_shape = lasagne.layers.get_output_shape(self.l_pool2)[1:]
        # sum_output = int(np.prod(output_shape))
        # self.l_pool2re = lasagne.layers.reshape(self.l_pool2, (-1, sum_output))
        # self.l_split = lasagne.layers.DenseLayer(self.l_in, 1000)

        # self.l_merge = lasagne.layers.ConcatLayer((self.l_split, self.l_pool2re))

        self.l_hid3 = lasagne.layers.DenseLayer(self.l_pool2, 256)
        self.l_out = lasagne.layers.DenseLayer(self.l_hid3, 3, nonlinearity=lasagne.nonlinearities.linear)

        net_output = lasagne.layers.get_output(self.l_out)

        objective = lasagne.objectives.MaskedObjective(self.l_out)
        loss = objective.get_loss()

        params = lasagne.layers.get_all_params(self.l_out)
        update = lasagne.updates.rmsprop(loss, params, 0.001)

        self.train = theano.function([self.l_in.input_var, objective.target_var, objective.mask_var], loss, updates=update)
        self.get_output = theano.function([self.l_in.input_var], outputs=net_output)


class LSTM:
    def __init__(self):
        self.l_in = lasagne.layers.InputLayer((None, 4, 105*80))
        self.l_hid1 = lasagne.layers.LSTMLayer(self.l_in, 105*80)
        print(self.l_hid1.get_output_shape())

        self.l_out = lasagne.layers.DenseLayer(self.l_hid1, 256, nonlinearity=lasagne.nonlinearities.linear)

        net_output = lasagne.layers.get_output(self.l_out)

        objective = lasagne.objectives.Objective(self.l_out)
        loss = objective.get_loss()

        params = lasagne.layers.get_all_params(self.l_out)
        update = lasagne.updates.rmsprop(loss, params)

        self.train = theano.function([self.l_in.input_var, objective.target_var], loss, updates=update)
        self.get_output = theano.function([self.l_in.input_var], outputs=net_output)