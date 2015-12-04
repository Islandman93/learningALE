import theano
import theano.tensor as T
import theano.tensor.signal.downsample as downsample
import lasagne
from lasagne.layers import *
import numpy as np


class CNN:
    def __init__(self, inpShape, outputNum, clip=None, stride=(4, 2)):
        self.l_in = lasagne.layers.InputLayer(inpShape)

        if stride is None:
            self.l_hid1 = lasagne.layers.Conv2DLayer(self.l_in, 16, (8, 8), untie_biases=True)
        else:
            self.l_hid1 = lasagne.layers.Conv2DLayer(self.l_in, 16, (8, 8), stride=stride[0], untie_biases=True)

        self.l_pool1 = lasagne.layers.MaxPool2DLayer(self.l_hid1, 2)

        if stride is None:
            self.l_hid2 = lasagne.layers.Conv2DLayer(self.l_pool1, 32, (4, 4), untie_biases=True)
        else:
            self.l_hid2 = lasagne.layers.Conv2DLayer(self.l_pool1, 32, (4, 4), stride=stride[1], untie_biases=True)
        self.l_pool2 = lasagne.layers.MaxPool2DLayer(self.l_hid2, 2)

        self.l_hid3 = lasagne.layers.DenseLayer(self.l_pool2, 256)
        self.l_out = lasagne.layers.DenseLayer(self.l_hid3, outputNum, nonlinearity=lasagne.nonlinearities.linear)

        net_output = lasagne.layers.get_output(self.l_out)
        truth = T.matrix()
        mask = T.matrix()
        loss = T.mean(mask*(net_output-truth)**2)

        params = lasagne.layers.get_all_params(self.l_out)

        # if clipping the grad
        if clip is not None:
            grads = lasagne.updates.total_norm_constraint(T.grad(loss, params), clip)
        else:
            grads = T.grad(loss, params)

        update = lasagne.updates.rmsprop(grads, params, 0.002)

        self.train = theano.function([self.l_in.input_var, truth, mask], loss, updates=update)
        self.get_output = theano.function([self.l_in.input_var], outputs=net_output)
        self.get_hid1_act = theano.function([self.l_in.input_var], outputs=lasagne.layers.get_output(self.l_hid1))
        self.get_hid2_act = theano.function([self.l_in.input_var], outputs=lasagne.layers.get_output(self.l_hid2))

    def load(self, file):
        import pickle
        with open(file, 'rb') as infile:
            parms = pickle.load(infile)
            lasagne.layers.set_all_param_values(self.l_out, parms)

    def save(self, file):
        parms = lasagne.layers.get_all_param_values(self.l_out)
        import pickle
        with open(file,'wb') as outfile:
            pickle.dump(parms, outfile)


class SPPLayer(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(SPPLayer, self).__init__(incoming, **kwargs)

        # divide by 4 gives 16 patches
        self.win1 = (int(np.floor(incoming.output_shape[2]/4.0)), int(np.floor(incoming.output_shape[3]/4.0)))
        self.str1 = (int(np.ceil(incoming.output_shape[2]/4.0)), int(np.ceil(incoming.output_shape[3]/4.0)))

        # divide by 2 gives 4 patches
        self.win2 = (int(np.floor(incoming.output_shape[2]/2.0)), int(np.floor(incoming.output_shape[3]/2.0)))
        self.str2 = (int(np.ceil(incoming.output_shape[2]/2.0)), int(np.ceil(incoming.output_shape[3]/2.0)))

        # no divide is one max patch, this is achieved by just doing T.maximum after reshaping

    def get_output_for(self, input, **kwargs):
        p1 = T.reshape(downsample.max_pool_2d(input, ds=self.win1, st=self.str1), (input.shape[0], input.shape[1], 16))
        p2 = T.reshape(downsample.max_pool_2d(input, ds=self.win2, st=self.str2), (input.shape[0], input.shape[1], 4))
        r3 = T.reshape(input, (input.shape[0], input.shape[1], input.shape[2]*input.shape[3]))
        p3 = T.reshape(T.max(r3, axis=2), (input.shape[0], input.shape[1], 1))
        return T.concatenate((p1, p2, p3), axis=2)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], 21)

class AlloEggoCnn:
    def __init__(self, inpShape, outputNum, clip):
        self.input = T.tensor4()
        # allocentric lasagne setup
        self.a_in = lasagne.layers.InputLayer(inpShape, input_var=self.input)
        self.a_hid1 = lasagne.layers.Conv2DLayer(self.a_in, 16, (8, 8))
        self.a_pool1 = lasagne.layers.MaxPool2DLayer(self.a_hid1, 2)
        self.a_hid2 = lasagne.layers.Conv2DLayer(self.a_pool1, 32, (4, 4))
        self.a_spp = SPPLayer(self.a_hid2)
        self.a_hid3 = lasagne.layers.DenseLayer(self.a_spp, 256)
        self.a_out = lasagne.layers.DenseLayer(self.a_hid3, outputNum, nonlinearity=lasagne.nonlinearities.linear)

        # egocentric lasagne setup
        self.e_input = self.input[:, :, 43:, :]
        self.l_in = lasagne.layers.InputLayer((inpShape[0], inpShape[1], 43, 80), input_var=self.e_input)
        self.l_hid1 = lasagne.layers.Conv2DLayer(self.l_in, 16, (8, 8), W=self.a_hid1.W, b=self.a_hid1.b)
        self.l_pool1 = lasagne.layers.MaxPool2DLayer(self.l_hid1, 2)
        self.l_hid2 = lasagne.layers.Conv2DLayer(self.l_pool1, 32, (4, 4), W=self.a_hid2.W, b=self.a_hid2.b)
        self.l_spp = SPPLayer(self.l_hid2)
        self.l_hid3 = lasagne.layers.DenseLayer(self.l_spp, 256, W=self.a_hid3.W, b=self.a_hid3.b)
        self.e_out = lasagne.layers.DenseLayer(self.l_hid3, outputNum, nonlinearity=lasagne.nonlinearities.linear,
                                               W=self.a_out.W, b=self.a_out.b)

        allo_output = lasagne.layers.get_output(self.a_out)
        ego_output = lasagne.layers.get_output(self.e_out)

        self.target = T.matrix()
        self.mask = T.matrix()

        allo_loss = T.mean(((self.target - allo_output)*self.mask)**2)
        ego_loss = T.mean(((self.target - ego_output)*self.mask)**2)
        loss = (allo_loss + ego_loss)/2
        # loss = ego_loss

        params = lasagne.layers.get_all_params(self.a_out)
        grads = lasagne.updates.total_norm_constraint(T.grad(loss, params), clip)
        update = lasagne.updates.rmsprop(grads, params, 0.002)
        self.train = theano.function([self.input, self.target, self.mask], loss, updates=update)
        self.get_output = theano.function([self.input], outputs=(allo_output+ego_output)/2)
        # self.get_output = theano.function([self.input], outputs=(ego_output))


class LSTM:
    def __init__(self, inpShape, outputNum, clip):
        num_units = 256
        # By setting the first two dimensions as None, we are allowing them to vary
        # They correspond to batch size and sequence length, so we will be able to
        # feed in batches of varying size with sequences of varying length.
        self.l_inp = InputLayer(inpShape)
        # We can retrieve symbolic references to the input variable's shape, which
        # we will later use in reshape layers.
        batchsize, seqlen, _ = self.l_inp.input_var.shape
        self.l_lstm = LSTMLayer(self.l_inp, num_units=num_units)
        # In order to connect a recurrent layer to a dense layer, we need to
        # flatten the first two dimensions (our "sample dimensions"); this will
        # cause each time step of each sequence to be processed independently
        l_shp = ReshapeLayer(self.l_lstm, (-1, num_units))
        self.l_dense = DenseLayer(l_shp, num_units=outputNum)
        # To reshape back to our original shape, we can use the symbolic shape
        # variables we retrieved above.
        self.l_out = ReshapeLayer(self.l_dense, (batchsize, seqlen, outputNum))

        net_output = lasagne.layers.get_output(self.l_out)
        truth = T.tensor3()
        mask = T.tensor3()
        loss = T.mean(mask*(net_output-truth)**2)

        params = lasagne.layers.get_all_params(self.l_out)
        grads = lasagne.updates.total_norm_constraint(T.grad(loss, params), clip)
        update = lasagne.updates.rmsprop(grads, params, 0.002)

        self.train = theano.function([self.l_inp.input_var, truth, mask], loss, updates=update)
        self.get_output = theano.function([self.l_inp.input_var], outputs=net_output)


class RAM:
    def __init__(self):
        self.l_in = lasagne.layers.InputLayer((None, 128*4))
        self.l_hid1 = lasagne.layers.DenseLayer(self.l_in, 1024)
        self.l_hid2 = lasagne.layers.DenseLayer(self.l_hid1, 256)
        self.l_out = lasagne.layers.DenseLayer(self.l_hid2, 3, nonlinearity=lasagne.nonlinearities.linear)

        net_output = lasagne.layers.get_output(self.l_out)

        objective = lasagne.objectives.MaskedObjective(self.l_out)
        loss = objective.get_loss()

        params = lasagne.layers.get_all_params(self.l_out)
        update = lasagne.updates.rmsprop(loss, params, 0.0002)
        self.train = theano.function([self.l_in.input_var, objective.target_var, objective.mask_var], loss, updates=update)
        self.get_output = theano.function([self.l_in.input_var], outputs=net_output)

class AlloEggoSeperateCnn:
    def __init__(self):
        self.input = T.tensor4()
        # allocentric lasagne setup
        self.a_in = lasagne.layers.InputLayer((None, 4, 105, 80), input_var=self.input)
        self.a_hid1 = lasagne.layers.Conv2DLayer(self.a_in, 16, (8, 8))
        self.a_pool1 = lasagne.layers.MaxPool2DLayer(self.a_hid1, 2)
        self.a_hid2 = lasagne.layers.Conv2DLayer(self.a_pool1, 32, (4, 4))
        self.a_spp = SPPLayer(self.a_hid2)
        self.a_hid3 = lasagne.layers.DenseLayer(self.a_spp, 256)
        self.a_out = lasagne.layers.DenseLayer(self.a_hid3, 3, nonlinearity=lasagne.nonlinearities.linear)

        # egocentric lasagne setup
        self.e_input = self.input[:, :, 47:, :]
        self.l_in = lasagne.layers.InputLayer((None, 4, 58, 80), input_var=self.e_input)
        self.l_hid1 = lasagne.layers.Conv2DLayer(self.l_in, 16, (8, 8))#, W=self.a_hid1.W, b=self.a_hid1.b)
        self.l_pool1 = lasagne.layers.MaxPool2DLayer(self.l_hid1, 2)
        self.l_hid2 = lasagne.layers.Conv2DLayer(self.l_pool1, 32, (4, 4))#, W=self.a_hid2.W, b=self.a_hid2.b)
        self.l_spp = SPPLayer(self.l_hid2)
        self.l_hid3 = lasagne.layers.DenseLayer(self.l_spp, 256)#, W=self.a_hid3.W, b=self.a_hid3.b)
        self.e_out = lasagne.layers.DenseLayer(self.l_hid3, 3, nonlinearity=lasagne.nonlinearities.linear)#,
                                               #W=self.a_out.W, b=self.a_out.b)

        allo_output = lasagne.layers.get_output(self.a_out)
        ego_output = lasagne.layers.get_output(self.e_out)

        self.target = T.matrix()
        self.mask = T.matrix()

        allo_loss = T.mean(((self.target - allo_output)*self.mask)**2)
        ego_loss = T.mean(((self.target - ego_output)*self.mask)**2)
        loss = (allo_loss + ego_loss)/2
        # loss = ego_loss

        params = lasagne.layers.get_all_params(self.a_out)
        egoParams = lasagne.layers.get_all_params(self.e_out)
        update = lasagne.updates.rmsprop(allo_loss, params, 0.002)
        egoUpdate = lasagne.updates.rmsprop(ego_loss, egoParams, 0.002)
        alloTrain = theano.function([self.input, self.target, self.mask], allo_loss, updates=update)
        egoTrain = theano.function([self.input, self.target, self.mask], ego_loss, updates=egoUpdate)
        def trainBoth(trainInput, trainTarget, trainMask):
            alloLoss = alloTrain(trainInput, trainTarget, trainMask)
            egoLoss = egoTrain(trainInput, trainTarget, trainMask)
            return alloLoss + egoLoss

        self.train = trainBoth
        self.get_output = theano.function([self.input], outputs=(allo_output+ego_output)/2)
        # self.get_output = theano.function([self.input], outputs=(ego_output))