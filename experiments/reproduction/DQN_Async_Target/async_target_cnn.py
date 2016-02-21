import numpy as np
from collections import OrderedDict
import theano
import theano.tensor as T
import lasagne
from learningALE.learners.nns import create_Async

class AsyncTargetCNN:
    def __init__(self, inpShape, outputNum, stride=(4, 2), untie_biases=False):
        # setup shared vars
        self.state = theano.shared(np.zeros((1, inpShape[1], inpShape[2], inpShape[3]), dtype=theano.config.floatX))
        self.state_tp1 = theano.shared(np.zeros((1, inpShape[1], inpShape[2], inpShape[3]), dtype=theano.config.floatX))

        network_dic = create_Async(inpShape, outputNum, stride=stride, untie_biases=untie_biases)
        self.l_in = network_dic['l_in']
        self.l_hid1 = network_dic['l_hid1']
        self.l_out = network_dic['l_out']
        self.target_l_out = create_Async(inpShape, outputNum, stride, untie_biases)['l_out']
        lasagne.layers.set_all_param_values(self.target_l_out, lasagne.layers.get_all_param_values(self.l_out))

        # network output vars
        net_output = lasagne.layers.get_output(self.l_out, inputs=self.state)
        target_output = lasagne.layers.get_output(self.target_l_out, inputs=self.state_tp1)

        # setup qlearning values and loss
        terminal = T.scalar()
        reward = T.scalar()
        action = T.iscalar()
        est_rew_tp1 = (1-terminal) * 0.95 * T.max(target_output, axis=1)
        y = reward + est_rew_tp1
        loss = T.mean((y - net_output[0, action])**2)

        # get layer parms
        params = lasagne.layers.get_all_params(self.l_out)
        grads = T.grad(loss, params)
        grads.append(loss)

        # updates
        w1_update = T.tensor4()
        w2_update = T.tensor4()
        if untie_biases:
            b1_update = T.tensor3()
            b2_update = T.tensor3()
        else:
            b1_update = T.vector()
            b2_update = T.vector()
        w3_update = T.matrix()
        b3_update = T.vector()
        w4_update = T.matrix()
        b4_update = T.vector()
        network_updates = [w1_update, b1_update, w2_update, b2_update, w3_update, b3_update, w4_update, b4_update]
        theano_updates = lasagne.updates.rmsprop(network_updates, params, 0.0001)
        # theano_updates = OrderedDict()
        # for param, update in zip(params, network_updates):
        #     theano_updates[param] = param - 0.0001 * update

        self._get_grads = theano.function([action, reward, terminal], grads)
        self._get_loss = theano.function([action, reward, terminal], loss)
        self._get_output = theano.function([], net_output)
        self._get_target_output = theano.function([], target_output)
        self._gradient_step = theano.function(network_updates, updates=theano_updates)

        self.accumulated_grads = None

    def accumulate_gradients(self, state, action, reward, state_tp1, terminal):
        self.state.set_value(state)
        self.state_tp1.set_value(state_tp1)
        grads = self._get_grads(action, reward, terminal)
        loss = grads.pop()
        if self.accumulated_grads is None:
            self.accumulated_grads = grads
        else:
            for new_grad, acc_grad in zip(grads, self.accumulated_grads):
                acc_grad += new_grad
        return loss

    def get_output(self, state):
        self.state.set_value(state)
        return self._get_output()

    def gradient_step(self, gradients):
        self._gradient_step(*gradients)  # I love python http://stackoverflow.com/questions/3941517/converting-list-to-args-in-python

    def set_parameters(self, new_parms):
        lasagne.layers.set_all_param_values(self.l_out, new_parms)

    def get_parameters(self):
        return lasagne.layers.get_all_param_values(self.l_out)

    def set_target_parameters(self, new_parms):
        lasagne.layers.set_all_param_values(self.target_l_out, new_parms)

    def get_gradients(self):
        return self.accumulated_grads

    def clear_gradients(self):
        self.accumulated_grads = None