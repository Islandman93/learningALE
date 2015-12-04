# __author__ = 'Ben'
# import numpy as np
# from scipy.special import expit
# import copy
# import theano
# import lasagne
# import theano.tensor as T
# from functools import partial
# import line_profiler
#
#
# class cosyne:
#     def __init__(self, netArchitecture, subPopLen):
#         self.nIn = netArchitecture[0]
#         self.nHid = netArchitecture[1]
#         self.nOut = netArchitecture[2]
#         self.subPopLen = subPopLen
#
#         numToBreed = np.floor(subPopLen/4)
#         if numToBreed % 2 != 0:
#             numToBreed -= 1
#
#         self.numToBreed = numToBreed
#         self.subpop_nodes = np.random.random((subPopLen, self.nIn*self.nHid + self.nHid*self.nOut))
#         self.subpop_nodes *= 0.5
#         self.subpop_nodes -= .25
#         self.hidInds = np.arange(self.nIn*self.nHid)
#         self.outInds = np.arange(self.nIn*self.nHid, len(self.hidInds) + self.nHid*self.nOut)
#         self.fitness = list()
#
#     # @profile
#     def trainOneEpoch(self, evaluator):
#         pass
#
#     # @profile
#     def _breed(self, fitn, weightSize):
#         # breed new node weights
#         sortInds = np.argsort(fitn)
#         merge = sortInds[0:self.numToBreed]
#         p1 = np.random.choice(merge[0::2], size=self.numToBreed/2, replace=False)
#         p2 = np.random.choice(merge[1::2], size=self.numToBreed/2, replace=False)
#         offspring = np.zeros((self.numToBreed/2, self.subpop_nodes.shape[1]))
#         offspring[:, 0::2] = self.subpop_nodes[p1, 0::2]
#         offspring[:, 1::2] = self.subpop_nodes[p2, 1::2]
#         mutation = np.random.uniform(size=offspring.shape)
#         mutationInd = np.where(mutation < 0.0005)
#         offspring[mutationInd[0], mutationInd[1]] += np.random.standard_normal(mutationInd[0].size)*weightSize
#         self.subpop_nodes[sortInds[-self.numToBreed/2:]] = offspring
#
#         # permute weights of non offspring
#         # for weightInd in range(self.subpop_nodes.shape[1]):
#         #     np.random.shuffle(self.subpop_nodes[sortInds[0:-self.numToBreed/2], weightInd])
#
# #
# # from theano.tensor.nnet import conv
# # from theano.tensor.signal import downsample
# #
# # netIn = T.tensor4()
# # w1 = T.tensor4()
# # b1 = T.vector()
# # w2 = T.tensor4()
# # b2 = T.vector()
# # w3 = T.matrix()
# # b3 = T.vector()
# # wout = T.matrix()
# # bout = T.vector()
# #
# # conv1 = lasagne.nonlinearities.tanh(conv.conv2d(netIn, w1) + b1.dimshuffle('x', 0, 'x', 'x'))
# # conv1Ds = downsample.max_pool_2d(conv1, (2,2))
# # conv2 = lasagne.nonlinearities.tanh(conv.conv2d(conv1Ds, w2) + b2.dimshuffle('x', 0, 'x', 'x'))
# # conv2Ds = downsample.max_pool_2d(conv2, (2,2))
# # dense = lasagne.nonlinearities.tanh(T.dot(T.reshape(conv2Ds, (netIn.shape[0], -1)), w3) + b3)
# # netOut = T.dot(dense, wout) + bout
# #
# # getNetOut = theano.function([w1, b1, w2, b2, w3, b3, wout, bout, netIn], outputs=[netOut, conv1], allow_input_downcast=True)
#
#
# class cosyneBreakout(cosyne):
#     def __init__(self, subPopLen):
#         self.nIn = (-1, 4, 105, 80)
#         self.nOut = 3
#         self.subPopLen = subPopLen
#
#         numToBreed = np.floor(subPopLen/4)
#         if numToBreed % 2 != 0:
#             numToBreed -= 1
#
#         self.numToBreed = numToBreed
#
#         self.filt1WInd = 4*8*8*8
#         self.filt1BInd = self.filt1WInd + 8
#         self.filt2WInd = self.filt1BInd + 8*16*4*4
#         self.filt2BInd = self.filt2WInd + 16
#         self.dense3WInd = self.filt2BInd + 256*16*23*17
#         self.dense3BInd = self.dense3WInd + 256
#         self.outWInd = self.dense3BInd + 256*3
#         self.outBInd = self.outWInd + 3
#
#         self.subpop_nodes = np.random.standard_normal((subPopLen, self.outBInd))
#         self.subpop_nodes *= 0.01
#         self.fitness = list()
#
#     def trainOneEpoch(self, evaluator):
#         fitnessNodes = np.zeros(self.subpop_nodes.shape[0])
#
#         for net in range(self.subPopLen):
#             filt1W = self.subpop_nodes[net, 0:self.filt1WInd].reshape((8, 4, 8, 8))
#             filt1B = self.subpop_nodes[net, self.filt1WInd:self.filt1BInd]
#             filt2W = self.subpop_nodes[net, self.filt1BInd:self.filt2WInd].reshape((16, 8, 4, 4))
#             filt2B = self.subpop_nodes[net, self.filt2WInd:self.filt2BInd]
#             dense3W = self.subpop_nodes[net, self.filt2BInd:self.dense3WInd].reshape((16*23*17, 256))
#             dense3B = self.subpop_nodes[net, self.dense3WInd:self.dense3BInd]
#             outW = self.subpop_nodes[net, self.dense3BInd:self.outWInd].reshape((256, 3))
#             outB = self.subpop_nodes[net, self.outWInd:self.outBInd]
#
#             fitness = evaluator.evaluate(partial(getNetOut, filt1W, filt1B, filt2W, filt2B, dense3W, dense3B, outW, outB), earlyReturn=True)
#             fitnessNodes[net] += -1*fitness
#
#         # breed new node weights
#         self._breed(fitnessNodes)
#         self.fitness.append(copy.deepcopy(fitnessNodes))
#
#
# class cosyneBreakoutRam(cosyne):
#     def __init__(self, subPopLen):
#         self.nIn = (-1, 128)
#         self.nOut = 3
#         self.subPopLen = subPopLen
#
#         numToBreed = np.floor(subPopLen/4)
#         if numToBreed % 2 != 0:
#             numToBreed -= 1
#
#         self.numToBreed = numToBreed
#
#         self.dense1WInd = 128*256
#         self.dense1BInd = self.dense1WInd + 256
#         self.outWInd = self.dense1BInd + 256*3
#         self.outBInd = self.outWInd + 3
#
#         self.subpop_nodes = np.random.standard_normal((subPopLen, self.outBInd))
#         self.subpop_nodes *= 0.3
#         self.fitness = list()
#
#     def trainOneEpoch(self, evaluator):
#         fitnessNodes = np.zeros(self.subpop_nodes.shape[0])
#
#         for net in range(self.subPopLen):
#             dense1W = self.subpop_nodes[net, 0:self.dense1WInd].reshape((128, 256))
#             dense1B = self.subpop_nodes[net, self.dense1WInd:self.dense1BInd]
#             outW = self.subpop_nodes[net, self.dense1BInd:self.outWInd].reshape((256, 3))
#             outB = self.subpop_nodes[net, self.outWInd:self.outBInd]
#
#             def getNetOut(netInput):
#                 denseAct = np.maximum(0, (np.dot(netInput, dense1W) + dense1B))
#                 return (np.dot(denseAct, outW) + outB)[0]
#
#             fitness = evaluator.evaluate(getNetOut, 5, 57, earlyReturn=True)
#             print(net, fitness)
#             fitnessNodes[net] += -1*fitness
#
#         # breed new node weights
#         self._breed(fitnessNodes, 0.3)
#         self.fitness.append(copy.deepcopy(fitnessNodes))
