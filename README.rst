.. image:: http://img.shields.io/badge/docs-latest-brightgreen.svg
    :target: http://lasagne.readthedocs.org/en/latest/

.. image:: https://travis-ci.org/Lasagne/Lasagne.svg?branch=master
    :target: https://travis-ci.org/Lasagne/Lasagne

.. image:: https://img.shields.io/coveralls/Lasagne/Lasagne.svg
    :target: https://coveralls.io/r/Lasagne/Lasagne

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: https://github.com/Lasagne/Lasagne/blob/master/LICENSE

.. image:: https://zenodo.org/badge/16974/Lasagne/Lasagne.svg
   :target: https://zenodo.org/badge/latestdoi/16974/Lasagne/Lasagne

learningALE
===========

Lasagne is a lightweight library to build and train neural networks in Theano.
Its main features are:

* Supports feed-forward networks such as Convolutional Neural Networks (CNNs),
  recurrent networks including Long Short-Term Memory (LSTM), and any
  combination thereof
* Allows architectures of multiple inputs and multiple outputs, including
  auxiliary classifiers
* Many optimization methods including Nesterov momentum, RMSprop and ADAM
* Freely definable cost function and no need to derive gradients due to
  Theano's symbolic differentiation
* Transparent support of CPUs and GPUs due to Theano's expression compiler

Its design is governed by `six principles
<http://lasagne.readthedocs.org/en/latest/user/development.html#philosophy>`_:

* Simplicity: Be easy to use, easy to understand and easy to extend, to
  facilitate use in research
* Transparency: Do not hide Theano behind abstractions, directly process and
  return Theano expressions or Python / numpy data types
* Modularity: Allow all parts (layers, regularizers, optimizers, ...) to be
  used independently of Lasagne
* Pragmatism: Make common use cases easy, do not overrate uncommon cases
* Restraint: Do not obstruct users with features they decide not to use
* Focus: "Do one thing and do it well"


Installation
------------

In short, you can install a known compatible version of Theano and the latest
Lasagne development version via:

.. code-block:: bash

  pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt
  pip install https://github.com/Lasagne/Lasagne/archive/master.zip

For more details and alternatives, please see the `Installation instructions
<http://lasagne.readthedocs.org/en/latest/user/installation.html>`_.


Documentation
-------------

Documentation is available online: http://lasagne.readthedocs.org/

For support, please refer to the `lasagne-users mailing list
<https://groups.google.com/forum/#!forum/lasagne-users>`_.


Example
-------

.. code-block:: python

    
    import numpy as np
    from learningALE.handlers.gamehandler import GameHandler
    from learningALE.learners.learner import learner
    from learningALE.handlers.actionhandler import ActionHandler, ActionPolicy
    from learningALE.handlers.experiencehandler import ExperienceHandler
    from learningALE.learners.nns import CNN
    
    # setup learner class
    class DQNLearner(learner):
        def __init__(self, skip_frame, num_actions):
            super().__init__()
    
            rand_vals = (1, 0.1, 1000000*skip_frame)  # starting at 1 anneal eGreedy policy to 0.1 over 1,000,000*skip_frame
            self.action_handler = ActionHandler(ActionPolicy.eGreedy, rand_vals)
            # minibatch 32, discount 0.9, num frame steps to keep 1,000,000/4
            self.expHandler = ExperienceHandler(32, 0.9, 1000000/skip_frame, num_actions) 
            self.cnn = CNN((None, skip_frame, 86, 80), num_actions, .1)
        
        def get_action(self, game_input):
            return self.cnn.get_output(game_input)[0]
    
        def get_game_action(self, game_input):
            return self.action_handler.action_vect_to_game_action(self.get_action(game_input))
    
        def frames_processed(self, frames, action_performed, reward):
            self.expHandler.addExperience(frames, self.action_handler.game_action_to_action_ind(action_performed), reward)
            self.expHandler.train_exp(self.cnn)
            self.action_handler.anneal()
    
        def set_legal_actions(self, legal_actions):
            self.action_handler.set_legal_actions(legal_actions)
    
        def save(self, file):
            self.cnn.save(file)
    
    # setup vars
    rom = b'path_to_rom'
    gamename = 'breakout'
    skip_frame = 4
    num_actions = 4
    learner = DQNLearner(skip_frame, num_actions)
    game_handler = GameHandler(rom, False, learner, skip_frame)
    
    st = time.time()
    for episode in range(1):
        total_reward = game_handler.run_one_game(learner)
    
        et = time.time()
        print("Episode " + str(episode) + " ended with score: " + str(total_reward))
        print('Total Time:', et - st, 'Frame Count:', game_handler.frameCount, 'FPS:', game_handler.frameCount / (et - st))
        
    # save
    learner.save('dqn{0}.pkl'.format(episode+1))

For a reproduction of the first DQN paper, see `experiments/reproduction/DQN_Original <experiments/reproduction/DQN_Original/breakout_dqn.py>`_,


Development
-----------

Lasagne is a work in progress, input is welcome.

Please see the `Contribution instructions
<http://lasagne.readthedocs.org/en/latest/user/development.html>`_ for details
on how you can contribute!

README lovingly edited from https://github.com/Lasagne/Lasagne