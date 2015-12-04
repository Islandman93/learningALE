learningALE
===========

learningALE is meant to be an easy to use collection of tools and helpers for integration of learners into the
`Arcade Learning Evironment <https://github.com/mgbellemare/Arcade-Learning-Environment>`_. There have been a ton
of new papers about DQN and all it's variants but no combined place or package to quickly implement and test. That is
the main purpose of this library.

There are two things I believe are fundamental to this project:
1. Prevent over-optimization. Speed is useful but it's important to be flexible and most importantly be easy to create
new learners that have full access to all needed variables/objects/game state/etc.

2. Some of these algorithms can be very complex. Code must be commented, documented, and be easily readable.


Requirements
------------

Standard scientific packages needed: numpy, scipy, matplotlib.

Neural net specific packages: `Theano <https://github.com/Theano/Theano>`_. and
`Lasagne <https://github.com/Lasagne/Lasagne>`_.

If you're using Windows and need help installing theano check out my blog post at
http://www.islandman93.com/2015/04/tutorial-python-34-theano-and-windows-7.html

Documentation
-------------

Documentation is available online: read the docs comming soon

For support, please email islandman93 at gmail.


Example DQN Learner
-------------------

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

For a full reproduction of the first DQN paper, see
`experiments/reproduction/DQN_Original <experiments/reproduction/DQN_Original/breakout_dqn.py>`_,


Development
-----------

This project is by no means finished and will constantly improve as I have time to work on it. I readily accept pull
requests, and will try to fix issues when they come up.

I'm still pretty new to github, docs, and python tests. I welcome refactoring, advice on folder structure and file
formats.

README lovingly edited from https://github.com/Lasagne/Lasagne without that project this one wouldn't be possible.