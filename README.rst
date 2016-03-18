learningALE
===========

learningALE is meant to be an easy to use collection of tools, helpers, and example learners for the
`Arcade Learning Evironment <https://github.com/mgbellemare/Arcade-Learning-Environment>`_. There have been a ton
of new papers about DQN and all it's variants but no combined place or package to quickly implement and test. That is
the main purpose of this library.

There are three things I believe are fundamental to this project:

1. Premature optimization is the root of all evil. That being said this code needs to be fast so we don't have to wait
weeks for it to train. Try to be smart about where you put optimizations so that they don't obfuscate your code.
2. Some of these algorithms can be very complex. Code must be commented, documented, and be easily readable.
3. On the same note as 2. Try to prevent 'spaghetti' code as much as possible. While it might be a better programming
practice to modularize code and prevent duplicate code we must prevent over-modularization. If a learner is composed of
10 different files it becomes impossible to read or to change just one thing as we so often do in research. Because of
this I try to keep almost all of the code for a learner in its own file. This causes code duplication but makes it easy
to read and to change.

Requirements & Installation
---------------------------

Standard scientific packages needed: numpy, scipy.

Neural net specific packages: `Theano <https://github.com/Theano/Theano>`_ and
`Lasagne <https://github.com/Lasagne/Lasagne>`_.

If you're using Windows and need help installing theano check out my blog post at
http://www.islandman93.com/2015/04/tutorial-python-34-theano-and-windows-7.html

DLLs for the ALE are included for Windows, Linux users will need to download and compile the
`Arcade Learning Evironment <https://github.com/mgbellemare/Arcade-Learning-Environment>`_.

After installing dependencies just use python setup.py install.

Documentation
-------------

Documentation is available online: http://learningale.readthedocs.org/

For support, please email islandman93 at gmail or submit an issue/pull request.


Example DQN Learner
-------------------

.. code-block:: python

    from learningALE.learners.learner import learner
    from learningALE.handlers.actionhandler import ActionHandler, ActionPolicy
    from learningALE.handlers.experiencehandler import ExperienceHandler
    from learningALE.handlers.trainhandler import TrainHandler
    from learningALE.learners.nns import CNN
    from learningALE.tools.life_ram_inds import BREAKOUT

    class DQNLearner(learner):
        def __init__(self, skip_frame, num_actions):
            super().__init__()

            rand_vals = (1, 0.1, 1000000/skip_frame)  # starting at 1 anneal eGreedy policy to 0.1 over 1,000,000*skip_frame
            self.action_handler = ActionHandler(ActionPolicy.eGreedy, rand_vals)

            self.exp_handler = ExperienceHandler(1000000/skip_frame)
            self.train_handler = TrainHandler(32, 0.9, num_actions)
            self.cnn = CNN((None, skip_frame, 86, 80), num_actions, .1)

        def get_action(self, game_input):
            return self.cnn.get_output(game_input)[0]

        def get_game_action(self, game_input):
            return self.action_handler.action_vect_to_game_action(self.get_action(game_input))

        def frames_processed(self, frames, action_performed, reward):
            self.exp_handler.add_experience(frames, self.action_handler.game_action_to_action_ind(action_performed), reward)
            self.train_handler.train_exp(self.exp_handler, self.cnn)
            self.action_handler.anneal()

        def set_legal_actions(self, legal_actions):
            self.action_handler.set_legal_actions(legal_actions)
    
    # setup vars
    rom = b'path_to_rom'
    gamename = 'breakout'
    skip_frame = 4
    num_actions = 4
    learner = DQNLearner(skip_frame, num_actions)
    game_handler = GameHandler(rom, False, learner, skip_frame)
    
    st = time.time()
    for episode in range(1):
        total_reward = game_handler.run_one_game(learner, lives=5, life_ram_ind=BREAKOUT)
    
        et = time.time()
        print("Episode " + str(episode) + " ended with score: " + str(total_reward))
        print('Total Time:', et - st, 'Frame Count:', game_handler.frameCount, 'FPS:', game_handler.frameCount / (et - st))
        
    # save
    learner.save('dqn{0}.pkl'.format(episode+1))

For a full reproduction of the first DQN paper, see
`experiments/reproduction/DQN_Original <experiments/reproduction/DQN_Original/>`_.


Development
-----------

This project is by no means finished and will constantly improve as I have time to work on it. I readily accept pull
requests, and will try to fix issues when they come up. All python code should use the PEP standard, and anything that
isn't PEP should be refactored (including my own code).

I'm still pretty new to github, docs, and python tests. I welcome refactoring, advice on folder structure and file
formats.

README lovingly edited from https://github.com/Lasagne/Lasagne without that project this one wouldn't be possible.