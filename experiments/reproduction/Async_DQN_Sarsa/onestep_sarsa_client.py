import numpy as np
from learningALE.handlers.async.PipeCmds import PipeCmds
from learningALE.handlers.async.AsyncTargetLearner import AsyncTargetLearner
from learningALE.learners.async_target_cnn import AsyncTargetCNNSarsa
from functools import partial


class Async1StepSarsaLearner(AsyncTargetLearner):
    def __init__(self, num_actions, initial_cnn_values, pipe, skip_frame=4, phi_length=4, async_update_step=5,
                 discount=0.95, target_update_frames=40000):
        cnn_partial = partial(AsyncTargetCNNSarsa, (1, phi_length, 84, 84), num_actions, discount)
        super().__init__(num_actions, initial_cnn_values, cnn_partial, pipe, skip_frame=skip_frame,
                         phi_length=phi_length, async_update_step=async_update_step,
                         target_update_frames=target_update_frames)

    def run_episode(self, emulator):
        # reset game
        emulator.reset()
        self.loss_list = list()
        total_score = 0

        # run until terminal
        terminal = False

        # get initial state action pair
        state = np.asarray(emulator.get_gamescreen()/255.0, dtype=np.float32)
        # get action
        self.add_state_to_buffer(state)
        action = self.get_game_action(self.frame_buffer)

        while not terminal and not self.done:
            # step
            reward = emulator.step(action, clip=1)
            total_score += reward

            # get new state action pair
            state_tp1 = np.asarray(emulator.get_gamescreen()/255.0, dtype=np.float32)
            frame_buffer_tp1 = self.frame_buffer_with(state_tp1)
            action_tp1 = self.get_game_action(frame_buffer_tp1)

            # accumulate gradients
            loss = self.cnn.accumulate_gradients(self.frame_buffer, self.action_handler.game_action_to_action_ind(action),
                                                 reward, self.action_handler.game_action_to_action_ind(action_tp1),
                                                 frame_buffer_tp1, terminal)
            self.loss_list.append(loss)

            # update loop vars
            self.frame_buffer = frame_buffer_tp1
            action = action_tp1
            self.thread_steps += 1

            # check for terminal
            terminal = emulator.get_game_over()

            if self.thread_steps % self.async_update_step == 0 or terminal:
                # process cmds from host, this will flush pipe recv commands
                self.process_host_cmds()

                # synchronous update parameters, this sends then waits for host to send back
                new_params, global_vars = self.synchronous_update(self.cnn.get_gradients(), self.thread_steps*self.skip_frame)
                self.cnn.clear_gradients()
                self.cnn.set_parameters(new_params)

                self.update_global_vars(global_vars)
                self.action_handler.anneal_to(global_vars['counter'])
                if self.check_update_target():
                    print(self, 'setting target')
                    self.cnn.set_target_parameters(new_params)

                # if terminal send stats
                if terminal:
                    stats = {'score': total_score, 'frames': self.thread_steps*self.skip_frame, 'loss': self.loss_list}
                    self.pipe.send((PipeCmds.ClientSendingStats, stats))
        print(self, 'ending episode. Step counter:', self.thread_steps,
          'Score:', total_score, 'Current Rand Val:', self.action_handler.curr_rand_val)
