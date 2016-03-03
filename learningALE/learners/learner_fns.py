def add_state_to_buffer(self, state):
    self.frame_buffer[0, 0:self.phi_length-1] = self.frame_buffer[0, 1:self.phi_length]
    self.frame_buffer[0, self.phi_length-1] = state

def frame_buffer_with(self, state):
    empty_buffer = np.zeros((1, self.phi_length, 84, 84), dtype=np.float32)
    empty_buffer[0, 0:self.phi_length-1] = self.frame_buffer[0, 1:self.phi_length]
    empty_buffer[0, self.phi_length-1] = state
    return empty_buffer

class AsyncLearnerTester(AsyncClient):
    def __init__(self, load_file, num_actions, cnn_class, skip_frame=4, phi_length=4, rand_val=0.05):
        rand_vals = (rand_val, rand_val, 2)
        self.action_handler = ActionHandler(rand_vals)

        self.cnn = cnn_class((1, phi_length, 84, 84), num_actions)
        import pickle
        with open(load_file, 'rb') as inp_file:
            self.cnn.set_parameters(pickle.load(inp_file))

        self.frame_buffer = np.zeros((1, phi_length, 84, 84), dtype=np.float32)
        self.skip_frame = skip_frame
        self.phi_length = phi_length

    def run(self, emulator):
        # reset game
        print(self, 'starting episode. Current Rand Val:', self.action_handler.curr_rand_val)
        emulator.reset()
        self.game_over()
        total_score = 0

        # run until terminal
        terminal = False
        while not terminal:
            # get initial state
            state = np.asarray(emulator.get_gamescreen()/255.0, dtype=np.float32)

            # get action
            self.add_state_to_buffer(state)
            action = self.get_game_action(self.frame_buffer)

            # step and get new state
            reward = emulator.step(action)
            total_score += reward

            # check for terminal
            terminal = emulator.get_game_over()

        return total_score