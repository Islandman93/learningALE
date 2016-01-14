r_tp1 = self.cnn.get_output(state_tp1s)
            rewardsasdf = rewards + (1-terminal) * self.discount * np.max(r_tp1, axis=1)

            net_out = self.cnn.get_output(states)
            est_rew = net_out[np.arange(32), actions]
            print(est_rew.shape)
            differ = rewardsasdf - net_out[np.arange(32), actions]
            cost, rews, diff = self.cnn.train(states, actions, rewards, state_tp1s, terminal)

            print(np.isclose(rews,rewardsasdf))
            print(diff, differ)
            print(np.isclose(diff, differ))
            exit()