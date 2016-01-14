import time

def evaluate_player(learner, game_handler, num_to_evaluate):
    rews = list()
    times = list()
    frames = list()
    old_frame_count = 0
    for test in range(num_to_evaluate):
        st = time.time()
        total_reward = game_handler.run_one_game(learner, neg_reward=False, clip=False)
        et = time.time()
        rews.append(total_reward)
        times.append(et-st)
        frames.append(game_handler.frameCount - old_frame_count)
        old_frame_count = game_handler.frameCount
    return rews, times, frames


