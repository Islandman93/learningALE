from learningALE.handlers.async.AsyncHost import AsyncLearnerHost
from learningALE.handlers.async.PipeCmds import PipeCmds


class NoveltyHost(AsyncLearnerHost):
    def __init__(self, host_cnn, learners, rom, skip_frame=4, show_rom=False):
        super().__init__(host_cnn, learners, rom, skip_frame, show_rom)
        self.novel_states = dict()

    def process_pipe(self, learner_ind, pipe):
        pipe_cmd, extras = pipe.recv()
        if pipe_cmd == PipeCmds.ClientSendingGradientsSteps:
            self.cnn.gradient_step(extras[0])
            self.process_novel_frames(extras[1])
            self.learner_frames[learner_ind] = extras[2]
            # send back new parameters to client
            pipe.send((PipeCmds.HostSendingGlobalParameters,
                       (self.cnn.get_parameters(), {'counter': sum(self.learner_frames), 'novel_frames': self.novel_states})))
        if pipe_cmd == PipeCmds.ClientSendingStats:
            self.learner_stats[learner_ind].append(extras)
            if extras['score'] > self.best_score:
                self.best_score = extras['score']

    def process_novel_frames(self, new_novel_frames):
        for state, count in new_novel_frames:
            if state in self.novel_states:
                self.novel_states[state] += count
            else:
                self.novel_states[state] = count