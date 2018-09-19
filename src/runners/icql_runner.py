from .episode_runner import EpisodeRunner
from random import random


class ICQLEpisodeRunner(EpisodeRunner):
    def run(self, test_mode=False, **kwargs):
        """ Chooses randomly to use the critic or the IQL policy for this episode.
            Running in test_mode samples only the IQL policy. """
        use_critic = random() < self.args.critic_sampling_prob and not test_mode
        return EpisodeRunner.run(self, test_mode, use_critic=use_critic, **kwargs)