from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Process, Pipe
import numpy as np
import torch as th


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        # Make subprocesses for the envs
        # TODO: Add a delay when making sc2 envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        self.ps = [Process(target=env_worker, args=(worker_conn, CloudpickleWrapper(partial(env_fn, env_args=self.args.env_args))))
                            for worker_conn in self.worker_conns]

        for p in self.ps:
            p.daemon = True
            p.start()

        # TODO: Close stuff if appropriate

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        # TODO: Will have to add stuff to episode batch for envs that terminate at different times to ensure filled is correct
        self.t = 0

        self.t_env = 0

        # TODO: Fix env testing and stats
        self.test_rewards = []
        self.test_env_stats = []

        self.log_train_stats_t = -100000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        # TODO: Remove these if the runner doesn't need them
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def reset(self):
        self.batch = self.new_batch()
        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]

        while not all_terminated:

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    parent_conn.send(("step", actions[action_idx]))
                    action_idx += 1 # actions is not a list over every env

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "actions": [],
                "reward": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": []
            }

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # Remaining data for this current timestep
                    post_transition_data["actions"].append(data["actions"])
                    post_transition_data["reward"].append((data["reward"],))

                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.t_env += 1

                    env_terminated = False
                    if data["terminated"] and not data["ep_limit"]:
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    if not data["terminated"]:
                        pre_transition_data["state"].append(data["state"])
                        pre_transition_data["avail_actions"].append(data["avail_actions"])
                        pre_transition_data["obs"].append(data["obs"])

            # Actions are a tensor, need to stack them
            # TODO: Make a bit cleaner
            post_transition_data["actions"] = th.stack(post_transition_data["actions"], dim=0).unsqueeze(1)

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Update terminated envs after adding post_transition_data
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            all_terminated = all(terminated)

            if not all_terminated:
                self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats",None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        # TODO: Sort out sc2/env stats logging
        if test_mode:
            self.test_rewards.extend(episode_returns)
            self.test_env_stats.extend(env_stats)
            n_test_runs = max(1, self.args.test_nepisode // self.batch_size)
            if len(self.test_rewards) == n_test_runs:
                self.logger.log_stat("mean_test_return", np.mean(self.test_rewards), self.t_env)
                self.logger.log_stat("std_test_return", np.std(self.test_rewards), self.t_env)
                for test_return in self.test_rewards:
                    self.logger.log_stat("test_return", test_return, self.t_env)

                # TODO: Move env stat aggregator out of environment
                self.parent_conns[0].send(("agg_stats", self.test_env_stats))
                aggregated_stats = self.parent_conns[0].recv()
                for k, v in aggregated_stats.items():
                    self.logger.log_stat("mean_test_{}".format(k), v, self.t_env)
                self.test_rewards = []
                self.test_env_stats = []
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self.logger.log_stat("mean_train_return", np.mean(episode_returns), self.t_env)
            self.logger.log_stat("std_train_return", np.std(episode_returns), self.t_env)
            self.logger.log_stat("mean_ep_length", np.mean(episode_lengths), self.t_env)

            # TODO: Move logging into the action selector for this stuff
            self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)

            # TODO: Move env stat aggregator out of environment
            self.parent_conns[0].send(("agg_stats", env_stats))
            aggregated_stats = self.parent_conns[0].recv()
            for k, v in aggregated_stats.items():
                self.logger.log_stat("mean_{}".format(k), v, self.t_env)

            self.log_train_stats_t = self.t_env

        return self.batch


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            reached_ep_limit = env_info.get("episode_limit", False)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                # Rest of the data for the current timestep
                "actions": actions,
                "reward": reward,
                "terminated": terminated,
                "ep_limit": reached_ep_limit
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        elif cmd == "agg_stats":
            agg_stats = env.get_agg_stats(data)
            remote.send(agg_stats)
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

