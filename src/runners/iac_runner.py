import numpy as np
from components.scheme import Scheme
import torch as th
from torch.distributions import Normal

from components.epsilon_schedules import FlatThenDecaySchedule
from runners import REGISTRY as r_REGISTRY

NStepRunner = r_REGISTRY["nstep"]

class IACRunner(NStepRunner):

    def _setup_data_scheme(self, data_scheme):
        self.data_scheme = Scheme([dict(name="observations",
                                        shape=(self.env_obs_size,),
                                        select_agent_ids=range(0, self.n_agents),
                                        dtype=np.float32,
                                        missing=np.nan,),
                                   dict(name="state",
                                        shape=(self.env_state_size,),
                                        dtype = np.float32,
                                        missing=np.nan,
                                        size=self.env_state_size),
                                   dict(name="actions",
                                        shape=(1,),
                                        select_agent_ids=range(0, self.n_agents),
                                        dtype=np.int32,
                                        missing=-1,),
                                   dict(name="avail_actions",
                                        shape=(self.n_actions,),
                                        select_agent_ids=range(0, self.n_agents),
                                        dtype=np.int32,
                                        missing=-1,),
                                   dict(name="reward",
                                        shape=(1,),
                                        dtype=np.float32,
                                        missing=np.nan),
                                   dict(name="agent_id",
                                        shape=(1,),
                                        dtype=np.int32,
                                        select_agent_ids=range(0, self.n_agents),
                                        missing=-1),
                                   dict(name="policies",
                                        shape=(self.n_actions,),
                                        select_agent_ids=range(0, self.n_agents),
                                        dtype=np.float32,
                                        missing=np.nan),
                                   dict(name="terminated",
                                        shape=(1,),
                                        dtype=np.bool,
                                        missing=False),
                                   dict(name="truncated",
                                        shape=(1,),
                                        dtype=np.bool,
                                        missing=False),
                                   dict(name="reset",
                                        shape=(1,),
                                        dtype=np.bool,
                                        missing=False),
                                   dict(name="iac_epsilons",
                                        scope="episode",
                                        shape=(1,),
                                        dtype=np.float32,
                                        missing=float("nan"))
                                  ]).agent_flatten()
        pass

    def _add_episode_stats(self, T_env):
        super()._add_episode_stats(T_env)

        test_suffix = "" if not self.test_mode else "_test"
        self._add_stat("policy_entropy",
                       self.episode_buffer.get_stat("policy_entropy"),
                       T_env=T_env,
                       suffix=test_suffix)
        return

    def reset(self):
        super().reset()

        # if no test_mode, calculate fresh set of epsilons/epsilon seeds and update epsilon variance
        if not self.test_mode:
            ttype = th.cuda.FloatTensor if self.episode_buffer.is_cuda else th.FloatTensor
            # calculate IAC_epsilon_schedule
            if not hasattr(self, "iac_epsilon_decay_schedule"):
                 self.iac_epsilon_decay_schedule = FlatThenDecaySchedule(start=self.args.iac_epsilon_start,
                                                                         finish=self.args.iac_epsilon_finish,
                                                                         time_length=self.args.iac_epsilon_time_length,
                                                                         decay=self.args.iac_epsilon_decay_mode)

            epsilons = ttype(self.batch_size, 1).fill_(self.iac_epsilon_decay_schedule.eval(self.T_env))
            self.episode_buffer.set_col(col="iac_epsilons",
                                        scope="episode",
                                        data=epsilons)
        pass

    def log(self, log_directly=True):
        log_str, log_dict = super().log(log_directly=False)
        if not self.test_mode:
            log_str += ", IAC_epsilon={:g}".format(self.iac_epsilon_decay_schedule.eval(self.T_env))
            self.logging_struct.py_logger.info("TRAIN RUNNER INFO: {}".format(log_str))
        else:
            self.logging_struct.py_logger.info("TEST RUNNER INFO: {}".format(log_str))
        return log_str, log_dict

    pass