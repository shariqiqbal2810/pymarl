# from utils.blitzz.runners import NStepRunner
# from utils.blitzz.replay_buffer import ContiguousReplayBuffer
# from utils.blitzz.scheme import Scheme
# from utils.blitzz.sequence_buffer import BatchEpisodeBuffer
# from utils.blitzz.learners.coma import COMALearner, COMAPolicyLoss, COMACriticLoss
from torch import nn
from utils.dict2namedtuple import convert

# from utils.blitzz.debug import to_np
# from utils.blitzz.transforms import _build_model_inputs

from models import REGISTRY as mo_REGISTRY

import datetime
import numpy as np
import torch as th
from torch.autograd import Variable

def test1():
    """
    Does the critic loss go to zero if if I keep the replay buffer sample fixed?
    """
    _args = dict(env="pred_prey",
                 n_agents=4,
                 t_max=1000000,
                 learner="coma",
                 env_args=dict(prey_movement="escape",
                               predator_prey_shape=(3,3),
                               nagent_capture_enabled=False),
                 tensorboard=True,
                 name="33pp_comatest_dqn_new",
                 target_critic_update_interval=200000*2000, # DEBUG # 200000*20
                 agent="basic",
                 agent_model="DQN",
                 observe=True,
                 observe_db=True,
                 action_selector="multinomial",
                 use_blitzz=True,
                 obs_last_action=True,
                 test_interval=5000,
                 epsilon_start=1.0,
                 epsilon_finish=0.05,
                 epsilon_time_length=100000,
                 epsilon_decay="exp",
                 test_nepisode=50,
                 batch_size=32,
                 batch_size_run=32,
                 n_critic_learner_reps=200
                 )

    # required args
    _required_args = dict(obs_epsilon=False,
                          obs_agent_id=True,
                          share_params=True,
                          use_cuda=True,
                          lr_critic=5e-4,
                          lr_agent=5e-4,
                          multiagent_controller="independent",
                          td_lambda=1.0,
                          gamma=0.99)

    level1_out = th.zeros(1,1,1,1)
    level2_out = th.zeros(1,1,1,1)
    level3_out = th.zeros(1,1,1,1)
    avail_actions_pair = th.zeros(1,1,1,1)
    actions = th.zeros(1,1,1,1)
    n_actions = 5
    n_agents = 5

    _args.update(_required_args)

    _args["target_update_interval"] = _args["batch_size_run"] * 500
    _args["env_args"] = {"predator_prey_shape":(3,3),
                         "prey_movement": "escape",
                         "nagent_capture_enabled":False}
    args = convert(_args)

    model_class = mo_REGISTRY["flounderl_agent"]

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            pass

    class tmp(model_class):
        def __init__(self, **kwargs):
            nn.Module.__init__(self)
            self.model_level1 = Model()
            self.models = {"level2_{}".format(0): Model(), "level3_{}".format(0): Model()}
            self.n_agents = n_agents
            self.n_actions = n_actions
            pass

    model = tmp()
    def level1_forward(*args, **kwargs):
        return level1_out, None, None, "a*bs*t*v"
    model.model_level1.forward = level1_forward

    def level2_forward(*args, **kwargs):
        return level2_out, None, None, "a*bs*t*v"
    model.models["level2_{}".format(0)].forward = level2_forward

    def level3_forward(*args, **kwargs):
        return level3_out, None, None, "a*bs*t*v"
    model.models["level3_{}".format(0)].forward = level3_forward

    tformat = {"level1":"a*bs*t*v", "level2":"a*bs*t*v", "level3": "a*bs*t*v"}
    inputs = {"level1":{"agent_input_level1": None},
              "level2": {"agent_input_level2": {"avail_actions_pair":avail_actions_pair}},
              "level3": {"agent_input_level3": None}}
    hidden_states = {"level1":None, "level2":None, "level3":None}
    ret = model(inputs=inputs, actions=actions, hidden_states=hidden_states, tformat=tformat, loss_fn=None)
    a = 5
    pass


def main():
    test1()

    pass

if __name__ == "__main__":
    main()