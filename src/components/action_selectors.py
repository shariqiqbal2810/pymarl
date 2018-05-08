import numpy as np
import torch as th
from torch.autograd import Variable
from torch.distributions import Categorical
from .transforms import _to_batch, _from_batch, _adim, _vdim, _bsdim

REGISTRY = {}

class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args
        self.output_type = "policies"
        pass

    def select_action(self, inputs, avail_actions, tformat, test_mode=False):
        assert tformat in ["a*bs*t*v"], "invalid format!"

        if isinstance(inputs["policies"], Variable):
            agent_policies = inputs["policies"].data.clone()
        else:
            agent_policies = inputs["policies"].clone()  # might not be necessary

        masked_policies, params, tformat = _to_batch(agent_policies * avail_actions, tformat)
        _samples = Categorical(masked_policies).sample().unsqueeze(1)
        samples = _from_batch(_samples, params, tformat)
        return samples, agent_policies, tformat

REGISTRY["multinomial"] = MultinomialActionSelector

class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args
        self.output_type = "qvalues"

    def _get_epsilons(self):
        assert False, "function _get_epsilon must be overwritten by user in runner!"
        pass

    def select_action(self, inputs, avail_actions, tformat, test_mode=False):
        assert tformat in ["a*bs*t*v"], "invalid format!"

        if isinstance(inputs["qvalues"], Variable):
            agent_qvalues = inputs["qvalues"].data.clone()
        else:
            agent_qvalues = inputs["qvalues"].clone() # might not be necessary

        # greedy action selection
        assert avail_actions.sum(dim=_vdim(tformat)).prod() > 0.0, \
            "at least one batch entry has no available action!"
        agent_qvalues[avail_actions == 0.0] = -float("inf") # should never be selected!

        masked_qvalues, params, tformat = _to_batch(agent_qvalues * avail_actions, tformat)
        _, _argmaxes = masked_qvalues.max(dim=1, keepdim=True)
        #_argmaxes.unsqueeze_(1)

        if not test_mode: # normal epsilon-greedy action selection
            epsilons, epsilons_tformat = self._get_epsilons()
            random_numbers = epsilons.clone().uniform_()
            _avail_actions, params, tformat = _to_batch(avail_actions, tformat)
            random_actions = Categorical(_avail_actions).sample().unsqueeze(1)
            epsilon_pos = (random_numbers < epsilons).repeat(agent_qvalues.shape[_adim(tformat)], 1) # sampling uniformly from actions available
            epsilon_pos = epsilon_pos[:random_actions.shape[0], :]
            _argmaxes[epsilon_pos] = random_actions[epsilon_pos]
            eps_argmaxes = _from_batch(_argmaxes, params, tformat)
            return eps_argmaxes, agent_qvalues, tformat
        else: # don't use epsilon!
            # sanity check: there always has to be at least one action available.
            argmaxes = _from_batch(_argmaxes, params, tformat)
            return argmaxes, agent_qvalues, tformat

REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector