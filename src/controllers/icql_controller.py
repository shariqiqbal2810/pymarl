from .basic_controller import BasicMAC
from modules.critics.coma import COMACritic
import itertools
from copy import deepcopy
import torch as th


# noinspection PyUnresolvedReferences
class ICQLMAC(BasicMAC):
    """ A basic MAC with an additional COMA critic for ICQL. """

    def __init__(self, scheme, groups, args):
        BasicMAC.__init__(self, scheme, groups, args)
        self.critic = COMACritic(scheme, args)
        self.one = th.cuda.FloatTensor(1).fill_(1.0) if args.use_cuda else th.FloatTensor(1).fill_(1.0)

    def parameters(self):
        """ Returns a generator of the parameters of this MAC"""
        return itertools.chain(BasicMAC.parameters(self), self.critic.parameters())

    def load_state(self, other_mac):
        """ Copies the parameters of another ICQLMAC into this one. """
        assert isinstance(other_mac, ICQLMAC), "To load the state of one MAC into another, both must be icql_mac."
        BasicMAC.load_state(self, other_mac)
        self.critic.load_state_dict(other_mac.critic.state_dict())

    def cuda(self):
        """ Moves the MAC to the GPU. """
        BasicMAC.cuda(self)
        self.critic.cuda()
        self.one.cuda()

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, use_critic=False, **kwargs):
        """ Returns actions that the MAC selects for the given batch/time-step.
            <use_critic> determines whether the Q-values of the critic (True) or the IQL agents (False) are used. """
        # Use either the IQL agents from the BasicMAC for sampling ...
        if not use_critic:
            return BasicMAC.select_actions(self, ep_batch, t_ep, t_env, bs, test_mode, **kwargs)
        # ... or use the critic conditioned on greedy actions of the IQL agents for sampling
        avail_actions = ep_batch["avail_actions"][bs, t_ep]
        greedy_actions = self.greedy_actions(ep_batch, t=t_ep, bs=bs, avail_actions=avail_actions)
        greedy_batch = self.change_actions(batch=ep_batch, actions=greedy_actions, t=t_ep, bs=bs)
        critic_outputs = self.critic(greedy_batch, t=t_ep).unsqueeze(0)  # keep bs dimension for action_selector
        return self.action_selector.select_action(critic_outputs[bs], avail_actions, t_env, test_mode=test_mode)

    def greedy_actions(self, ep_batch, t=None, bs=slice(None), avail_actions=None, agent_outputs=None):
        """ Returns the actions that are greedy w.r.t. the (optional) IQL agents' <agent_output>. """
        avail_actions = ep_batch["avail_actions"][bs, t] if avail_actions is None else avail_actions
        agent_outputs = self.forward(ep_batch, t) if agent_outputs is None else agent_outputs
        agent_outputs[avail_actions == 0] = -9999999  # remove unavailable actions
        return agent_outputs.max(dim=len(agent_outputs.shape)-1, keepdim=True)[1]  # max over last dimension

    def change_actions(self, batch, actions: th.Tensor, t=None, bs=None):
        """ Returns a copied EpisodeBatch <batch> with the specified actions at time step <t> and batch slice <bs>. """
        # Copy batch and make sure t and bs refer to the entire range if not specified
        batch = deepcopy(batch)
        bs = slice(0, actions.shape[0]) if bs is None else bs
        t = slice(0, actions.shape[1]) if t is None else t
        # Copy actions into batch
        if 'actions' in batch.data.transition_data:
            batch.data.transition_data['actions'][bs, t] = actions
        # Generate one-hot actions in the batch
        if 'actions_onehot' in batch.data.transition_data:
            one_hot = batch.data.transition_data['actions_onehot'][bs, t]
            one_hot.fill_(0.0)
            one_hot.scatter_(dim=len(one_hot.shape)-1, index=actions, src=self.one.expand_as(actions))
        # Return the altered batch
        return batch
