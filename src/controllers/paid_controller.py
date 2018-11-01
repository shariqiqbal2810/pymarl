from .basic_controller import BasicMAC
import itertools
from copy import deepcopy
import torch as th


class PaidMAC (BasicMAC):
    """ A controller that implements the Planning-As-Inference-Distillation (PAID) of decentral MARL policies. """

    def __init__(self, scheme, groups, args):
        BasicMAC.__init__(self, scheme, groups, args)
        # Create a decetranlised MAC that will be distilled
        self.decentral_mac = BasicMAC(scheme, groups, args)
        # Define things specific to the PAID MAC
        pass

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, **kwargs):
        """ Action selection (runner). In test_mode, use the decentral_mac, otherwise use the central_mac (self). """
        use_critic = kwargs.get("use_critic", False)
        if test_mode or use_critic:
            return self.decentral_mac.select_actions(ep_batch, t_ep, t_env, bs, test_mode, **kwargs)
        else:
            return BasicMAC.select_actions(self, ep_batch, t_ep, t_env, bs, test_mode, **kwargs)

    def decentral_forward(self, ep_batch, t, test_mode=False):
        """ Returns the probabilities of the decentralised MAC for the given batch/time step. """
        return self.decentral_mac.forward(ep_batch, t, test_mode)

    def init_hidden(self, batch_size):
        """ Initialises both MACs' initial state. """
        BasicMAC.init_hidden(self, batch_size)
        self.decentral_mac.init_hidden(batch_size)

    def parameters(self):
        """ Returns a generator over the parameters of both MACs."""
        params = BasicMAC.parameters(self)
        return itertools.chain(params, self.decentral_mac.parameters())

    def load_state(self, other_mac):
        BasicMAC.load_models(self, other_mac)
        self.decentral_mac.load_models(other_mac)

    def save_models(self, path):
        raise NotImplementedError

    def load_models(self, path):
        raise NotImplementedError

    def cuda(self):
        """ Moves both MACs to the GPU. """
        BasicMAC.cuda(self)
        self.decentral_mac.cuda()

    def post_episode(self, batch, test_mode):
        """ Empty interface to comply with ICQLParallelRunner. """
        pass

    def _build_inputs(self, batch, t):
        """ Overwrites the BasicMAC to build inputs that contain the state and the last action. """
        # Assumes homogeneous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["state"][:, t].unsqueeze(dim=1).repeat(1, self.n_agents, 1))  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        """ This controller is centralised, i.e., it is an RNN that takes the state and the previous action."""
        input_shape = scheme["state"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape
