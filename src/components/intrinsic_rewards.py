import torch as th
from functools import reduce


# noinspection PyUnresolvedReferences
class LinearVarianceReward:
    def __init__(self, args):
        # Parameters
        self.alpha = args.visit_alpha
        self.beta = args.visit_beta
        self.bias = args.visit_bias
        self.scale = args.visit_reward
        self.normalize = False
        # Observation counter
        self.num_observations = 0
        # Matrices (not initialized yet)
        self.correlation = None
        self.inverse = None

    def observe(self, obs: th.Tensor):
        # Reshape obs in matrix
        v, bs = obs.shape[-1], reduce(lambda x, y: x * y, obs.shape[0:-1], 1)
        obs = obs.clone().detach().resize_(bs, v)
        # Normalize the observations, if specified
        if self.normalize:
            obs /= th.norm(obs, p=2, dim=0, keepdim=True)
        # Initialize correlation matrix not defined, initialize with small eye
        if self.num_observations == 0:
            initialize_value = 1E-1
            self.correlation = th.eye(v) * initialize_value
            if obs.is_cuda:
                self.correlation = self.correlation.cuda()
        # Update correlation matrix
        self.correlation *= self.alpha * self.correlation
        self.correlation += self.beta * th.mm(obs.t(), obs)
        # Update inverse correlation matrix
        if self.num_observations % 100 == 0:
            # Regularly invert the matrix, e.g. if correlation matrix has just been created
            self.inverse = th.inverse(self.correlation)
        else:
            # Update the inverted matrix otherwise using the matrix inversion lemma
            obs_inv = th.mm(obs, self.inverse)
            reg_mat = th.mm(obs_inv, obs.t())
            reg_mat += self.alpha / self.beta * th.diag(obs.new_ones(bs))
            self.inverse -= th.mm(obs_inv.t(), th.mm(th.inverse(reg_mat), obs_inv))
            self.inverse /= self.alpha
        # Increase observation counter by number of given observations <bs>
        self.num_observations += bs

    def reward(self, obs: th.Tensor):
        uncertainty = 0.0
        if self.inverse is not None:
            # Reshape obs in matrix
            shape = obs.shape
            v, bs = shape[-1], reduce(lambda x, y: x * y, shape[0:-1], 1)
            obs = obs.clone().detach().resize_(bs, v)
            # Compute the approximate "posterior standard deviation" for all observations
            uncertainty = th.sqrt(th.sum(th.mm(obs, self.inverse) * obs, dim=1)).view(shape[:-1])
            # We are only interested in the largest uncertainty
            uncertainty, _ = th.min(uncertainty, dim=-1, keepdim=True)
        return self.scale * (uncertainty - self.bias)

    def cuda(self):
        if self.correlation is not None:
            self.correlation.cuda()
        if self.inverse is not None:
            self.inverse.cuda()
