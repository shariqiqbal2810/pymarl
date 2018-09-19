from .q_learner import QLearner
from components.episode_buffer import EpisodeBatch
from torch.optim import RMSprop
import torch as th


# noinspection PyUnresolvedReferences
class ICQLLearner(QLearner):
    """ Trains ICQL agents, which consist of IQL agents and a central COMA critic of their greedy policy. """
    def __init__(self, mac, scheme, logger, args):
        QLearner.__init__(self, mac, scheme, logger, args)
        self.critic_optimiser = None
        if self.args.separate_critic_optimisation:
            self.critic_params = list(self.mac.critic.parameters())
            self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.lr, alpha=args.optim_alpha,
                                            eps=args.optim_eps)

    def td_lambda_target(self, reward, value, terminated):
        # Assumes <reward> in B*T-1*1, <value> in B*T*A and <terminated> in B*T-1*1
        # Also assumes all Tensors are filled with 0/nan after the episode terminates
        td_lambda = self.args.td_lambda
        gamma = self.args.gamma
        # Create a mask for currently running (i.e. not terminated) episodes
        mask = 1 - th.sum(terminated, dim=1)
        # Initialise last lambda-return for currently running episodes
        ret = th.zeros(*value.shape)
        ret[:, -1] = value[:, -1] * mask
        # Backwards recursive update of the "forward view"
        for t in range(ret.shape[1] - 2, -1, -1):
            # Update the mask of currently running episodes
            mask = mask + terminated[:, t]
            # Recursive update of the lambda-return of running episodes
            ret[:, t] = td_lambda * gamma * ret[:, t + 1] \
                        + mask * (reward[:, t] + (1 - td_lambda) * gamma * value[:, t + 1])
        # Returns lambda-return in B*T-1*A, i.e. from t=0 to t=T-1
        return ret[:, 0:-1]

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        """ One training step with the given <batch>. """
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999  # From OG deepmarl

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out[avail_actions == 0] = -9999999
            cur_max_actions = mac_out[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # ---------------------------------- ICQL Critic loss ----------------------------------------------------------
        # Compute the chosen Q-values of the current critic
        critic_out = self.mac.critic(batch).view_as(mac_out)
        chosen_critic_qvals = th.gather(critic_out[:, :-1], dim=3, index=actions).squeeze(3)

        # Select the greedy actions of the individual agents and change the batch accordingly
        greedy_actions = self.mac.greedy_actions(batch, avail_actions=avail_actions, agent_outputs=mac_out)
        greedy_batch = self.mac.change_actions(batch, greedy_actions)

        # Compute the Q-values of the target critic with greedy_actions
        target_critic_out = self.target_mac.critic(greedy_batch).view_as(mac_out)
        target_critic_pol = th.gather(target_critic_out, dim=3, index=greedy_actions).squeeze(3)

        # Compute the loss function of the critic and add it to the IQL loss computed above
        if self.args.td_lambda == 0.0:
            critic_target = rewards + self.args.gamma * (1 - terminated) * target_critic_pol[:, 1:]
        else:
            critic_target = self.td_lambda_target(rewards, target_critic_pol, terminated)
        critic_td_error = chosen_critic_qvals - critic_target.detach()
        critic_loss = ((critic_td_error * mask) ** 2).sum() / mask.sum()

        # Either add the loss to the optimiser or optimise it independently
        if self.critic_optimiser is None:
            loss = loss + critic_loss
        else:
            self.critic_optimiser.zero_grad()
            critic_loss.backward()
            th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
            self.critic_optimiser.step()

        # ---------------------------------- Finish (from IQL) ---------------------------------------------------------

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("critic_loss", critic_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("critic_q_taken_mean",
                                 (chosen_critic_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env
