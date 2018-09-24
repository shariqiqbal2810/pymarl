from .q_learner import QLearner
from components.episode_buffer import EpisodeBatch
from torch.optim import RMSprop
import torch as th
from utils.rl_utils import build_td_lambda_targets


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
        mac_hidden = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            # Compute agent output
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            # Remember the agents' hidden states
            hidden_shape = self.mac.hidden_states.shape
            mac_hidden.append(self.mac.hidden_states.view(hidden_shape[0] // self.args.n_agents, -1, hidden_shape[1]))
        # Concat both over time
        mac_out = th.stack(mac_out, dim=1)
        mac_hidden = th.stack(mac_hidden, dim=1)

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

        # Compute potential intrinsic reward
        intrinsic_rewards = 0
        if self.args.visit_reward > 0:
            intrinsic_rewards = self.mac.intrinsic_agents.reward(mac_hidden[:, 1:])

        # Compute the loss function of the critic and add it to the IQL loss computed above
        if self.args.td_lambda == 0.0:
            # This is the technically correct 1-step TD(0)-return
            critic_target = rewards + intrinsic_rewards + self.args.gamma * (1 - terminated) * target_critic_pol[:, 1:]
        else:
            # But to propagate intrinsic rewards we may want to use a TD(lambda)-return instead
            critic_target = build_td_lambda_targets(rewards + intrinsic_rewards, terminated, mask, target_critic_pol,
                                                    self.args.n_agents, self.args.gamma, self.args.td_lambda)
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
            self.logger.log_stat("intrinsic_reward", intrinsic_rewards.mean().item(), t_env)
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
