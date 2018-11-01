from .actor_critic_learner import ActorCriticLearner as ACL
from controllers.paid_controller import PaidMAC
from components.episode_buffer import EpisodeBatch
import torch as th


class PaidLearner(ACL):
    """ Learns PaidMAC using Planning-As-Inference-Distillation of decentralised MARL policies."""
    def __init__(self, mac, scheme, logger, args):
        ACL.__init__(self, mac, scheme, logger, args)
        assert isinstance(mac, PaidMAC), "The MAC trained by PAID must be derived from PaidMAC."
        self.distillation_factor = getattr(args, "distillation_factor", 1.0)
        self.prior_factor = getattr(args, "prior_factor", self.distillation_factor)
        self.propagate_divergence = getattr(args, "propagate_divergence", True)

    def _compute_policy(self, batch, avail_actions, forward_fun):
        """ Computes the full policy with the given <forward_fun>. """
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = forward_fun(batch, t=t)
            mac_out.append(agent_outs)
        pi = th.stack(mac_out, dim=1)  # Concat over time
        # Mask out unavailable actions, renormalise (as in action selection)
        pi[avail_actions == 0] = 0
        pi = pi / pi.sum(dim=-1, keepdim=True)
        pi[avail_actions == 0] = 0
        return pi

    def _chosen_policy(self, policy, actions, mask):
        """ Selects the chosen <actions> from the full policy and masks unavailable actions with 1. """
        pi_taken = th.gather(policy, dim=3, index=actions).squeeze(3)
        pi_taken[mask == 0] = 1.0
        return pi_taken

    def _compute_baseline(self, batch, critic_rewards, terminated, actions, avail_actions, mask, policy, q_sa, v_s):
        """ Computes the baseline for the advantage. """
        if self.separate_baseline_critic:
            for _ in range(self.args.critic_train_reps):
                q_sa_baseline, v_s_baseline, critic_train_stats_baseline = \
                    self.critic_train_fn(self.baseline_critic, self.target_baseline_critic, self.baseline_critic_optimiser,
                                         batch, critic_rewards, terminated, actions, avail_actions, mask)
            if self.args.critic_baseline_fn == "coma":
                baseline = (q_sa_baseline * policy).sum(-1).detach()
            else:
                baseline = v_s_baseline
        else:
            if self.args.critic_baseline_fn == "coma":
                baseline = (q_sa * policy).sum(-1).detach()
            else:
                baseline = v_s
        return baseline

    def _train_advantage(self, batch, rewards, divergence, terminated, actions, avail_actions, mask, policy):
        """ Trains the critic and returns the <advantages> of the current batch. """
        # Train the critic critic_train_reps times
        mask = mask.clone()
        critic_rewards = rewards - (divergence if self.propagate_divergence else 0)
        for _ in range(self.args.critic_train_reps):
            q_sa, v_s, critic_train_stats = self.critic_train_fn(self.critic, self.target_critic, self.critic_optimiser,
                                                                 batch, critic_rewards, terminated, actions,
                                                                 avail_actions, mask)
        # Compute advantage
        baseline = self._compute_baseline(batch, critic_rewards, terminated, actions, avail_actions, mask, policy,
                                          q_sa, v_s)
        if self.critic.output_type == "q":
            q_sa = th.gather(q_sa, dim=3, index=actions).squeeze(3)
            if self.args.critic_q_fn == "coma" and self.args.coma_mean_q:
                q_sa = q_sa.mean(2, keepdim=True).expand(-1, -1, self.n_agents)
        q_sa = self.nstep_returns(rewards - divergence, mask, q_sa, self.args.q_nstep)
        advantages = (q_sa - baseline).detach().squeeze()
        # Return advantages
        return advantages, critic_train_stats

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        """ Trains both a centralised and a decentraslised policy using Planning-As-Inference-Distillation.
            Is based on ActorCriticLearner.train() """

        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]

        # No experiences to train on in this minibatch
        if mask.sum() == 0:
            self.logger.log_stat("Mask_Sum_Zero", 1, t_env)
            self.logger.console_logger.error("Actor Critic Learner: mask.sum() == 0 at t_env {}".format(t_env))
            return
        mask = mask.repeat(1, 1, self.n_agents)

        # Compute both policies' outputs and the corresponding logarithm for the given batch
        pi = self._compute_policy(batch, avail_actions, self.mac.forward)
        pi_decentral = self._compute_policy(batch, avail_actions, self.mac.decentral_forward)
        log_pi = th.log(self._chosen_policy(pi, actions, mask))
        log_pi_decentral = th.log(self._chosen_policy(pi_decentral, actions, mask))

        # Compute the divergence of prior and posterior
        divergence = self.distillation_factor * (log_pi - log_pi_decentral).sum(dim=-1, keepdim=True)
        divergence[mask[:, :, 0] == 0] = 0

        # Train the critic and compute the advantage out of the critic's Q-values and the baseline
        advantages, critic_train_stats = self._train_advantage(batch, rewards, divergence, terminated, actions,
                                                               avail_actions, mask, pi)

        # Calculate both policies' lossess with mask (negative of the maximised loss)
        pg_loss = - ((advantages * log_pi) * mask).sum() / mask.sum()
        prior_loss = - self.prior_factor * (log_pi_decentral * mask).sum() / mask.sum()

        # Optimise agents
        self.agent_optimiser.zero_grad()
        (pg_loss + prior_loss).backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        # Update target networks if necessary
        if (self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_step = self.critic_training_steps

        # Log learning statistics every now and then
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean", "target_mean"]:
                self.logger.log_stat(key, sum(critic_train_stats[key])/ts_logged, t_env)

            self.logger.log_stat("advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("prior_loss", prior_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm, t_env)
            self.logger.log_stat("pi_max", (pi.max(dim=-1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("abs_divergence", (divergence.abs() * mask[:, :, 0].unsqueeze(2)).sum().item()
                                 / mask[:, :, 0].sum().item(), t_env)
            self.log_stats_t = t_env
