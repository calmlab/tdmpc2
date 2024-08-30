import numpy as np
import torch
import torch.nn.functional as F
import importlib

from common import math
from common.scale import RunningScale
from common.dual_world_model import DualModel, DualWorldModel

# import pdb

class DialecticAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.domain, self.task = self.cfg.task.replace('-', '_').split('_', 1)
        self.domain_module = importlib.import_module(f'envs.tasks.{self.domain}')
        self.device = torch.device(cfg.device)
        self._get_action_obs_dims()
        self.reset()
        cfg.obs_dim_l = self.obs_dim_l
        cfg.obs_dim_r = self.obs_dim_r
        cfg.action_dim_l = self.action_dim_l
        cfg.action_dim_r = self.action_dim_r
        

    def _get_action_obs_dims(self):
        def get_dim(desc_list, actor_key):
            dim = 0
            for desc in desc_list:
                if desc[actor_key]:
                    dim += 1
            return dim
        action_desc = self.domain_module.ACTION_DESCRIPTIONS
        obs_desc = self.domain_module.OBSERVATION_DESCRIPTIONS
        self.action_dim_l = get_dim(action_desc, 'left_actor')
        self.action_dim_r = get_dim(action_desc, 'right_actor')
        self.obs_dim_l = get_dim(obs_desc, 'left_actor')
        self.obs_dim_r = get_dim(obs_desc, 'right_actor')
        self.action_filter_l = [a['left_actor'] for a in action_desc]
        self.action_filter_r = [a['right_actor'] for a in action_desc]
        self.obs_filter_l = [a['left_actor'] for a in obs_desc]
        self.obs_filter_r = [a['right_actor'] for a in obs_desc]
        
    
    def reset(self):
        self.last_action_l = torch.zeros(self.action_dim_l, device=self.device).unsqueeze(0)
        self.last_action_r = torch.zeros(self.action_dim_r, device=self.device).unsqueeze(0)
        self.brain_switch = False


class DialecticMPC(DialecticAgent):
    """
    DialecticMPC agent. Implements training + inference.
    Can be used for both single-task and multi-task experiments,
    and supports both state and pixel observations.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        self.model = DualWorldModel(cfg).to(self.device)
        self.optim = torch.optim.Adam([
            {'params': self.model._encoder_l.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
            {'params': self.model._encoder_r.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
            {'params': self.model._dynamics_l.parameters()},
            {'params': self.model._dynamics_r.parameters()},
            {'params': self.model._reward.parameters()},
            {'params': self.model._Qs.parameters()},
            {'params': self.model._task_emb.parameters() if self.cfg.multitask else []}
        ], lr=self.cfg.lr)
        self.pi_optim = torch.optim.Adam([
            {'params': self.model._pi_l.parameters()},
            {'params': self.model._pi_r.parameters()}
        ], lr=self.cfg.lr, eps=1e-5)
        self.model.eval()
        self.scale = RunningScale(cfg)
        self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
        self.discount = torch.tensor(
            [self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device=cfg.device
        ) if self.cfg.multitask else self._get_discount(cfg.episode_length)


    def _get_discount(self, episode_length):
        """
        Returns discount factor for a given episode length.
        Simple heuristic that scales discount linearly with episode length.
        Default values should work well for most tasks, but can be changed as needed.

        Args:
            episode_length (int): Length of the episode. Assumes episodes are of fixed length.

        Returns:
            float: Discount factor for the task.
        """
        frac = episode_length/self.cfg.discount_denom
        return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

    def save(self, fp):
        """
        Save state dict of the agent to filepath.
        
        Args:
            fp (str): Filepath to save state dict to.
        """
        torch.save({"model": self.model.state_dict()}, fp)

    def load(self, fp):
        """
        Load a saved state dict from filepath (or dictionary) into current agent.
        
        Args:
            fp (str or dict): Filepath or state dict to load.
        """
        state_dict = fp if isinstance(fp, dict) else torch.load(fp)
        self.model.load_state_dict(state_dict["model"])

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        """
        Select an action by planning in the latent space of the world model.
        
        Args:
            obs (torch.Tensor): Observation from the environment.
            t0 (bool): Whether this is the first observation in the episode.
            eval_mode (bool): Whether to use the mean of the action distribution.
            task (int): Task index (only used for multi-task experiments).
        
        Returns:
            torch.Tensor: Action to take in the environment.
        """
        obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        if task is not None:
            task = torch.tensor([task], device=self.device)
        z = self.model.encode(obs, task)
        if self.cfg.mpc:
            a = self.plan(z, t0=t0, eval_mode=eval_mode, task=task)
        else:
            a = self.model.pi(z, task)[int(not eval_mode)][0]
        return a.cpu()
    
    @torch.no_grad()
    def rand_act(self, env):
        self.brain_switch = not self.brain_switch
        randact = env.rand_act()
        if self.brain_switch:  # left brain
            self.last_action_l = randact[self.action_filter_l].cuda()
        else:
            self.last_action_r = randact[self.action_filter_r].cuda()
            
        action = torch.concat([self.last_action_l, self.last_action_r], dim=0)
        return action.cpu(), self.brain_switch

    @torch.no_grad()
    def _estimate_value(self, z, actions, task):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        for t in range(self.cfg.horizon):
            reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
            z = self.model.next(z, actions[t], task)
            G += discount * reward
            discount *= self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
        return G + discount * self.model.Q(z, self.model.pi(z, task)[1], task, return_type='avg')

    @torch.no_grad()
    def plan(self, z, t0=False, eval_mode=False, task=None):
        """
        Plan a sequence of actions using the learned world model.
        
        Args:
            z (torch.Tensor): Latent state from which to plan.
            t0 (bool): Whether this is the first observation in the episode.
            eval_mode (bool): Whether to use the mean of the action distribution.
            task (Torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            torch.Tensor: Action to take in the environment.
        """		
        # Sample policy trajectories
        if self.cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
            _z = z.repeat(self.cfg.num_pi_trajs, 1)
            for t in range(self.cfg.horizon-1):
                pi_actions[t] = self.model.pi(_z, task)[1]
                _z = self.model.next(_z, pi_actions[t], task)
            pi_actions[-1] = self.model.pi(_z, task)[1]

        # Initialize state and parameters
        z = z.repeat(self.cfg.num_samples, 1)
        mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        std = self.cfg.max_std*torch.ones(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        if not t0:
            mean[:-1] = self._prev_mean[1:]
        actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
        if self.cfg.num_pi_trajs > 0:
            actions[:, :self.cfg.num_pi_trajs] = pi_actions

        # Iterate MPPI
        for _ in range(self.cfg.iterations):

            # Sample actions
            actions[:, self.cfg.num_pi_trajs:] = (mean.unsqueeze(1) + std.unsqueeze(1) * \
                torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)) \
                .clamp(-1, 1)
            if self.cfg.multitask:
                actions = actions * self.model._action_masks[task]

            # Compute elite actions
            value = self._estimate_value(z, actions, task).nan_to_num_(0)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.cfg.temperature*(elite_value - max_value))
            score /= score.sum(0)
            mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9)) \
                .clamp_(self.cfg.min_std, self.cfg.max_std)
            if self.cfg.multitask:
                mean = mean * self.model._action_masks[task]
                std = std * self.model._action_masks[task]

        # Select action
        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = mean
        a, std = actions[0], std[0]
        if not eval_mode:
            a += std * torch.randn(self.cfg.action_dim, device=std.device)
        return a.clamp_(-1, 1)
        
    def update_pi(self, zs, task):
        """
        Update policy using a sequence of latent states.
        
        Args:
            zs (torch.Tensor): Sequence of latent states.
            task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            float: Loss of the policy update.
        """
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_q_grad(False)
        _, pis, log_pis, _ = self.model.pi(zs, task)
        qs = self.model.Q(zs, pis, task, return_type='avg')
        self.scale.update(qs[0])
        qs = self.scale(qs)

        # Loss is a weighted sum of Q-values
        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1,2)) * rho).mean()
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
        self.pi_optim.step()
        self.model.track_q_grad(True)

        return pi_loss.item()


    @torch.no_grad()
    def _td_target(self, next_z, reward, task):
        """
        Compute the TD-target from a reward and the observation at the following time step.
        
        Args:
            next_z (torch.Tensor): Latent state at the following time step.
            reward (torch.Tensor): Reward at the current time step.
            task (torch.Tensor): Task index (only used for multi-task experiments).
        
        Returns:
            torch.Tensor: TD-target.
        """
        pi = self.model.pi(next_z, task)[1]
        discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
        return reward + discount * self.model.Q(next_z, pi, task, return_type='min', target=True)


    def update(self, buffer_l, buffer_r):
        """
        Main update function. Corresponds to one iteration of model learning.
        
        Args:
            buffer (common.buffer.Buffer): Replay buffer.
        
        Returns:
            dict: Dictionary of training statistics.
        """
        obs, action, reward, task = buffer.sample()

        # Compute targets
        with torch.no_grad():
            next_z = self.model.encode(obs[1:], task)
            td_targets = self._td_target(next_z, reward, task)

        # Prepare for update
        self.optim.zero_grad(set_to_none=True)
        self.model.train()

        # Latent rollout
        zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
        z = self.model.encode(obs[0], task)
        zs[0] = z
        consistency_loss = 0
        for t in range(self.cfg.horizon):
            z = self.model.next(z, action[t], task)
            consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho**t
            zs[t+1] = z

        # Predictions
        _zs = zs[:-1]
        qs = self.model.Q(_zs, action, task, return_type='all')
        reward_preds = self.model.reward(_zs, action, task)
        
        # Compute losses
        reward_loss, value_loss = 0, 0
        for t in range(self.cfg.horizon):
            reward_loss += math.soft_ce(reward_preds[t], reward[t], self.cfg).mean() * self.cfg.rho**t
            for q in range(self.cfg.num_q):
                value_loss += math.soft_ce(qs[q][t], td_targets[t], self.cfg).mean() * self.cfg.rho**t
        consistency_loss *= (1/self.cfg.horizon)
        reward_loss *= (1/self.cfg.horizon)
        value_loss *= (1/(self.cfg.horizon * self.cfg.num_q))
        total_loss = (
            self.cfg.consistency_coef * consistency_loss +
            self.cfg.reward_coef * reward_loss +
            self.cfg.value_coef * value_loss
        )

        # Update model
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.optim.step()

        # Update policy
        pi_loss = self.update_pi(zs.detach(), task)

        # Update target Q-functions
        self.model.soft_update_target_Q()

        # Return training statistics
        self.model.eval()
        return {
            "consistency_loss": float(consistency_loss.mean().item()),
            "reward_loss": float(reward_loss.mean().item()),
            "value_loss": float(value_loss.mean().item()),
            "pi_loss": pi_loss,
            "total_loss": float(total_loss.mean().item()),
            "grad_norm": float(grad_norm),
            "pi_scale": float(self.scale.value),
        }

    def update_l(self, buffer):
        obs, action, reward, task = buffer.sample()
        obs_l = obs[:, :, self.obs_filter_l]
        obs_r = obs[:, :, self.obs_filter_r]
        action_l = action[:, :, self.action_filter_l]
        action_r = action[:, :, self.action_filter_r]
        
        obs_p = obs_l[:-1]
        inputs = torch.cat([obs_p, action_r], dim=2)
        
        self.batch_update(self.model._brain_l, inputs, self.action_dim_l)
        
        # Return training statistics
        self.model.eval()
        return {
            "loss_l": float(consistency_loss.mean().item()),
            "reward_loss": float(reward_loss.mean().item()),
            "value_loss": float(value_loss.mean().item()),
            "pi_loss": pi_loss,
            "total_loss": float(total_loss.mean().item()),
            "grad_norm": float(grad_norm),
            "pi_scale": float(self.scale.value),
        }


class DialecticImitation(DialecticAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.model = DualModel(cfg).to(self.device)
        self.optim_l = torch.optim.Adam([
			{'params': self.model._brain_l.parameters()}
		], lr=self.cfg.lr)
        self.optim_r = torch.optim.Adam([
			{'params': self.model._brain_r.parameters()}
		], lr=self.cfg.lr)
        self.gamma = cfg.disconunt_gamma
        self.model.eval()
    
    
    def act(self, obs, t0=False, eval_mode=False, task=None):
        """
        Select an action by planning in the latent space of the world model.

        Args:
            obs (torch.Tensor): Observation from the environment.
            t0 (bool): Whether this is the first observation in the episode.
            eval_mode (bool): Whether to use the mean of the action distribution.
            task (int): Task index (only used for multi-task experiments).

        Returns:
            torch.Tensor: Action to take in the environment.
        """
        self.brain_switch = not self.brain_switch
        if self.brain_switch:  # left brain
            obs = obs[self.obs_filter_l].to(self.device, non_blocking=True).unsqueeze(0)
            brain = self.model._brain_l
            last_opposite_action = self.last_action_r.detach()
        else:  # right brain
            obs = obs[self.obs_filter_r].to(self.device, non_blocking=True).unsqueeze(0)
            brain = self.model._brain_r
            last_opposite_action = self.last_action_l.detach()
            
        state = torch.concat([obs, last_opposite_action], dim=1)
        #assert(not state.requires_grad)
        a = brain(state)
        # if not eval_mode:
        #     assert(a.requires_grad)
        brain_action, mu, sigma_sq = self._calculate_action(a, self.action_dim_l if self.brain_switch else self.action_dim_r, eval_mode)
        if self.brain_switch:  # left brain
            self.last_action_l = brain_action
        else:  # right brain
            self.last_action_r = brain_action
        action = torch.concat([self.last_action_l, self.last_action_r], dim=1)
        # if not eval_mode:
        #     assert(action.requires_grad)
        return action.cpu(), self.brain_switch, (mu, sigma_sq)

        
    def update(self, tds_l, tds_r):
        train_metrics = {}
        # train_metrics.update(self.update_l(tds_l))
        train_metrics.update(self.update_r(tds_r))

        # Return training statistics
        return train_metrics


    # from https://github.com/chingyaoc/pytorch-REINFORCE/blob/master/reinforce_continuous.py
    def normal(self, x, mu, sigma_sq):
        pi = torch.tensor([math.pi], device=self.device)
        a = (-1*(x-mu).pow(2)/(2*sigma_sq)).exp()
        b = 1/(2*sigma_sq*pi).sqrt()
        return a*b
    
    
    def entropy(self, sigma_sq):
        pi = torch.tensor([math.pi], device=self.device)
        return -(0.5*(sigma_sq+2*pi.expand_as(sigma_sq)).log()+0.5*math.e)
    
    
    def _calculate_action(self, a, action_dim, eval_mode):
        mus = a[:, :action_dim]
        if eval_mode:
            return mus, mus, torch.zeros_like(mus).to(self.device)
        else:
            sigma_sqs = F.softplus(a[:, action_dim:])
            sigmas = sigma_sqs.sqrt()
            eps = torch.randn(mus.size()).to(self.device)
            actions = mus + sigmas * eps
            return actions, mus, sigma_sqs
        
    
    def _calculate_loss(self, rewards, log_probs, entropies):
        R = torch.zeros_like(rewards[0])
        loss = 0
        for i in reversed(range(len(rewards))):
            R = self.gamma * R + rewards[i]
            loss = loss - (log_probs[i]*R).sum()# - (0.0001*entropies[i]).sum()
        loss = loss / len(rewards)
        return loss


    # REINFORCE update
    def _update(self, brain, optimizer, actions, rewards, mus, sigma_sqs):
        probs = self.normal(actions, mus, sigma_sqs)
        log_probs = probs.log()
        entropies = self.entropy(sigma_sqs)
        loss = self._calculate_loss(rewards, log_probs, entropies)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    # REINFORCE (ref. https://dilithjay.com/blog/reinforce-a-quick-introduction-with-code)
    def update_l(self, tds_l):
        actions = torch.cat([td['action'] for td in tds_l]).to(self.device)
        rewards = torch.cat([td['reward'] for td in tds_l])
        mus = torch.cat([td['mu'] for td in tds_l])
        sigma_sqs = torch.cat([td['sigma_sq'] for td in tds_l])
        actions_l = actions[:, :, self.action_filter_l]

        loss = self._update(self.model._brain_l, self.optim_l, actions_l, rewards, mus, sigma_sqs)
        
        # Return training statistics
        return {
            "loss_l": loss,
        }
        
    def update_r(self, tds_r):
        actions = torch.cat([td['action'] for td in tds_r]).to(self.device)
        rewards = torch.cat([td['reward'] for td in tds_r])
        mus = torch.cat([td['mu'] for td in tds_r])
        sigma_sqs = torch.cat([td['sigma_sq'] for td in tds_r])
        actions_r = actions[:, :, self.action_filter_r]
        
        loss = self._update(self.model._brain_r, self.optim_r, actions_r, rewards, mus, sigma_sqs)
        
        # Return training statistics
        return {
            "loss_r": loss,
        }
        
        
        