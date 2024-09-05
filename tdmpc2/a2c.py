import numpy as np
import torch
import torch.nn.functional as F
import importlib

from common import math
from common.scale import RunningScale
from common.world_model import SingleModel, SingleDiscreteModel, SingleOneModel
from reinforce import ReinforceAgent

# import pdb

class A2CAgent(ReinforceAgent):
    def __init__(self, cfg):
        self.cfg = cfg
        self.domain, self.task = self.cfg.task.replace('-', '_').split('_', 1)
        self.domain_module = importlib.import_module(f'envs.tasks.{self.domain}')
        self.device = torch.device(cfg.device)
        self._get_action_obs_dims()
        cfg.obs_dim = self.obs_dim
        cfg.action_dim = self.action_dim
        
        self.td_agent = cfg.td_agent
        self.horizon = cfg.horizon
        
        self.model = SingleModel(cfg).to(self.device)
        self.optim_p = torch.optim.Adam([
			{'params': self.model._policy.parameters()}
		], lr=self.cfg.lr)
        self.optim_v = torch.optim.Adam([
			{'params': self.model._value.parameters()}
		], lr=self.cfg.value_lr)
        self.gamma = cfg.discount_gamma
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
        obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        policy = self.model._policy
        value_func = self.model._value
            
        a = policy(obs)
        v = value_func(obs)
        policy_action, mu, log_sigma = self._calculate_action(a, self.action_dim, eval_mode)
        action = policy_action.detach()
        return action.cpu(), (mu, log_sigma), v, torch.zeros(1, self.obs_dim)

        
    def update(self, tds):
        torch.autograd.set_detect_anomaly(True)
        train_metrics = {}
        target_qs = self._calculate_target_q(tds)
        advantages = self._calculate_advantages(tds, target_qs)
        train_metrics.update(self.update_policy(tds, advantages))
        train_metrics.update(self.update_value(tds, target_qs))

        # Return training statistics
        return train_metrics
    
    
    def _calculate_target_q(self, tds):#rewards, next_values, dones):
        next_obs = torch.cat([td['obs'] for td in tds]).to(self.device)
        rewards = torch.cat([td['reward'] for td in tds]).to(self.device)
        dones = torch.cat([td['done'] for td in tds]).to(self.device)
        next_v = self.model._value(next_obs).squeeze(1)
        target_q = rewards + self.gamma * next_v * (1 - dones)
        return target_q
    
    
    def _calculate_advantages(self, tds, target_qs):
        values = torch.cat([td['value'] for td in tds]).squeeze(1).squeeze(1)
        advantages = target_qs - values
        return advantages
        
    
    def _calculate_loss_value(self, values, target_qs):
        return F.mse_loss(values, target_qs.detach())


    # # REINFORCE (ref. https://dilithjay.com/blog/reinforce-a-quick-introduction-with-code)
    def update_policy(self, tds, advantages, retain_graph=False):
        # actions = torch.cat([td['action'] for td in tds]).to(self.device)
        actions, _, mus, log_sigmas, advantages = self.get_policy_train_data(tds, advantages)
        loss = self._update_p(self.optim_p, actions, advantages.detach(), mus, log_sigmas, retain_graph=retain_graph)
        
        # Return training statistics
        return {
            "loss_p": loss,
        }
        
        
    def get_policy_train_data(self, _tds, advantages):
        if self.td_agent:
            idx = len(_tds) % self.horizon   # horizon으로 나누어 떨어지는 만큼만 데이터로 사용
            action_list = []
            reward_list = []
            mu_list = []
            log_sigma_list = []
            advantage_list = []
            while idx < len(_tds):
                tds = _tds[idx:min(idx+self.horizon, len(_tds))]
                action_list.append(torch.cat([td['action'] for td in tds]))
                reward_list.append(torch.cat([td['reward'] for td in tds]))
                mu_list.append(torch.cat([td['mu'] for td in tds]))
                log_sigma_list.append(torch.cat([td['log_sigma'] for td in tds]))
                advantage_list.append(advantages[idx:min(idx+self.horizon, len(advantages))])
                idx += self.horizon
            actions = torch.stack(action_list).permute(1, 0, 2, 3).to(self.device)
            rewards = torch.stack(reward_list).permute(1, 0).to(self.device)
            mus = torch.stack(mu_list).permute(1, 0, 2, 3).to(self.device)
            log_sigmas = torch.stack(log_sigma_list).permute(1, 0, 2, 3).to(self.device)
            advantages = torch.stack(advantage_list).permute(1, 0).to(self.device)
        else:
            actions = torch.cat([td['action'] for td in _tds]).to(self.device)
            rewards = torch.cat([td['reward'] for td in _tds]).to(self.device)
            mus = torch.cat([td['mu'] for td in _tds]).to(self.device)
            log_sigmas = torch.cat([td['log_sigma'] for td in _tds]).to(self.device)
            
        return actions, rewards, mus, log_sigmas, advantages
        
    
    def _update_v(self, optimizer, values, target_qs, retain_graph=False):
        loss = self._calculate_loss_value(values, target_qs)
        optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        # torch.nn.utils.clip_grad_norm_(self.model._value, max_norm=1.0)
        optimizer.step()
        return loss
        
        
    def update_value(self, tds, target_qs, retain_graph=False):
        values = torch.cat([td['value'] for td in tds]).squeeze(1).squeeze(1)

        loss = self._update_v(self.optim_v, values, target_qs, retain_graph=retain_graph)
        
        # Return training statistics
        return {
            "loss_v": loss,
        }


class A2CDiscreteAgent(A2CAgent):
    def __init__(self, cfg):
        self.cfg = cfg
        self.domain, self.task = self.cfg.task.replace('-', '_').split('_', 1)
        self.domain_module = importlib.import_module(f'envs.tasks.{self.domain}')
        self.device = torch.device(cfg.device)
        self._get_action_obs_dims()
        cfg.obs_dim = self.obs_dim
        cfg.action_dim = self.action_dim
        
        self.td_agent = cfg.td_agent
        self.horizon = cfg.horizon
        
        self.model = SingleDiscreteModel(cfg).to(self.device)
        self.optim_p = torch.optim.Adam([
			{'params': self.model._encoder.parameters()},
            {'params': self.model._policy.parameters()}
		], lr=self.cfg.lr)
        self.optim_v = torch.optim.Adam([
			{'params': self.model._value.parameters()}
		], lr=self.cfg.value_lr)
        self.gamma = cfg.discount_gamma
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
        obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        encoder = self.model._encoder
        policy = self.model._policy
        value_func = self.model._value
        
        z = encoder(obs)
        a = policy(z)
        v = value_func(z.detach())
        policy_action, mu, log_sigma = self._calculate_action(a, self.action_dim, eval_mode)
        action = policy_action.detach()
        return action.cpu(), (mu, log_sigma), v, torch.zeros(1, self.obs_dim)
    
    
    def _calculate_target_q(self, tds):
        next_obs = torch.cat([td['obs'] for td in tds]).to(self.device)
        rewards = torch.cat([td['reward'] for td in tds]).to(self.device)
        dones = torch.cat([td['done'] for td in tds]).to(self.device)
        next_z = self.model._encoder(next_obs)
        next_v = self.model._value(next_z).squeeze(1)
        target_q = rewards + self.gamma * next_v * (1 - dones)
        return target_q
    
    
# Actor와 Critic의 lr이 달라야 하므로, 애초에 One Model은 불가능함        
# class A2COneModelAgent(A2CAgent):
#     def __init__(self, cfg):
#         self.cfg = cfg
#         self.domain, self.task = self.cfg.task.replace('-', '_').split('_', 1)
#         self.domain_module = importlib.import_module(f'envs.tasks.{self.domain}')
#         self.device = torch.device(cfg.device)
#         self._get_action_obs_dims()
#         cfg.obs_dim = self.obs_dim
#         cfg.action_dim = self.action_dim
        
#         self.model = SingleOneModel(cfg).to(self.device)
#         self.optim = torch.optim.Adam([
#             {'params': self.model._policy.parameters()}
#         ], lr=self.cfg.lr)
#         self.gamma = cfg.discount_gamma
#         self.model.eval()
        
        
#     def act(self, obs, t0=False, eval_mode=False, task=None):
#         """
#         Select an action by planning in the latent space of the world model.

#         Args:
#             obs (torch.Tensor): Observation from the environment.
#             t0 (bool): Whether this is the first observation in the episode.
#             eval_mode (bool): Whether to use the mean of the action distribution.
#             task (int): Task index (only used for multi-task experiments).

#         Returns:
#             torch.Tensor: Action to take in the environment.
#         """
#         obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
#         policy = self.model._policy
            
        # output = policy(obs)
        # policy_action, mu, log_sigma = self._calculate_action(output, self.action_dim, eval_mode)
        # v = self._calculate_value(output, self.action_dim)
        # action = policy_action.detach()
        # return action.cpu(), (mu, log_sigma), v
    
    
#     def _calculate_value(self, a, action_dim):
#         return a[:, 2*action_dim:]
    
    
#     def update(self, tds):
#         torch.autograd.set_detect_anomaly(True)
#         train_metrics = {}
#         target_qs = self._calculate_target_q(tds)
#         advantages = self._calculate_advantages(tds, target_qs)
#         train_metrics.update(self.update_model(tds, target_qs, advantages))

#         # Return training statistics
#         return train_metrics


#     def update_model(self, tds, target_qs, advantages, retain_graph=False):
#         actions = torch.cat([td['action'] for td in tds]).to(self.device)
#         mus = torch.cat([td['mu'] for td in tds])
#         log_sigmas = torch.cat([td['log_sigma'] for td in tds])
#         values = torch.cat([td['value'] for td in tds]).squeeze(1).squeeze(1)

#         loss_p, loss_v = self._update(self.optim, actions, advantages.detach(), values, target_qs, mus, log_sigmas, retain_graph=retain_graph)
        
#         # Return training statistics
#         return {
#             "loss_p": loss_p,
#             "loss_v": loss_v,
#         }
        
        
#     def _update(self, optimizer, actions, rewards, values, target_qs, mus, log_sigmas, retain_graph=False):
#         log_probs = self.log_normal(actions, mus, log_sigmas)
#         log_probs = torch.clamp(log_probs, min=-100, max=0)  # clamp log_probs to prevent instability
#         entropies = self.entropy(log_sigmas)
#         loss_p = self._calculate_loss_policy(rewards, log_probs, entropies)
#         loss_v = self._calculate_loss_value(values, target_qs)
#         loss = loss_p + loss_v
#         optimizer.zero_grad()
#         loss.backward(retain_graph=retain_graph)
#         # torch.nn.utils.clip_grad_norm_(self.model._policy, max_norm=1.0)
#         optimizer.step()
#         return loss_p, loss_v
    
    
#     def _calculate_target_q(self, tds):#rewards, next_values, dones):
#         next_obs = torch.cat([td['obs'] for td in tds]).to(self.device)
#         rewards = torch.cat([td['reward'] for td in tds]).to(self.device)
#         dones = torch.cat([td['done'] for td in tds]).to(self.device)
#         next_v = self.model._policy(next_obs)[:, 2*self.action_dim:].squeeze(1)
#         target_q = rewards + self.gamma * next_v * (1 - dones)
#         return target_q#.to(self.device)