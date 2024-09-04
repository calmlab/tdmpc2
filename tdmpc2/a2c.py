import numpy as np
import torch
import torch.nn.functional as F
import importlib

from common import math
from common.scale import RunningScale
# from common.dual_world_model import DualModel, DualWorldModel
from common.world_model import SingleModel
from common.dual_world_model import DualModel
from reinforce import ReinforceAgent, DialecticBase, DialecticReinforceAgent

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
        
        self.model = SingleModel(cfg).to(self.device)
        self.optim_p = torch.optim.Adam([
			{'params': self.model._brain.parameters()}
		], lr=self.cfg.lr)
        self.optim_v = torch.optim.Adam([
			{'params': self.model._value.parameters()}
		], lr=self.cfg.lr)
        self.gamma = cfg.disconunt_gamma
        self.model.eval()
        

    # def _get_action_obs_dims(self):
    #     def get_dim(desc_list):
    #         return len(desc_list)
    #     action_desc = self.domain_module.ACTION_DESCRIPTIONS
    #     obs_desc = self.domain_module.OBSERVATION_DESCRIPTIONS
    #     self.action_dim = get_dim(action_desc)
    #     self.obs_dim = get_dim(obs_desc)
        
    
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
        brain = self.model._brain
        value_fn = self.model._value
            
        a = brain(obs)
        v = value_fn(obs)
        brain_action, mu, log_sigma = self._calculate_action(a, self.action_dim, eval_mode)
        action = brain_action.detach()
        return action.cpu(), (mu, log_sigma), v

        
    def update(self, tds):
        torch.autograd.set_detect_anomaly(True)
        train_metrics = {}
        target_qs = self._calculate_target_q(tds)
        advantages = self._calculate_advantages(tds, target_qs)
        train_metrics.update(self.update_policy(tds, advantages))
        train_metrics.update(self.update_value(tds, target_qs))

        # Return training statistics
        return train_metrics


    # from https://github.com/chingyaoc/pytorch-REINFORCE/blob/master/reinforce_continuous.py
    # def normal(self, x, mu, sigma_sq):
    #     pi = torch.tensor([math.pi], device=self.device)
    #     a = (-1*(x-mu).pow(2)/(2*sigma_sq)).exp()
    #     b = 1/(2*sigma_sq*pi).sqrt()
    #     return a*b
    
    
    # def entropy(self, sigma_sq):
    #     pi = torch.tensor([math.pi], device=self.device)
    #     return -(0.5*(sigma_sq+2*pi.expand_as(sigma_sq)).log()+0.5*math.e)
    
    
    def _calculate_target_q(self, tds):#rewards, next_values, dones):
        next_obs = torch.cat([td['obs'] for td in tds]).to(self.device)
        rewards = torch.cat([td['reward'] for td in tds]).to(self.device)
        dones = torch.cat([td['done'] for td in tds]).to(self.device)
        # print('Reinforce: next_obs: ' , next_obs.shape)
        next_v = self.model._value(next_obs).squeeze(1)
        # print('Reinforce: next_v: ' , next_v.shape)
        target_q = rewards + self.gamma * next_v * (1 - dones)
        return target_q#.to(self.device)
    
    
    def _calculate_advantages(self, tds, target_qs):
        values = torch.cat([td['value'] for td in tds]).squeeze(1).squeeze(1)
        advantages = target_qs - values
        return advantages
    
    
    def _calculate_action(self, a, action_dim, eval_mode):
        mus = a[:, :action_dim]
        if eval_mode:
            actions = mus
            return actions, mus, torch.zeros_like(mus).to(self.device)
        else:
            log_sigmas = a[:, action_dim:]
            sigmas = torch.exp(log_sigmas)
            eps = torch.randn_like(sigmas)
            actions = mus + sigmas * eps
            return actions, mus, log_sigmas

    
    def _calculate_loss_value(self, values, target_qs):
        return F.mse_loss(values, target_qs.detach())


    # # REINFORCE (ref. https://dilithjay.com/blog/reinforce-a-quick-introduction-with-code)
    def update_policy(self, tds, advantages, retain_graph=False):
        actions = torch.cat([td['action'] for td in tds]).to(self.device)
        mus = torch.cat([td['mu'] for td in tds])
        log_sigmas = torch.cat([td['log_sigma'] for td in tds])

        loss = self._update_p(self.optim_p, actions, advantages.detach(), mus, log_sigmas, retain_graph=retain_graph)
        
        # Return training statistics
        return {
            "loss_p": loss,
        }
        
    
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
        


class DialecticA2CAgent(DialecticReinforceAgent):
    def __init__(self, cfg):
        super(DialecticReinforceAgent, self).__init__(cfg)
        
        self.model = DualModel(cfg).to(self.device)
        self.optim_pi_l = torch.optim.Adam([{'params': self.model._brain_l.parameters()}], lr=self.cfg.lr)
        self.optim_pi_r = torch.optim.Adam([{'params': self.model._brain_r.parameters()}], lr=self.cfg.lr)
        self.optim_v_l = torch.optim.Adam([{'params': self.model._value_l.parameters()}], lr=self.cfg.lr)
        self.optim_v_r = torch.optim.Adam([{'params': self.model._value_r.parameters()}], lr=self.cfg.lr)
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
            last_opposite_action = self.last_action_r.detach()
            obs = obs[self.obs_filter_l].to(self.device, non_blocking=True).unsqueeze(0)
            brain = self.model._brain_l
            value_fn = self.model._value_l
        else:  # right brain
            last_opposite_action = self.last_action_l.detach()
            obs = obs[self.obs_filter_r].to(self.device, non_blocking=True).unsqueeze(0)
            brain = self.model._brain_r
            value_fn = self.model._value_r
            
        state = torch.concat([obs, last_opposite_action], dim=1)

        a_dist = brain(state)
        v = value_fn(state)
        action, mu, log_sigma = self._calculate_action(a_dist, self.action_dim_l if self.brain_switch else self.action_dim_r, eval_mode)
        if self.brain_switch:                      # left brain
            self.last_action_l = action
        else:                                      # right brain
            self.last_action_r = action
        if self.act_individually:
            action_indiv = torch.concat([self.last_action_l, self.last_action_r], dim=1).detach()
            return action_indiv.cpu(), (mu, log_sigma), v, self.brain_switch, state.squeeze(0)
        else:
            return action.cpu().detach(), (mu, log_sigma), v, self.brain_switch, state.squeeze(0)

        
    def update(self, tds_l, tds_r):
        # torch.autograd.set_detect_anomaly(True)

        target_qs_l = self._calculate_target_q_l(tds_l)
        advantages_l = self._calculate_advantages(tds_l, target_qs_l)
        target_qs_r = self._calculate_target_q_r(tds_r)
        advantages_r = self._calculate_advantages(tds_r, target_qs_r)

        train_metrics = {}
        train_metrics.update(self.update_policy_l(tds_l, advantages_l))
        train_metrics.update(self.update_policy_r(tds_r, advantages_r))
        train_metrics.update(self.update_value_l(tds_l, target_qs_l, retain_graph=True))
        train_metrics.update(self.update_value_r(tds_r, target_qs_r, retain_graph=True))

        # Return training statistics
        return train_metrics


    # from https://github.com/chingyaoc/pytorch-REINFORCE/blob/master/reinforce_continuous.py
    def _calculate_target_q_l(self, tds_l):
        next_obs = torch.cat([td['obs'] for td in tds_l]).to(self.device)
        rewards = torch.cat([td['reward'] for td in tds_l]).to(self.device)
        dones = torch.cat([td['done'] for td in tds_l]).to(self.device)
        next_v = self.model._value_l(next_obs).squeeze(1)
        target_q = rewards + self.gamma * next_v * (1 - dones)
        return target_q
    

    def _calculate_target_q_r(self, tds_r):
        next_obs = torch.cat([td['obs'] for td in tds_r]).to(self.device)
        rewards = torch.cat([td['reward'] for td in tds_r]).to(self.device)
        dones = torch.cat([td['done'] for td in tds_r]).to(self.device)
        next_v = self.model._value_l(next_obs).squeeze(1)
        target_q = rewards + self.gamma * next_v * (1 - dones)
        return target_q
    
    
    def _calculate_advantages(self, tds, target_qs):
        values = torch.cat([td['value'] for td in tds]).squeeze(1).squeeze(1)
        # print('fdasa: target_qs' , target_qs.shape)
        advantages = target_qs - values
        return advantages

    

    def _calculate_action(self, a_dist, action_dim, eval_mode):
        mus = a_dist[:, :action_dim]
        if eval_mode:
            return mus, mus, torch.zeros_like(mus).to(self.device)
        else:
            log_sigmas = a_dist[:, action_dim:]
            sigmas = torch.exp(log_sigmas)
            eps = torch.randn_like(sigmas)
            actions = mus + sigmas * eps
            return actions, mus, log_sigmas
    
    
    def _calculate_loss_value(self, values, target_qs):
        return F.mse_loss(values, target_qs.detach())


    # REINFORCE (ref. https://dilithjay.com/blog/reinforce-a-quick-introduction-with-code)
    def update_policy_l(self, tds_l, advantages_l, retain_graph=False):
        actions = torch.cat([td['action'] for td in tds_l]).to(self.device)
        mus = torch.cat([td['mu'] for td in tds_l])
        log_sigmas = torch.cat([td['log_sigma'] for td in tds_l])
        actions_l = actions[:, :, self.action_filter_l]
        loss = self._update_p(self.optim_pi_l, actions_l, advantages_l.detach(), mus, log_sigmas, retain_graph=retain_graph)
        
        # Return training statistics
        return {
            "loss_p_l": loss,
        }
    
    def update_policy_r(self, tds_r, advantages_r, retain_graph=False):
        actions = torch.cat([td['action'] for td in tds_r]).to(self.device)
        mus = torch.cat([td['mu'] for td in tds_r])
        log_sigmas = torch.cat([td['log_sigma'] for td in tds_r])
        actions_r = actions[:, :, self.action_filter_r]

        loss = self._update_p(self.optim_pi_r, actions_r, advantages_r.detach(), mus, log_sigmas, retain_graph=retain_graph)
        
        # Return training statistics
        return {
            "loss_p_r": loss,
        }
        
    def _update_v(self, optimizer, values, target_qs, retain_graph=False):
        loss = self._calculate_loss_value(values, target_qs)
        optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        # torch.nn.utils.clip_grad_norm_(self.model._value, max_norm=1.0)
        optimizer.step()
        return loss
    

    def update_value_l(self, tds_l, target_qs_l, retain_graph=False):
        values = torch.cat([td['value'] for td in tds_l]).squeeze(1).squeeze(1)

        loss = self._update_v(self.optim_v_l, values, target_qs_l, retain_graph=retain_graph)
        
        # Return training statistics
        return {
            "loss_v_l": loss,
        }
    

    def update_value_r(self, tds_r, target_qs_r, retain_graph=False):
        values = torch.cat([td['value'] for td in tds_r]).squeeze(1).squeeze(1)

        loss = self._update_v(self.optim_v_r, values, target_qs_r, retain_graph=retain_graph)
        
        # Return training statistics
        return {
            "loss_v_r": loss,
        }