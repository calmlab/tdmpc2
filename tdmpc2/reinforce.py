import numpy as np
import torch
import torch.nn.functional as F
import importlib

from common import math
from common.scale import RunningScale
from common.dual_world_model import DualModel, DualWorldModel
from common.world_model import SingleModel

# import pdb

class ReinforceAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.domain, self.task = self.cfg.task.replace('-', '_').split('_', 1)
        self.domain_module = importlib.import_module(f'envs.tasks.{self.domain}')
        self.device = torch.device(cfg.device)
        self._get_action_obs_dims()
        cfg.obs_dim = self.obs_dim
        cfg.action_dim = self.action_dim
        
        self.model = SingleModel(cfg).to(self.device)
        self.optim = torch.optim.Adam([
			{'params': self.model._brain.parameters()}
		], lr=self.cfg.lr)
        self.gamma = cfg.disconunt_gamma
        self.model.eval()
        

    def _get_action_obs_dims(self):
        def get_dim(desc_list):
            return len(desc_list)
        action_desc = self.domain_module.ACTION_DESCRIPTIONS
        obs_desc = self.domain_module.OBSERVATION_DESCRIPTIONS
        self.action_dim = get_dim(action_desc)
        self.obs_dim = get_dim(obs_desc)
        
    
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
            
        a = brain(obs)
        brain_action, mu, log_sigma = self._calculate_action(a, self.action_dim, eval_mode)
        action = brain_action.detach()
        return action.cpu(), (mu, log_sigma), torch.zeros(1, 1)

        
    def update(self, tds):
        torch.autograd.set_detect_anomaly(True)
        train_metrics = {}
        train_metrics.update(self.update_policy(tds))

        # Return training statistics
        return train_metrics


    # from https://github.com/chingyaoc/pytorch-REINFORCE/blob/master/reinforce_continuous.py
    def normal(self, x, mu, sigma_sq):
        pi = torch.tensor([math.pi], device=self.device)
        a = (-1*(x-mu).pow(2)/(2*sigma_sq)).exp()
        b = 1/(2*sigma_sq*pi).sqrt()
        return a*b
    
    
    def log_normal(self, x, mu, log_sigma):
        pi = torch.tensor([math.pi], device=self.device)
        sigma = torch.exp(log_sigma)
        return -((x - mu).pow(2) / (2 * sigma.pow(2))) - log_sigma - 0.5 * torch.log(2 * pi)
    
    
    def entropy(self, log_sigma):
        pi = torch.tensor([math.pi], device=self.device)
        sigma = torch.exp(log_sigma)
        return -(0.5*(sigma.pow(2)+2*pi.expand_as(log_sigma)).log()+0.5*math.e)
    
    
    def _calculate_action(self, a, action_dim, eval_mode):
        mus = a[:, :action_dim]
        if eval_mode:
            return mus, mus, torch.zeros_like(mus).to(self.device)
        else:
            log_sigmas = a[:, action_dim:]
            sigmas = torch.exp(log_sigmas)
            eps = torch.randn_like(sigmas)
            actions = mus + sigmas * eps
            return actions, mus, log_sigmas
        
    
    def _calculate_loss_policy(self, rewards, log_probs, entropies):
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)  # normalize the rewards
        R = torch.zeros_like(rewards[0])
        loss = 0
        for i in reversed(range(len(rewards))):
            R = self.gamma * R + rewards[i]
            loss = loss - (log_probs[i]*R).sum()# - (0.0001*entropies[i]).sum()
        loss = loss / len(rewards)
        return loss


    # REINFORCE update
    def _update_p(self, optimizer, actions, rewards, mus, log_sigmas, retain_graph=False):
        log_probs = self.log_normal(actions, mus, log_sigmas)
        log_probs = torch.clamp(log_probs, min=-100, max=0)  # clamp log_probs to prevent instability
        entropies = self.entropy(log_sigmas)
        loss = self._calculate_loss_policy(rewards, log_probs, entropies)
        optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        # torch.nn.utils.clip_grad_norm_(self.model._brain, max_norm=1.0)
        optimizer.step()
        return loss

    # REINFORCE (ref. https://dilithjay.com/blog/reinforce-a-quick-introduction-with-code)
    def update_policy(self, tds, retain_graph=False):
        actions = torch.cat([td['action'] for td in tds]).to(self.device)
        rewards = torch.cat([td['reward'] for td in tds])
        mus = torch.cat([td['mu'] for td in tds])
        log_sigmas = torch.cat([td['log_sigma'] for td in tds])

        loss = self._update_p(self.optim, actions, rewards, mus, log_sigmas, retain_graph=retain_graph)
        
        # Return training statistics
        return {
            "loss_p": loss,
        }
        




""" 아래는 변증법 """


class DialecticBase:
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
        self.act_individually = cfg.act_individually
        

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

    def act(self, obs, t0=False, eval_mode=False, task=None):
        raise NotImplementedError

    def update(self, tds_l, tds_r):
        raise NotImplementedError
    

class DialecticReinforceAgent(DialecticBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.model = DualModel(cfg).to(self.device)
        self.optim_l = torch.optim.Adam([{'params': self.model._brain_l.parameters()}], lr=self.cfg.lr)
        self.optim_r = torch.optim.Adam([{'params': self.model._brain_r.parameters()}], lr=self.cfg.lr)
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

        a_dist = brain(state)
        action, mu, log_sigma = self._calculate_action(a_dist, self.action_dim_l if self.brain_switch else self.action_dim_r, eval_mode)
        if self.brain_switch:  # left brain
            self.last_action_l = action
        else:  # right brain
            self.last_action_r = action
        if self.act_individually:
            action_indiv = torch.concat([self.last_action_l, self.last_action_r], dim=1).detach()
            return action_indiv.cpu(), (mu, log_sigma), torch.zeros(1, 1), state, self.brain_switch
        else:
            return action.cpu(), (mu, log_sigma), torch.zeros(1, 1), state, self.brain_switch

        
    def update(self, tds_l, tds_r):
        torch.autograd.set_detect_anomaly(True)
        train_metrics = {}
        train_metrics.update(self.update_policy_l(tds_l))
        train_metrics.update(self.update_policy_r(tds_r))

        # Return training statistics
        return train_metrics


    # from https://github.com/chingyaoc/pytorch-REINFORCE/blob/master/reinforce_continuous.py
    def normal(self, x, mu, sigma_sq):
        pi = torch.tensor([math.pi], device=self.device)
        a = (-1*(x-mu).pow(2)/(2*sigma_sq)).exp()
        b = 1/(2*sigma_sq*pi).sqrt()
        return a*b


    def log_normal(self, x, mu, log_sigma):
        pi = torch.tensor([math.pi], device=self.device)
        sigma = torch.exp(log_sigma)
        return -((x - mu).pow(2) / (2 * sigma.pow(2))) - log_sigma - 0.5 * torch.log(2 * pi)
    
    
    def entropy(self, log_sigma):
        pi = torch.tensor([math.pi], device=self.device)
        sigma = torch.exp(log_sigma)
        return -(0.5*(sigma.pow(2)+2*pi.expand_as(log_sigma)).log()+0.5*math.e)
    
    
    def _calculate_action(self, a_dist, action_dim, eval_mode):
        mus = a_dist[:, :action_dim]
        if eval_mode:
            # deterministic
            actions = mus
            return actions, mus, torch.zeros_like(mus).to(self.device)
        else:
            # stochastic
            log_sigmas = a_dist[:, action_dim:]
            sigmas = torch.exp(log_sigmas)
            eps = torch.randn_like(sigmas)
            actions = mus + sigmas * eps
            return actions, mus, log_sigmas
        
    
    def _calculate_loss_policy(self, rewards, log_probs, entropies):
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)  # normalize the rewards
        R = torch.zeros_like(rewards[0])
        loss = 0
        for i in reversed(range(len(rewards))):
            R = self.gamma * R + rewards[i]
            loss = loss - (log_probs[i]*R).sum()# - (0.0001*entropies[i]).sum()
        loss = loss / len(rewards)
        return loss


    # REINFORCE update
    def _update_p(self, optimizer, actions, rewards, mus, log_sigmas, retain_graph=False):
        log_probs = self.log_normal(actions, mus, log_sigmas)
        log_probs = torch.clamp(log_probs, min=-100, max=0)  # clamp log_probs to prevent instability
        entropies = self.entropy(log_sigmas)
        loss = self._calculate_loss_policy(rewards, log_probs, entropies)
        optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        # torch.nn.utils.clip_grad_norm_(self.model._brain, max_norm=1.0)
        optimizer.step()
        return loss


    # REINFORCE (ref. https://dilithjay.com/blog/reinforce-a-quick-introduction-with-code)
    def update_policy_l(self, tds_l, retain_graph=False):
        actions = torch.cat([td['action'] for td in tds_l]).to(self.device)
        rewards = torch.cat([td['reward'] for td in tds_l])
        mus = torch.cat([td['mu'] for td in tds_l])
        log_sigmas = torch.cat([td['log_sigma'] for td in tds_l])
        actions_l = actions[:, :, self.action_filter_l]

        loss = self._update_p(self.optim_l, actions_l, rewards, mus, log_sigmas, retain_graph=retain_graph)
        
        # Return training statistics
        return {
            "loss_pi_l": loss,
        }
        
    def update_policy_r(self, tds_r, retain_graph=False):
        actions = torch.cat([td['action'] for td in tds_r]).to(self.device)
        rewards = torch.cat([td['reward'] for td in tds_r])
        mus = torch.cat([td['mu'] for td in tds_r])
        log_sigmas = torch.cat([td['log_sigma'] for td in tds_r])
        actions_r = actions[:, :, self.action_filter_r]
        
        loss = self._update_p(self.optim_r, actions_r, rewards, mus, log_sigmas, retain_graph=retain_graph)
        
        # Return training statistics
        return {
            "loss_pi_r": loss,
        }

