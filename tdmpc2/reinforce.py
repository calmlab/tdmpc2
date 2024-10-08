import numpy as np
import torch
import torch.nn.functional as F
import importlib

from common import math
from common.scale import RunningScale
from common.world_model import SingleModel, SingleDiscreteModel, SinglePredictiveOneModel, SingleDiscretePredictiveModel

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
        
        self.td_agent = cfg.td_agent
        self.horizon = cfg.horizon
        
        self.model = SingleModel(cfg).to(self.device)
        self.optim = torch.optim.Adam([
			{'params': self.model._policy.parameters()}
		], lr=self.cfg.lr)
        self.gamma = cfg.discount_gamma
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
        policy = self.model._policy
            
        a = policy(obs)
        policy_action, mu, log_sigma = self._calculate_action(a, self.action_dim, eval_mode)
        action = policy_action.detach()
        return action.cpu(), (mu, log_sigma), torch.zeros(1, 1), torch.zeros(1, self.obs_dim)

        
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
            log_sigmas = a[:, action_dim:2*action_dim]
            sigmas = torch.exp(log_sigmas)
            eps = torch.randn_like(sigmas)
            actions = mus + sigmas * eps
            return actions, mus, log_sigmas
        
    
    def _calculate_loss_policy(self, rewards, log_probs, entropies):
        if len(rewards.shape) == 1:
            rewards = rewards.unsqueeze(0)
            log_probs = log_probs.unsqueeze(0)
            entropies = entropies.unsqueeze(0)

        batch_size = rewards.shape[0]
        seq_length = rewards.shape[1]
        
        # R을 배치 크기에 맞게 초기화 (batch_size, 1, 1)
        R = torch.zeros((batch_size, 1, 1), device=rewards.device)
        loss = 0
        
        for i in reversed(range(seq_length)):
            R = self.gamma * R + rewards[:, i].unsqueeze(-1).unsqueeze(-1)  # (batch_size, 1, 1)
            
            # log_probs[i]는 (batch_size, 1, 6) 형태
            # R은 (batch_size, 1, 1) 형태로, 자동으로 브로드캐스팅됨
            step_loss = -(log_probs[:, i] * R).sum(dim=(1, 2))  # (batch_size,)
            loss = loss + step_loss.mean()  # 배치에 대한 평균
        
        loss = loss / seq_length
        return loss


    # REINFORCE update
    def _update_p(self, optimizer, actions, rewards, mus, log_sigmas, retain_graph=False):
        log_probs = self.log_normal(actions, mus, log_sigmas)
        log_probs = torch.clamp(log_probs, min=-100, max=0)  # clamp log_probs to prevent instability
        entropies = self.entropy(log_sigmas)
        loss = self._calculate_loss_policy(rewards, log_probs, entropies)
        optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        # torch.nn.utils.clip_grad_norm_(self.model._policy, max_norm=1.0)
        optimizer.step()
        return loss

    # REINFORCE (ref. https://dilithjay.com/blog/reinforce-a-quick-introduction-with-code)
    def update_policy(self, tds, retain_graph=False):
        actions, rewards, mus, log_sigmas = self.get_policy_train_data(tds)
        loss = self._update_p(self.optim, actions, rewards, mus, log_sigmas, retain_graph=retain_graph)
        
        # Return training statistics
        return {
            "loss_p": loss,
        }
        
        
    def get_policy_train_data(self, _tds):
        if self.td_agent:
            idx = len(_tds) % self.horizon   # horizon으로 나누어 떨어지는 만큼만 데이터로 사용
            action_list = []
            reward_list = []
            mu_list = []
            log_sigma_list = []
            while idx < len(_tds):
                tds = _tds[idx:min(idx+self.horizon, len(_tds))]
                action_list.append(torch.cat([td['action'] for td in tds]).to(self.device))
                reward_list.append(torch.cat([td['reward'] for td in tds]))
                mu_list.append(torch.cat([td['mu'] for td in tds]).to(self.device))
                log_sigma_list.append(torch.cat([td['log_sigma'] for td in tds]).to(self.device))
                idx += self.horizon
            actions = torch.stack(action_list).permute(1, 0, 2, 3).to(self.device)
            rewards = torch.stack(reward_list).permute(1, 0).to(self.device)
            mus = torch.stack(mu_list).permute(1, 0, 2, 3).to(self.device)
            log_sigmas = torch.stack(log_sigma_list).permute(1, 0, 2, 3).to(self.device)
        else:
            actions = torch.cat([td['action'] for td in _tds]).to(self.device)
            rewards = torch.cat([td['reward'] for td in _tds]).to(self.device)
            mus = torch.cat([td['mu'] for td in _tds]).to(self.device)
            log_sigmas = torch.cat([td['log_sigma'] for td in _tds]).to(self.device)
            
        return actions, rewards, mus, log_sigmas
        
        
    
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



class ReinforceDiscreteAgent(ReinforceAgent):
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
        self.optim = torch.optim.Adam([
			{'params': self.model._encoder.parameters()},
            {'params': self.model._policy.parameters()}
		], lr=self.cfg.lr)
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
        
        z = encoder(obs)    
        a = policy(z)
        policy_action, mu, log_sigma = self._calculate_action(a, self.action_dim, eval_mode)
        action = policy_action.detach()
        return action.cpu(), (mu, log_sigma), torch.zeros(1, 1), torch.zeros(1, self.obs_dim)


class ReinforcePredictiveAgent(ReinforceAgent):
    def __init__(self, cfg):
        self.cfg = cfg
        self.domain, self.task = self.cfg.task.replace('-', '_').split('_', 1)
        self.domain_module = importlib.import_module(f'envs.tasks.{self.domain}')
        self.device = torch.device(cfg.device)
        self._get_action_obs_dims()
        cfg.obs_dim = self.obs_dim
        cfg.action_dim = self.action_dim
        
        self.model = SinglePredictiveOneModel(cfg).to(self.device)
        self.optim = torch.optim.Adam([
			{'params': self.model._policy.parameters()}
		], lr=self.cfg.lr)
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
            
        output = policy(obs)
        policy_action, mu, log_sigma = self._calculate_action(output, self.action_dim, eval_mode)
        pred = self._calculate_prediction(output, self.action_dim, self.obs_dim)
        action = policy_action.detach()
        return action.cpu(), (mu, log_sigma), torch.zeros(1, 1), pred
    
    
    def _calculate_prediction(self, a, action_dim, obs_dim):
        return a[:, 2*action_dim:2*action_dim+obs_dim]
    
    
    def update(self, tds):
        torch.autograd.set_detect_anomaly(True)
        train_metrics = {}
        train_metrics.update(self.update_model(tds))

        # Return training statistics
        return train_metrics
    
    
    def update_model(self, tds, retain_graph=False):
        actions = torch.cat([td['action'] for td in tds]).to(self.device)
        rewards = torch.cat([td['reward'] for td in tds])
        mus = torch.cat([td['mu'] for td in tds])
        log_sigmas = torch.cat([td['log_sigma'] for td in tds])
        next_obs = torch.cat([td['obs'] for td in tds]).to(self.device)
        preds = torch.cat([td['pred'] for td in tds])

        loss_p, loss_d = self._update(self.optim, actions, rewards, mus, log_sigmas, preds, next_obs, retain_graph=retain_graph)
        
        # Return training statistics
        return {
            "loss_p": loss_p,
            "loss_d": loss_d,
        }
        
        
    def _update(self, optimizer, actions, rewards, mus, log_sigmas, preds, next_obs, retain_graph=False):
        log_probs = self.log_normal(actions, mus, log_sigmas)
        log_probs = torch.clamp(log_probs, min=-100, max=0)  # clamp log_probs to prevent instability
        entropies = self.entropy(log_sigmas)
        loss_p = self._calculate_loss_policy(rewards, log_probs, entropies)
        loss_d = self._calculate_loss_dynamics(preds, next_obs)
        loss = loss_p + loss_d
        optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        # torch.nn.utils.clip_grad_norm_(self.model._policy, max_norm=1.0)
        optimizer.step()
        return loss_p, loss_d
    
    
    def _calculate_loss_dynamics(self, preds, next_obs):
        return F.mse_loss(preds, next_obs)
    
    
class ReinforceDiscretePredictiveAgent(ReinforceAgent):
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
        
        self.model = SingleDiscretePredictiveModel(cfg).to(self.device)
        self.optim_d = torch.optim.Adam([
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.enc_lr},
            {'params': self.model._dynamics.parameters()}
		], lr=self.cfg.lr)
        self.optim_p = torch.optim.Adam([
			{'params': self.model._policy.parameters()}
		], lr=self.cfg.lr)
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
        dynamics = self.model._dynamics
        policy = self.model._policy
        
        z = encoder(obs)  
        z_ = z.detach()
        p_output = policy(z_)
        policy_action, mu, log_sigma = self._calculate_action(p_output, self.action_dim, eval_mode)
        action = policy_action.detach()
        dynamics_input = torch.cat([z, action], dim=1)
        pred_z = dynamics(dynamics_input)
        return action.cpu(), (mu, log_sigma), torch.zeros(1, 1), pred_z
    
    
    def update(self, tds):
        torch.autograd.set_detect_anomaly(True)
        train_metrics = {}
        train_metrics.update(self.update_worldmodel(tds))
        train_metrics.update(self.update_policy(tds))
        
        # Return training statistics
        return train_metrics
    
    
    def update_worldmodel(self, tds, retain_graph=False):
        pred_zs, next_obs = self.get_worldmodel_train_data(tds)
        loss = self._update_d(self.optim_d, pred_zs, next_obs, retain_graph=retain_graph)
        
        # Return training statistics
        return {
            "loss_d": loss,
        }
        
        
    def update_policy(self, tds, retain_graph=False):
        actions, rewards, mus, log_sigmas = self.get_policy_train_data(tds)
        loss = self._update_p(self.optim_p, actions, rewards, mus, log_sigmas, retain_graph=retain_graph)
        
        # Return training statistics
        return {
            "loss_p": loss,
        }
        
        
    def _update_d(self, optimizer, pred_zs, next_obs, retain_graph=False):
        loss = self._calculate_loss_dynamics(pred_zs, next_obs)
        optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        # torch.nn.utils.clip_grad_norm_(self.model._policy, max_norm=1.0)
        optimizer.step()
        return loss
        
    
    def _calculate_loss_dynamics(self, pred_zs, next_obs):
        next_zs = self.model._encoder(next_obs)
        return F.mse_loss(pred_zs, next_zs.detach())
    
    
    def get_worldmodel_train_data(self, _tds):
        if self.td_agent:
            idx = len(_tds) % self.horizon   # horizon으로 나누어 떨어지는 만큼만 데이터로 사용
            pred_list = []
            obs_list = []
            while idx < len(_tds):
                tds = _tds[idx:min(idx+self.horizon, len(_tds))]
                pred_list.append(torch.cat([td['pred'] for td in tds]).to(self.device))
                obs_list.append(torch.cat([td['obs'].unsqueeze(0) for td in tds]).to(self.device))
                idx += self.horizon
            preds = torch.stack(pred_list).permute(1, 0, 2, 3).to(self.device)
            next_obs = torch.stack(obs_list).permute(1, 0, 2, 3).to(self.device)
        else:
            preds = torch.cat([td['pred'] for td in _tds]).to(self.device)
            next_obs = torch.cat([td['obs'] for td in _tds]).to(self.device)
            
        return preds, next_obs
    
    