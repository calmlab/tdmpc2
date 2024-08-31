import numpy as np
import torch
import torch.nn.functional as F
import importlib

from common import math
from common.scale import RunningScale
# from common.dual_world_model import DualModel, DualWorldModel
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
        brain_action, mu, sigma_sq = self._calculate_action(a, self.action_dim, eval_mode)
        action = brain_action.detach()
        return action.cpu(), (mu, sigma_sq)

        
    def update(self, tds):
        torch.autograd.set_detect_anomaly(True)
        train_metrics = {}
        train_metrics.update(self.update_(tds))

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
    def _update(self, brain, optimizer, actions, rewards, mus, sigma_sqs, retain_graph=False):
        probs = self.normal(actions, mus, sigma_sqs)
        log_probs = probs.log()
        entropies = self.entropy(sigma_sqs)
        loss = self._calculate_loss(rewards, log_probs, entropies)
        optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        optimizer.step()
        return loss

    # REINFORCE (ref. https://dilithjay.com/blog/reinforce-a-quick-introduction-with-code)
    def update_(self, tds, retain_graph=False):
        actions = torch.cat([td['action'] for td in tds]).to(self.device)
        rewards = torch.cat([td['reward'] for td in tds])
        mus = torch.cat([td['mu'] for td in tds])
        sigma_sqs = torch.cat([td['sigma_sq'] for td in tds])

        loss = self._update(self.model._brain, self.optim, actions, rewards, mus, sigma_sqs, retain_graph=retain_graph)
        
        # Return training statistics
        return {
            "loss_": loss,
        }
        
        