from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict

from trainer.base import Trainer


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		return dict(
			step=self._step,
			episode=self._ep_idx,
			total_time=time() - self._start_time,
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		ep_rewards, ep_successes = [], []
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, t = self.env.reset(), False, 0, 0
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			while not done:
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
				obs, reward, done, info = self.env.step(action)
				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_success=np.nanmean(ep_successes),
		)

	def to_td(self, obs, action=None, reward=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan'))
		td = TensorDict(dict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
		), batch_size=(1,))
		return td

	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next = {}, True, True
		while self._step <= self.cfg.steps:

			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True

			# Reset environment
			if done:
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

				if self._step > 0:
					train_metrics.update(
						episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
						episode_success=info['success'],
					)
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					self._ep_idx = self.buffer.add(torch.cat(self._tds))

				obs = self.env.reset()
				self._tds = [self.to_td(obs)]

			# Collect experience
			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=len(self._tds)==1)
			else:
				action = self.env.rand_act()
			obs, reward, done, info = self.env.step(action)
			self._tds.append(self.to_td(obs, action, reward))

			# Update agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = self.cfg.seed_steps
					print('Pretraining agent on seed data...')
				else:
					num_updates = 1
				for _ in range(num_updates):
					_train_metrics = self.agent.update(self.buffer)
				train_metrics.update(_train_metrics)

			self._step += 1
	
		self.logger.finish(self.agent)


class OnlineDialecticTrainer(OnlineTrainer):
    def __init__(self, cfg, env, agent, buffer_l, buffer_r, logger):
        self.buffer_l = buffer_l
        self.buffer_r = buffer_r
        super().__init__(cfg, env, agent, None, logger)

    
    def to_td(self, obs, action=None, reward=None):
        """Creates a TensorDict for a new episode."""
        if isinstance(obs, dict):
            obs = TensorDict(obs, batch_size=(), device='cpu')
        else:
            obs = obs.unsqueeze(0).cpu()
        if action is None:
            action = torch.full_like(self.env.rand_act(), float('nan'))
        if reward is None:
            reward = torch.tensor(float('nan'))
        td = TensorDict(dict(
            obs=obs,
            action=action,
            reward=reward,
        ), batch_size=(1,), device=self.device)
        return td


    def train(self):
        """Train a DialecticMPC agent."""
        train_metrics, done, eval_next = {}, True, True
        while self._step <= self.cfg.steps:

            # Evaluate agent periodically
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True

            # Reset environment
            if done:
                if eval_next:
                    eval_metrics = self.eval()
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(eval_metrics, 'eval')
                    eval_next = False

                if self._step > 0:
                    train_metrics.update(
                        episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
                        episode_success=info['success'],
                    )
                    train_metrics.update(self.common_metrics())
                    self.logger.log(train_metrics, 'train')
                    self._ep_idx = self.buffer_l.add(torch.cat(self._tds_l))
                    self._ep_idx = self.buffer_r.add(torch.cat(self._tds_r))

                
                obs = self.env.reset()
                self._tds_l = [self.to_td(obs)]
                self._tds_r = [self.to_td(obs)]
                self.agent.reset()

            # Collect experience
            if self._step > self.cfg.seed_steps:
                action, is_act_left = self.agent.act(obs, t0=len(self._tds)==1)
            else:
                action, is_act_left = self.agent.rand_act(self.env)
                
            obs, reward, done, info = self.env.step(action)
            
            if is_act_left:
                self._tds_l.append(self.to_td(obs, action, reward))
            else:
                self._tds_r.append(self.to_td(obs, action, reward))

            # Update agent
            if self._step >= self.cfg.seed_steps:
                if self._step == self.cfg.seed_steps:
                    num_updates = self.cfg.seed_steps
                    print('Pretraining agent on seed data...')
                else:
                    num_updates = 1
                for _ in range(num_updates):
                    _train_metrics = self.agent.update(self.buffer)
                train_metrics.update(_train_metrics)

            self._step += 1

        self.logger.finish(self.agent)
    
class OnlineDialecticImitationTrainer(OnlineTrainer):
    def __init__(self, cfg, env, agent, logger):
        super().__init__(cfg, env, agent, None, logger)
        
        
    def to_td(self, obs, action=None, reward=None, mu=None, log_sigma=None, value=None, done=None):
        """Creates a TensorDict for a new episode."""
        if isinstance(obs, dict):
            obs = TensorDict(obs, batch_size=(), device='cpu')
        else:
            obs = obs.unsqueeze(0).cpu()
        if action is None:
            action = torch.full_like(self.env.rand_act(), float('nan'))
        if reward is None:
            reward = torch.tensor(float('nan'))
        if mu is None:
            mu = torch.full_like(self.env.rand_act(), float('nan'))
        if log_sigma is None:
            log_sigma = torch.full_like(self.env.rand_act(), float('nan'))
        if value is None:
            value = torch.tensor(float('nan'))
        if done is None:
            done = torch.tensor(float('nan'))
        td = TensorDict(dict(
            obs=obs,
            action=action.unsqueeze(0),
            reward=reward.unsqueeze(0),
            mu=mu.unsqueeze(0),
            log_sigma=log_sigma.unsqueeze(0),
            value=value.unsqueeze(0),
            done=done.unsqueeze(0),
        ), batch_size=(1,))
        return td
    
    
    @torch.no_grad()
    def eval(self):
        """Evaluate a TD-MPC2 agent."""
        ep_rewards, ep_successes = [], []
        for i in range(self.cfg.eval_episodes):
            obs, done, ep_reward, t = self.env.reset(), False, 0, 0
            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=(i==0))
            while not done:
                if self.cfg.act_individually:
                    action, _, _, _, _ = self.agent.act(obs, t0=t==0, eval_mode=True)
                else:
                    action_l, _, _, _, _ = self.agent.act(obs, t0=t==0, eval_mode=True)
                    action_r, _, _, _, _ = self.agent.act(obs, t0=t==0, eval_mode=True)
                    action = torch.concat([action_l, action_r], dim=1).detach()
                obs, reward, done, info = self.env.step(action)
                ep_reward += reward
                t += 1
                if self.cfg.save_video:
                    self.logger.video.record(self.env)
                
            ep_rewards.append(ep_reward)
            ep_successes.append(info['success'])
            if self.cfg.save_video:
                self.logger.video.save(self._step)
        return dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_success=np.nanmean(ep_successes),
        )
        
    
    def train(self):
        """Train a DialecticImitation agent."""
        train_metrics, done, eval_next = {}, True, True
        data_count = 0
        self._tds_l = []
        self._tds_r = []
        while self._step <= self.cfg.steps:

            # Evaluate agent periodically
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True

            # On Episode End 
            if done:# or data_count >= 2*self.cfg.batch_size:
                if eval_next:
                    eval_metrics = self.eval()
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(eval_metrics, 'eval')
                    eval_next = False

                if self._step > 0:
                    # Update agent with last episode data
                    _train_metrics = self.agent.update(self._tds_l, self._tds_r)
                    self.agent.model.eval()
                    train_metrics.update(_train_metrics)
                    
                    episode_reward = torch.tensor([td['reward'] for td in self._tds_l[1:]]).sum()
                    episode_reward += torch.tensor([td['reward'] for td in self._tds_r[1:]]).sum()
                    train_metrics.update(
                        episode_reward=episode_reward,
                        episode_success=info['success'],
                    )
                    train_metrics.update(self.common_metrics())
                    self.logger.log(train_metrics, 'train')

                obs = self.env.reset()
                self.agent.reset()
                self.agent.model.train()
                self._ep_idx += 1
                self._tds_l = []
                self._tds_r = []
                data_count = 0
            
            if self.cfg.act_individually:    
                action, dist, value, state, is_act_left = self.agent.act(obs, t0=len(self._tds_l)==1)
            else:
                action_l, dist_l, value, state, _ = self.agent.act(obs, t0=len(self._tds_l)==1)
                action_r, dist_r, value, state, _ = self.agent.act(obs, t0=len(self._tds_l)==1)
                action = torch.concat([action_l, action_r], dim=1).detach()
            action_np = action[0].detach().cpu()#.numpy()
            next_obs, reward, done, info = self.env.step(action_np)
            done = torch.ones_like(reward) if done else torch.zeros_like(reward)
            if self.cfg.act_individually:
                if is_act_left:
                    self._tds_l.append(self.to_td(state, action, reward, dist[0], dist[1], value, done))
                else:
                    self._tds_r.append(self.to_td(state, action, reward, dist[0], dist[1], value, done))
            else:
                self._tds_l.append(self.to_td(state, action, reward, dist_l[0], dist_l[1], value, done))
                self._tds_r.append(self.to_td(state, action, reward, dist_r[0], dist_r[1], value, done))

            obs = next_obs
            self._step += 1
            data_count += 1

        self.logger.finish(self.agent)
        
        
class OnlineSingleImitationTrainer(OnlineTrainer):
    def __init__(self, cfg, env, agent, logger):
        super().__init__(cfg, env, agent, None, logger)
        
        
    def to_td(self, obs, action=None, reward=None, mu=None, log_sigma=None, value=None, done=None):
        """Creates a TensorDict for a new episode."""
        if isinstance(obs, dict):
            obs = TensorDict(obs, batch_size=(), device='cpu')
        else:
            obs = obs.unsqueeze(0).cpu()
        if action is None:
            action = torch.full_like(self.env.rand_act(), float('nan'))
        if reward is None:
            reward = torch.tensor(float('nan'))
        if mu is None:
            mu = torch.full_like(self.env.rand_act(), float('nan'))
        if log_sigma is None:
            log_sigma = torch.full_like(self.env.rand_act(), float('nan'))
        if value is None:
            value = torch.tensor(float('nan'))
        if done is None:
            done = torch.tensor(float('nan'))
        td = TensorDict(dict(
            obs=obs,
            action=action.unsqueeze(0),
            reward=reward.unsqueeze(0),
            mu=mu.unsqueeze(0),
            log_sigma=log_sigma.unsqueeze(0),
            value=value.unsqueeze(0),
            done=done.unsqueeze(0),
        ), batch_size=(1,))
        return td
        
        
    @torch.no_grad()
    def eval(self):
        """Evaluate a TD-MPC2 agent."""
        ep_rewards, ep_successes = [], []
        for i in range(self.cfg.eval_episodes):
            obs, done, ep_reward, t = self.env.reset(), False, 0, 0
            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=(i==0))
            while not done:
                action, _, _ = self.agent.act(obs, t0=t==0, eval_mode=True)
                obs, reward, done, info = self.env.step(action)
                ep_reward += reward
                t += 1
                if self.cfg.save_video:
                    self.logger.video.record(self.env)
            ep_rewards.append(ep_reward)
            ep_successes.append(info['success'])
            if self.cfg.save_video:
                self.logger.video.save(self._step)
        return dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_success=np.nanmean(ep_successes),
        )


    def train(self):
        """Train a SingleImitation agent."""
        train_metrics, done, eval_next = {}, True, True
        data_count = 0
        self._tds = []
        while self._step <= self.cfg.steps:

            # Evaluate agent periodically
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True

            # On Episode End 
            if done:
                if eval_next:
                    eval_metrics = self.eval()
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(eval_metrics, 'eval')
                    eval_next = False

                if self._step > 0:
                    # Update agent with last episode data
                    _train_metrics = self.agent.update(self._tds)
                    self.agent.model.eval()
                    train_metrics.update(_train_metrics)
                    
                    episode_reward = torch.tensor([td['reward'] for td in self._tds[1:]]).sum()
                    train_metrics.update(
                        episode_reward=episode_reward,
                        episode_success=info['success'],
                    )
                    train_metrics.update(self.common_metrics())
                    self.logger.log(train_metrics, 'train')

                obs = self.env.reset()
                self.agent.model.train()
                self._ep_idx += 1
                self._tds = []
                data_count = 0
                
            action, dist, value = self.agent.act(obs, t0=len(self._tds)==1)
            action_np = action[0].detach().cpu()
            obs, reward, done, info = self.env.step(action_np)
            done = torch.ones_like(reward) if done else torch.zeros_like(reward)
            self._tds.append(self.to_td(obs, action, reward, dist[0], dist[1], value, done))

            self._step += 1
            data_count += 1

        self.logger.finish(self.agent)