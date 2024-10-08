import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['LAZY_LEGACY_OP'] = '0'
import warnings
warnings.filterwarnings('ignore')
import torch

import hydra
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_env
# from dialectic import DialecticMPC, DialecticImitation, SingleImitation
from reinforce import ReinforceAgent, ReinforceDiscreteAgent, ReinforcePredictiveAgent, ReinforceDiscretePredictiveAgent
from a2c import A2CAgent, A2CDiscreteAgent, A2CDiscretePredictiveAgent
from trainer.offline_trainer import OfflineTrainer
from trainer.online_trainer import OnlineTrainer, OnlineDialecticTrainer, OnlineDialecticImitationTrainer, OnlineSingleImitationTrainer
from common.logger import Logger

torch.backends.cudnn.benchmark = True


@hydra.main(config_name='dialectic_config', config_path='.')
def train(cfg: dict):
    """
    Script for training single-task / multi-task DialecticMPC agents.

    Most relevant args:
        `task`: task name (or mt30/mt80 for multi-task training)
        `model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
        `steps`: number of training/environment steps (default: 10M)
        `seed`: random seed (default: 1)

    See config.yaml for a full list of args.

    Example usage:
    ```
        $ python train.py task=mt80 model_size=48
        $ python train.py task=mt30 model_size=317
        $ python train.py task=dog-run steps=7000000
    ```
    """
    assert torch.cuda.is_available()
    assert cfg.steps > 0, 'Must train for at least 1 step.'
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)

    if cfg.agent_class == 'reinforce':
        agent_cls = ReinforceAgent
    elif cfg.agent_class == 'reinforce_discrete':
        agent_cls = ReinforceDiscreteAgent
    elif cfg.agent_class == 'reinforce_pred':
        agent_cls = ReinforcePredictiveAgent
    elif cfg.agent_class == 'reinforce_pred_discrete':
        agent_cls = ReinforceDiscretePredictiveAgent
    elif cfg.agent_class == 'a2c':
        agent_cls = A2CAgent
    elif cfg.agent_class == 'a2c_discrete':
        agent_cls = A2CDiscreteAgent
    elif cfg.agent_class == 'a2c_pred_discrete':
        agent_cls = A2CDiscretePredictiveAgent
    else:
        raise ValueError(f'Invalid agent class: {cfg.agent}')
        
    trainer_cls = OfflineTrainer if cfg.multitask else OnlineSingleImitationTrainer
    trainer = trainer_cls(
        cfg=cfg,
        env=make_env(cfg),
        agent=agent_cls(cfg),
        logger=Logger(cfg),
    )
    trainer.train()
    print('\nTraining completed successfully')


if __name__ == '__main__':
	train()
