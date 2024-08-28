import os

from dm_control.rl import control
from dm_control.suite import common
from dm_control.suite import walker
from dm_control.utils import rewards
from dm_control.utils import io as resources

_TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tasks')

_YOGA_STAND_HEIGHT = 1.0
_YOGA_LIE_DOWN_HEIGHT = 0.08
_YOGA_LEGS_UP_HEIGHT = 1.1


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return resources.GetResource(os.path.join(_TASKS_DIR, 'walker.xml')), common.ASSETS

# [action]
# motor: right_hip
# motor: right_knee
# motor: right_ankle
# motor: left_hip
# motor: left_knee
# motor: left_ankle

# [observation]
# orientation: torso xx
# orientation: torso xz
# orientation: right_thigh xx
# orientation: right_thigh xz
# orientation: right_leg  xx
# orientation: right_leg  xz
# orientation: right_foot  xx
# orientation: right_foot  xz
# orientation: left_thigh  xx
# orientation: left_thigh  xz
# orientation: left_leg  xx
# orientation: left_leg  xz
# orientation: left_foot  xx
# orientation: left_foot  xz
# height
# velocity: rootz
# velocity: rootx
# velocity: rooty
# velocity: right_hip
# velocity: right_knee
# velocity: right_ankle
# velocity: left_hip
# velocity: left_knee
# velocity: left_ankle

ACTION_DESCRIPTIONS = [
    {'index': 0, 'object': 'right_hip', 'min': -1, 'max': 1, 'object_type': 'motor', 'action_type': 'continuous', 'left_actor': True, 'right_actor': False},
    {'index': 1, 'object': 'right_knee', 'min': -1, 'max': 1, 'object_type': 'motor', 'action_type': 'continuous', 'left_actor': True, 'right_actor': False},
    {'index': 2, 'object': 'right_ankle', 'min': -1, 'max': 1, 'object_type': 'motor', 'action_type': 'continuous', 'left_actor': True, 'right_actor': False},
    {'index': 3, 'object': 'left_hip', 'min': -1, 'max': 1, 'object_type': 'motor', 'action_type': 'continuous', 'left_actor': False, 'right_actor': True},
    {'index': 4, 'object': 'left_knee', 'min': -1, 'max': 1, 'object_type': 'motor', 'action_type': 'continuous', 'left_actor': False, 'right_actor': True},
    {'index': 5, 'object': 'left_ankle', 'min': -1, 'max': 1, 'object_type': 'motor', 'action_type': 'continuous', 'left_actor': False, 'right_actor': True},
]

OBSERVATION_DESCRIPTIONS = [
    {'index': 0, 'object': 'torso', 'object_type': 'orientation', 'axis': 'xx', 'left_actor': True, 'right_actor': True},
    {'index': 1, 'object': 'torso', 'object_type': 'orientation', 'axis': 'xz', 'left_actor': True, 'right_actor': True},
    {'index': 2, 'object': 'right_thigh', 'object_type': 'orientation', 'axis': 'xx', 'left_actor': True, 'right_actor': False},
    {'index': 3, 'object': 'right_thigh', 'object_type': 'orientation', 'axis': 'xz', 'left_actor': True, 'right_actor': False},
    {'index': 4, 'object': 'right_leg', 'object_type': 'orientation', 'axis': 'xx', 'left_actor': True, 'right_actor': False},
    {'index': 5, 'object': 'right_leg', 'object_type': 'orientation', 'axis': 'xz', 'left_actor': True, 'right_actor': False},
    {'index': 6, 'object': 'right_foot', 'object_type': 'orientation', 'axis': 'xx', 'left_actor': True, 'right_actor': False},
    {'index': 7, 'object': 'right_foot', 'object_type': 'orientation', 'axis': 'xz', 'left_actor': True, 'right_actor': False},
    {'index': 8, 'object': 'left_thigh', 'object_type': 'orientation', 'axis': 'xx', 'left_actor': False, 'right_actor': True},
    {'index': 9, 'object': 'left_thigh', 'object_type': 'orientation', 'axis': 'xz', 'left_actor': False, 'right_actor': True},
    {'index': 10, 'object': 'left_leg', 'object_type': 'orientation', 'axis': 'xx', 'left_actor': False, 'right_actor': True},
    {'index': 11, 'object': 'left_leg', 'object_type': 'orientation', 'axis': 'xz', 'left_actor': False, 'right_actor': True},
    {'index': 12, 'object': 'left_foot', 'object_type': 'orientation', 'axis': 'xx', 'left_actor': False, 'right_actor': True},
    {'index': 13, 'object': 'left_foot', 'object_type': 'orientation', 'axis': 'xz', 'left_actor': False, 'right_actor': True},
    {'index': 14, 'object': 'torso', 'object_type': 'height', 'left_actor': True, 'right_actor': True},
    {'index': 15, 'object': 'rootz', 'object_type': 'velocity', 'left_actor': True, 'right_actor': True},
    {'index': 16, 'object': 'rootx', 'object_type': 'velocity', 'left_actor': True, 'right_actor': True},
    {'index': 17, 'object': 'rooty', 'object_type': 'velocity', 'left_actor': True, 'right_actor': True},
    {'index': 18, 'object': 'right_hip', 'object_type': 'velocity', 'left_actor': True, 'right_actor': False},
    {'index': 19, 'object': 'right_knee', 'object_type': 'velocity', 'left_actor': True, 'right_actor': False},
    {'index': 20, 'object': 'right_ankle', 'object_type': 'velocity', 'left_actor': True, 'right_actor': False},
    {'index': 21, 'object': 'left_hip', 'object_type': 'velocity', 'left_actor': False, 'right_actor': True},
    {'index': 22, 'object': 'left_knee', 'object_type': 'velocity', 'left_actor': False, 'right_actor': True},
    {'index': 23, 'object': 'left_ankle', 'object_type': 'velocity', 'left_actor': False, 'right_actor': True},
]


@walker.SUITE.add('custom')
def walk_backwards(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Walk Backwards task."""
  physics = walker.Physics.from_xml_string(*get_model_and_assets())
  task = BackwardsPlanarWalker(move_speed=walker._WALK_SPEED, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
      **environment_kwargs)


@walker.SUITE.add('custom')
def run_backwards(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Run Backwards task."""
  physics = walker.Physics.from_xml_string(*get_model_and_assets())
  task = BackwardsPlanarWalker(move_speed=walker._RUN_SPEED, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
      **environment_kwargs)


@walker.SUITE.add('custom')
def arabesque(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Arabesque task."""
  physics = walker.Physics.from_xml_string(*get_model_and_assets())
  task = YogaPlanarWalker(goal='arabesque', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
      **environment_kwargs)


@walker.SUITE.add('custom')
def lie_down(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Lie Down task."""
  physics = walker.Physics.from_xml_string(*get_model_and_assets())
  task = YogaPlanarWalker(goal='lie_down', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
      **environment_kwargs)


@walker.SUITE.add('custom')
def legs_up(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Legs Up task."""
  physics = walker.Physics.from_xml_string(*get_model_and_assets())
  task = YogaPlanarWalker(goal='legs_up', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
      **environment_kwargs)


@walker.SUITE.add('custom')
def headstand(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Headstand task."""
  physics = walker.Physics.from_xml_string(*get_model_and_assets())
  task = YogaPlanarWalker(goal='flip', move_speed=0, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
      **environment_kwargs)


@walker.SUITE.add('custom')
def flip(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Flip task."""
  physics = walker.Physics.from_xml_string(*get_model_and_assets())
  task = YogaPlanarWalker(goal='flip', move_speed=walker._RUN_SPEED*0.75, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
      **environment_kwargs)


@walker.SUITE.add('custom')
def backflip(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Backflip task."""
  physics = walker.Physics.from_xml_string(*get_model_and_assets())
  task = YogaPlanarWalker(goal='flip', move_speed=-walker._RUN_SPEED*0.75, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
      **environment_kwargs)


class BackwardsPlanarWalker(walker.PlanarWalker):
    """Backwards PlanarWalker task."""
    def __init__(self, move_speed, random=None):
        super().__init__(move_speed, random)
    
    def get_reward(self, physics):
        standing = rewards.tolerance(physics.torso_height(),
                                 bounds=(walker._STAND_HEIGHT, float('inf')),
                                 margin=walker._STAND_HEIGHT/2)
        upright = (1 + physics.torso_upright()) / 2
        stand_reward = (3*standing + upright) / 4
        if self._move_speed == 0:
            return stand_reward
        else:
            move_reward = rewards.tolerance(physics.horizontal_velocity(),
                                            bounds=(-float('inf'), -self._move_speed),
                                            margin=self._move_speed/2,
                                            value_at_margin=0.5,
                                            sigmoid='linear')
            return stand_reward * (5*move_reward + 1) / 6


class YogaPlanarWalker(walker.PlanarWalker):
    """Yoga PlanarWalker tasks."""
    
    def __init__(self, goal='arabesque', move_speed=0, random=None):
        super().__init__(0, random)
        self._goal = goal
        self._move_speed = move_speed
    
    def _arabesque_reward(self, physics):
        standing = rewards.tolerance(physics.torso_height(),
                                bounds=(_YOGA_STAND_HEIGHT, float('inf')),
                                margin=_YOGA_STAND_HEIGHT/2)
        left_foot_height = physics.named.data.xpos['left_foot', 'z']
        right_foot_height = physics.named.data.xpos['right_foot', 'z']
        left_foot_down = rewards.tolerance(left_foot_height,
                                bounds=(-float('inf'), _YOGA_LIE_DOWN_HEIGHT),
                                margin=_YOGA_STAND_HEIGHT/2)
        right_foot_up = rewards.tolerance(right_foot_height,
                                bounds=(_YOGA_STAND_HEIGHT, float('inf')),
                                margin=_YOGA_STAND_HEIGHT/2)
        upright = (1 - physics.torso_upright()) / 2
        arabesque_reward = (3*standing + left_foot_down + right_foot_up + upright) / 6
        return arabesque_reward
    
    def _lie_down_reward(self, physics):
        torso_down = rewards.tolerance(physics.torso_height(),
                                bounds=(-float('inf'), _YOGA_LIE_DOWN_HEIGHT),
                                margin=_YOGA_LIE_DOWN_HEIGHT/2)
        thigh_height = (physics.named.data.xpos['left_thigh', 'z'] + physics.named.data.xpos['right_thigh', 'z']) / 2
        thigh_down = rewards.tolerance(thigh_height,
                                bounds=(-float('inf'), _YOGA_LIE_DOWN_HEIGHT),
                                margin=_YOGA_LIE_DOWN_HEIGHT/2)
        feet_height = (physics.named.data.xpos['left_foot', 'z'] + physics.named.data.xpos['right_foot', 'z']) / 2
        feet_down = rewards.tolerance(feet_height,
                                bounds=(-float('inf'), _YOGA_LIE_DOWN_HEIGHT),
                                margin=_YOGA_LIE_DOWN_HEIGHT/2)
        upright = (1 - physics.torso_upright()) / 2
        lie_down_reward = (3*torso_down + thigh_down + upright) / 5
        return lie_down_reward
    
    def _legs_up_reward(self, physics):
        torso_down = rewards.tolerance(physics.torso_height(),
                                bounds=(-float('inf'), _YOGA_LIE_DOWN_HEIGHT),
                                margin=_YOGA_LIE_DOWN_HEIGHT/2)
        thigh_height = (physics.named.data.xpos['left_thigh', 'z'] + physics.named.data.xpos['right_thigh', 'z']) / 2
        thigh_down = rewards.tolerance(thigh_height,
                                bounds=(-float('inf'), _YOGA_LIE_DOWN_HEIGHT),
                                margin=_YOGA_LIE_DOWN_HEIGHT/2)
        feet_height = (physics.named.data.xpos['left_foot', 'z'] + physics.named.data.xpos['right_foot', 'z']) / 2
        legs_up = rewards.tolerance(feet_height,
                                bounds=(_YOGA_LEGS_UP_HEIGHT, float('inf')),
                                margin=_YOGA_LEGS_UP_HEIGHT/2)
        upright = (1 - physics.torso_upright()) / 2
        legs_up_reward = (3*torso_down + 2*legs_up + thigh_down + upright) / 7
        return legs_up_reward
    
    def _flip_reward(self, physics):
        thigh_height = (physics.named.data.xpos['left_thigh', 'z'] + physics.named.data.xpos['right_thigh', 'z']) / 2
        thigh_up = rewards.tolerance(thigh_height,
                                bounds=(_YOGA_STAND_HEIGHT, float('inf')),
                                margin=_YOGA_STAND_HEIGHT/2)
        feet_height = (physics.named.data.xpos['left_foot', 'z'] + physics.named.data.xpos['right_foot', 'z']) / 2
        legs_up = rewards.tolerance(feet_height,
                                bounds=(_YOGA_LEGS_UP_HEIGHT, float('inf')),
                                margin=_YOGA_LEGS_UP_HEIGHT/2)
        upside_down_reward = (3*legs_up + 2*thigh_up) / 5
        if self._move_speed == 0:
            return upside_down_reward
        move_reward = rewards.tolerance(physics.horizontal_velocity(),
                                    bounds=(self._move_speed, float('inf')) if self._move_speed > 0 else (-float('inf'), self._move_speed),
                                    margin=abs(self._move_speed)/2,
                                    value_at_margin=0.5,
                                    sigmoid='linear')
        return upside_down_reward * (5*move_reward + 1) / 6
    
    def get_reward(self, physics):
        if self._goal == 'arabesque':
            return self._arabesque_reward(physics)
        elif self._goal == 'lie_down':
            return self._lie_down_reward(physics)
        elif self._goal == 'legs_up':
            return self._legs_up_reward(physics)
        elif self._goal == 'flip':
            return self._flip_reward(physics)
        else:
            raise NotImplementedError(f'Goal {self._goal} is not implemented.')


if __name__ == '__main__':
    env = legs_up()
    obs = env.reset()
    import numpy as np
    next_obs, reward, done, info = env.step(np.zeros(6))
