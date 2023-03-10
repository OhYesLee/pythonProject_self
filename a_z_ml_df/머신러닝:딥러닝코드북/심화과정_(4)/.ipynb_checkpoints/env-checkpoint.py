import numpy as np
import gym
import pybullet as p
import pybullet_envs
import pybullet_envs.bullet.minitaur_gym_env as minitaur_gym_env
import pybullet_envs.bullet.racecarGymEnv as racecarGymEnv
import pybullet_envs.bullet.kukaGymEnv as kukaGymEnv
from custom_envs.minitaur_duck import MinitaurDuckBulletEnv
from custom_envs.minitaur_ball import MinitaurBallBulletEnv

def make_env(env_name, seed=-1, render_mode=False):
  if (env_name.startswith("RacecarBulletEnv")):
    print("bullet_racecar_started")
    env = racecarGymEnv.RacecarGymEnv(isDiscrete=False, renders=render_mode)
  elif (env_name.startswith("AugmentBipedalWalker")):
    if (env_name.startswith("AugmentBipedalWalkerHardcore")):
      if (env_name.startswith("AugmentBipedalWalkerHardcoreSmallLegs")):
        from box2d.walker_env import AugmentBipedalWalkerHardcoreSmallLegs
        env = AugmentBipedalWalkerHardcoreSmallLegs()
      else:
        from box2d.walker_env import AugmentBipedalWalkerHardcore
        env = AugmentBipedalWalkerHardcore()
    elif (env_name.startswith("AugmentBipedalWalkerSmallLegs")):
      from box2d.walker_env import AugmentBipedalWalkerSmallLegs
      env = AugmentBipedalWalkerSmallLegs()
    elif (env_name.startswith("AugmentBipedalWalkerTallLegs")):
      from box2d.walker_env import AugmentBipedalWalkerTallLegs
      env = AugmentBipedalWalkerTallLegs()
    else:
      from box2d.walker_env import AugmentBipedalWalker
      env = AugmentBipedalWalker()
  elif (env_name.startswith("MinitaurBulletEnv")):
    print("bullet_minitaur_started")
    env = minitaur_gym_env.MinitaurBulletEnv(render=render_mode)
  elif (env_name.startswith("MinitaurDuckBulletEnv")):
    print("bullet_minitaur_duck_started")
    env = MinitaurDuckBulletEnv(render=render_mode)
  elif (env_name.startswith("MinitaurBallBulletEnv")):
    print("bullet_minitaur_ball_started")
    env = MinitaurBallBulletEnv(render=render_mode)
  elif (env_name.startswith("KukaBulletEnv")):
    print("bullet_kuka_grasping started")
    env = kukaGymEnv.KukaGymEnv(renders=render_mode,isDiscrete=False)
  else:
    if env_name.startswith("Augment"):
      import robogym
    if env_name.startswith("AugmentAnt"):
      from robogym import AugmentAnt
      env = AugmentAnt()
    elif env_name.startswith("AugmentHopper"):
      from robogym import AugmentHopper
      env = AugmentHopper()
    elif env_name.startswith("AugmentHalfCheetah"):
      from robogym import AugmentHalfCheetah
      env = AugmentHalfCheetah()
    else:
      env = gym.make(env_name)
    if render_mode and not env_name.startswith("Augment"):
      env.render("human")
  if (seed >= 0):
    env.seed(seed)
  return env
