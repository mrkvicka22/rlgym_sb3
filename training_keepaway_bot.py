import os

import numpy as np
import torch.optim
from rlgym.envs import Match
from rlgym.utils.action_parsers import DiscreteAction, DefaultAction
from new_ppo.new_ppo import PPO
#from stable_baselines3.ppo import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
from stable_baselines3.ppo import MlpPolicy

from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.reward_functions.common_rewards import VelocityBallToGoalReward, EventReward, VelocityPlayerToBallReward
from rlgym.utils.reward_functions.combined_reward import CombinedReward
from custom_rewards.keepaway_rewards import AntiJumpReward
from rlgym.utils.state_setters import DefaultState, RandomState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition, NoTouchTimeoutCondition
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction
from custom_state_setters.proximity_random_setter import ProximityRandomState
from rlgym_tools.sb3_utils.sb3_instantaneous_fps_callback import SB3InstantaneousFPSCallback
from custom_state_setters.symmetric_setter import KickoffLikeSetter
from rlgym_tools.extra_state_setters.weighted_sample_setter import WeightedSampleSetter
from rlgym_tools.extra_state_setters.replay_setter import ReplaySetter

if __name__ == '__main__':  # Required for multiprocessing
    frame_skip = 8  # Number of ticks to repeat an action
    half_life_seconds = 10  # Easier to conceptualize, after this many seconds the reward discount is 0.5
    n_instances = 4
    team_size = 1

    fps = 120 / frame_skip
    proportion_after = 0.1
    proportion_life_seconds = 15
    gamma = proportion_after**(1/(fps*proportion_life_seconds))
    print(f"fps={fps}, gamma={gamma})")

    reward_func = CombinedReward(
        (VelocityBallToGoalReward(), VelocityPlayerToBallReward()),
        (0.05, 0.01))

    replays1v1_path = "custom_state_setters/ssl_1v1.npy"
    def get_match():  # Need to use a function so that each instance can call it and produce their own objects
        return Match(
            team_size=team_size,  # 3v3 to get as many agents going as possible, will make results more noisy
            tick_skip=frame_skip,
            reward_function=reward_func,  # Simple reward since example code
            self_play=True,
            terminal_conditions=[TimeoutCondition(round(fps * 60)),NoTouchTimeoutCondition(round(fps * 15))],  # Some basic terminals
            obs_builder=AdvancedObs(),  # Not that advanced, good default
            state_setter=WeightedSampleSetter((KickoffLikeSetter(ball_on_ground=False),RandomState(cars_on_ground=False),ReplaySetter(replays1v1_path)),(10,40,40)),
            action_parser=KBMAction(),  # Discrete > Continuous don't @ me
        )

    load_model = input("[L]oad/[T]rain").lower() == "l"
    env = SB3MultipleInstanceEnv(get_match, n_instances, force_paging=False)  # Start 2 instances, waiting 60 seconds between each
    env = VecCheckNan(env)  # Optional
    env = VecMonitor(env)  # Recommended, logs mean reward and ep_len to Tensorboard
    env = VecNormalize(env, norm_obs=False, gamma=gamma)  # Highly recommended, normalizes rewards

    # Hyperparameters presumably better than default; inspired by original PPO paper
    model = PPO(
        MlpPolicy,
        env,
        n_epochs=32,  # PPO calls for multiple epochs
        learning_rate=1e-4,  # Around this is fairly common for PPO
        ent_coef=0.005,  # From PPO Atari
        vf_coef=1.,  # From PPO Atari
        gamma=gamma,  # Gamma as calculated using half-life
        verbose=3,  # Print out all the info as we're going
        batch_size=4096 * 8,  # Batch size as high as possible within reason
        n_steps=4096 * 8,  # Number of steps to perform before optimizing network
        tensorboard_log="out/logs",  # `tensorboard --logdir out/logs` in terminal to see graphs
        device="auto",  # Uses GPU if available
        policy_kwargs={'net_arch': [dict(pi=[256,256,256,256], vf=[256,256,256,256])], 'optimizer_class':torch.optim.Adam}
    )
    callback = CheckpointCallback(round(5_000_000 / env.num_envs), save_path="experiment_policy", name_prefix="rl_model")
    if load_model:
        model = PPO.load("experiment_policy/rl_model_error_checkpoint.zip", env, custom_objects=dict(n_envs=n_instances*2*team_size),n_epochs=32)
        env.reset()
        try:
            model.learn(100_000_000_000, callback=callback, reset_num_timesteps=False)
        except:
            model.save("experiment_policy/rl_model_error_checkpoint")
            print("saved on error and save yo arse")
    else:
        try:
            model.learn(100_000_000_000, callback=callback)
        except:
            model.save("experiment_policy/rl_model_error_checkpoint")
            print("saved on error and save yo arse")

