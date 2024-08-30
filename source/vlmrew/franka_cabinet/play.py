# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from Stable-Baselines3."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from Stable-Baselines3.")
# parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Isaaclab-Panda_cabinet-v0")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="/home/levi/projects/IsaacLab/source/vlmrew/franka_cabinet/logs/2024-07-01_17-31-30/model_4375000_steps.zip",
    help="Path to model checkpoint."
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--max_iterations", type=int, default=350, help="RL Policy training iterations.")
parser.add_argument("--chkpnt_step_cnt", type=int, default=125000, help="Checkpoint step count.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import torch
from datetime import datetime


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.parse_cfg import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
from franka_cabinet_env import FrankaCabinetEnv, FrankaCabinetEnvCfg
import yaml
from pickle import dump as dump_pickle


def read_yaml_file(filepath):
    with open(filepath, "r") as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)


def main():
    """Play with stable-baselines agent."""
    # Get the env config
    env_cfg = FrankaCabinetEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = 5.0

    args_cli.chkpnt_step_cnt = f'{int(args_cli.checkpoint.split("/")[-1].split("_")[1]):07}'

    # Create the environment
    env = FrankaCabinetEnv(cfg=env_cfg, render_mode="rgb_array", chkpnt_step_cnt=args_cli.chkpnt_step_cnt)

    # Get the agent config
    agent_cfg = read_yaml_file("/home/levi/projects/IsaacLab/source/vlmrew/franka_cabinet/sb3_ppo_cfg.yaml")
    if args_cli.max_iterations:
        agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs

    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env)

    # normalize environment (if needed)
    if "normalize_input" in agent_cfg:
        env = VecNormalize(
            env,
            training=True,
            norm_obs="normalize_input" in agent_cfg and agent_cfg.pop("normalize_input"),
            norm_reward="normalize_value" in agent_cfg and agent_cfg.pop("normalize_value"),
            clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )

    # directory for logging into
    log_root_path = os.path.join(
        "/home/levi/projects/IsaacLab/source/vlmrew/franka_cabinet/logs"
    )
    log_root_path = os.path.abspath(log_root_path)

    # check checkpoint is valid
    print(f"checkpoint path: {args_cli.checkpoint}")
    if args_cli.checkpoint is None:
        if args_cli.use_last_checkpoint:
            checkpoint = "model_.*.zip"
        else:
            checkpoint = "model.zip"
        checkpoint_path = get_checkpoint_path(log_root_path, ".*", checkpoint)
    else:
        checkpoint_path = args_cli.checkpoint
    # create agent from stable baselines
    print(f"Loading checkpoint from: {checkpoint_path}")
    agent = PPO.load(checkpoint_path, env, print_system_info=True)

    # reset environment
    obs = env.reset()
    step_cnt = 0
    rewards = []
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            step_cnt += 1
            # agent stepping
            actions, _ = agent.predict(obs, deterministic=True)
            # env stepping
            obs, reward, done, info = env.step(actions)
            rewards.append(reward[0])
            if step_cnt == 1100:
                break
    # close the simulator
    #     print(f"Total rewards: {sum(rewards)}")
    env.close()
    # save the rewards array as a pickle file
    with open(f"/home/levi/projects/IsaacLab/source/vlmrew/franka_cabinet/logs/rewards.pkl", "wb") as f:
        dump_pickle(rewards, f)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
