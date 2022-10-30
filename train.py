import torch
import numpy as np
import pickle
import wandb
import dataclasses
import tqdm
from AnalysisEnv import AnalysisEnv, MultiDatasetEnv
import imitation.util.logger as imit_logger
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.rewards import reward_wrapper
from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms.bc import BC
from imitation.scripts.train_adversarial import save as save_trainer
from imitation.util import networks
from stable_baselines3.ppo import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from config_classes import Config

# Edit this line to specify the correct config file
from experiments.example1 import config

# if config.dataset_num is None:
if config.dataset_type == "NETWORKS":
    dataset_paths = [f'./raw_datasets/{i}.tsv' for i in range(1, 5)]
    print(dataset_paths)
elif config.dataset_type == "FLIGHTS":
    dataset_paths = [
        f'benchmark/atena/datasets/flights/{i}.tsv' for i in [1, 3, 4]]

print(dataset_paths)

def train_bc(config: Config, logger=None):

    if config.trajectories_path is None:
        config.trajectories_path = f'./expert_trajectories/train/{config.dataset_num}.pkl'

    with open(config.trajectories_path, 'rb') as f:
        expert_trajs = pickle.load(f)

    env = MultiDatasetEnv(dataset_paths)

    policy = ActorCriticPolicy(
        env.observation_space,
        env.action_space,
        lr_schedule=lambda x: config.bc.lr,
        net_arch=([dict(pi=config.bc.arch, vf=config.bc.arch)])
    )

    trainer = BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        policy=policy,
        demonstrations=expert_trajs,
        custom_logger=logger
    )

    trainer.train(n_epochs=config.bc.n_epochs, log_interval=10)
    trainer.save_policy(f'./saved_agents/{config.experiment_name}.BCAgent')


def train_gail(config: Config, logger=None, bc_init=False):

    venv = DummyVecEnv([lambda:MultiDatasetEnv(
        dataset_paths, max_steps=config.gail.episode_length) for _ in range(config.gail.parallel_envs)])

    # Generator definition
    policy_kwargs = dict(
        net_arch=[dict(pi=config.gail.arch, vf=config.gail.arch)])
    learner = PPO(ActorCriticPolicy, venv, n_steps=config.gail.ppo_n_steps,
                  batch_size=config.gail.ppo_batch_size, policy_kwargs=policy_kwargs)
    if bc_init:
        bc_policy = torch.load(
            f'./saved_agents/{config.experiment_name}.BCAgent')
        learner.policy.load_state_dict(bc_policy.state_dict())

    # Discriminator definition
    env = MultiDatasetEnv(dataset_paths, max_steps=config.gail.episode_length)
    reward_net = BasicRewardNet(env.observation_space, env.action_space)

    # Expert trajectories
    if config.trajectories_path is None:
        config.trajectories_path = f'./expert_trajectories/train/{config.dataset_num}.pkl'
    with open(config.trajectories_path, 'rb') as f:
        rollouts = pickle.load(f)

    trainer = GAIL(
        demonstrations=rollouts,
        demo_batch_size=config.gail.demo_batch_size,
        gen_replay_buffer_capacity=config.gail.gen_replay_buffer_capacity,
        n_disc_updates_per_round=config.gail.n_disc_updates_per_round,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        custom_logger=logger,
        allow_variable_horizon=True
    )
    
    # Wrapping up trainer venv to include penalties 
    def get_custom_reward_function(trainer):
        venv = trainer.venv_wrapped

        def repeat_penalty(old_obs, acts, obs, dones):
            return np.array([env.repeat_penalty() for env in venv.envs])

        def custom_reward_function(old_obs, acts, obs, dones):
            r1 = trainer.reward_train.predict_processed(old_obs, acts, obs, dones)
            r2 = np.array(repeat_penalty(old_obs, acts, obs, dones))
            return r1 + r2

        return custom_reward_function

    r = get_custom_reward_function(trainer)
    trainer.venv_wrapped = reward_wrapper.RewardVecEnvWrapper(
        trainer.venv_buffering,
        reward_fn=r
    )
    trainer.venv_train = trainer.venv_wrapped
    trainer.gen_algo.set_env(trainer.venv_train)
    trainer.gen_algo.set_logger(trainer.logger)

    # Checking number of rounds is at least 1
    n_rounds = config.gail.total_steps // (
        config.gail.n_gen_updates_per_round * trainer.gen_train_timesteps)
    assert n_rounds >= 1, (
        "No updates (need at least "
        f"{trainer.gen_train_timesteps} timesteps, have only "
        f"total_timesteps={config.gail.total_steps})!"
    )

    # Training loop
    for r in tqdm.tqdm(range(0, n_rounds), desc="round"):
        for _ in range(config.gail.n_gen_updates_per_round):
            trainer.train_gen(trainer.gen_train_timesteps)
        for _ in range(trainer.n_disc_updates_per_round):
            with networks.training(trainer.reward_train):
                # switch to training mode (affects dropout, normalization)
                trainer.train_disc()
        # Add callback here if needed
        trainer.logger.dump(trainer._global_step)

    save_trainer(trainer, f'./saved_agents/{config.experiment_name}.GAILAgent')


if __name__ == '__main__':

    wandb.init(project='ProjectName', name=config.experiment_name,
               config=dataclasses.asdict(config))
    logger = imit_logger.configure(
        format_strs=["stdout", "log", "csv", "wandb"])

    if config.algo == "BC":
        train_bc(config, logger)

    elif config.algo == "GAIL":
        train_gail(config, logger)

    elif config.algo == "BC_GAIL":
        train_bc(config)
        train_gail(config, logger, bc_init=True)
