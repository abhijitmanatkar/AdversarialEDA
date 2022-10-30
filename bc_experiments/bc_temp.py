import numpy as np
import pickle
from imitation.algorithms.bc import BC
from Networks import custom_ac_policy_class_builder
from AnalysisEnv import AnalysisEnv
from stable_baselines3.common.policies import ActorCriticPolicy

with open('./expert_trajectories/train/1.pkl', 'rb') as f:
    expert_trajs = pickle.load(f)

env = AnalysisEnv('./raw_datasets/1.tsv')

policy = ActorCriticPolicy(env.observation_space, env.action_space, lr_schedule=lambda x: 1e-4, net_arch=([dict(pi=[100,100,100], vf=[100,100,100])]))

bc_trainer = BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    policy=policy,
    demonstrations=expert_trajs,
)

bc_trainer.train(n_epochs=200, log_interval=10)

bc_trainer.save_policy('./bc_policy_temp')
