"""An example of training PPO against OpenAI Gym Envs.
This script is an example of training a PPO agent against OpenAI Gym envs.
Both discrete and continuous action spaces are supported.
To solve CartPole-v0, run:
    python train_ppo_gym.py --env CartPole-v0
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA


from future import standard_library

from models.chinerrl_models.parameterized_softmax import FCParamSoftmaxPolicy

standard_library.install_aliases()  # NOQA
import argparse

import chainer
from chainer import functions as F
import gym
import gym.wrappers
import numpy as np

import chainerrl
from chainerrl.agents import a3c
from chainerrl.agents import PPO
from chainerrl import experiments
from chainerrl import links
from chainerrl import misc
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay
from chainerrl import policies


from logging import getLogger

logger = getLogger(__name__)

import Utilities.Configuration.config as cfg


class A3CFFSoftmax(chainer.ChainList, a3c.A3CModel):
    """An example of A3C feedforward softmax policy."""

    # def __init__(self, ndim_obs, n_actions, hidden_sizes=(200, 200)):
    #     self.pi = policies.SoftmaxPolicy(
    #         model=links.MLP(ndim_obs, n_actions, hidden_sizes))
    #     self.v = links.MLP(ndim_obs, 1, hidden_sizes=hidden_sizes)
    #     super().__init__(self.pi, self.v)

    def __init__(self, ndim_obs, n_discrete_entries,
                 n_hidden_layers=2, n_hidden_channels=400, beta=1.0):
        self.pi = policies.FCSoftmaxPolicy(
            ndim_obs, n_discrete_entries,
            n_hidden_layers, n_hidden_channels, beta=beta)
        self.v = chainerrl.v_functions.FCVFunction(
            ndim_obs,
            n_hidden_channels=n_hidden_channels,
            n_hidden_layers=n_hidden_layers,
            last_wscale=0.01,
        )
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)
    

class A3CFFParamSoftmax(chainer.ChainList, a3c.A3CModel):
    """An example of A3C feedforward softmax policy."""

    def __init__(self, ndim_obs, n_discrete_entries, parametric_segments, parametric_segments_sizes,
                 n_hidden_layers=2, n_hidden_channels=400, beta=1.0):
        self.pi = FCParamSoftmaxPolicy(
            ndim_obs, n_discrete_entries, parametric_segments, parametric_segments_sizes,
            n_hidden_layers, n_hidden_channels, beta=beta)
        self.v = chainerrl.v_functions.FCVFunction(
            ndim_obs,
            n_hidden_channels=n_hidden_channels,
            n_hidden_layers=n_hidden_layers,
            last_wscale=0.01,
        )
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


class A3CFFMellowmax(chainer.ChainList, a3c.A3CModel):
    """An example of A3C feedforward mellowmax policy."""

    def __init__(self, ndim_obs, n_actions, hidden_sizes=(200, 200)):
        self.pi = policies.MellowmaxPolicy(
            model=links.MLP(ndim_obs, n_actions, hidden_sizes))
        self.v = links.MLP(ndim_obs, 1, hidden_sizes=hidden_sizes)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


class A3CFFGaussian(chainer.Chain, a3c.A3CModel):
    """An example of A3C feedforward Gaussian policy."""

    def __init__(self, obs_size, action_space,
                 n_hidden_layers=2, n_hidden_channels=64,
                 bound_mean=None):
        assert bound_mean in [False, True]
        super().__init__()
        hidden_sizes = (n_hidden_channels,) * n_hidden_layers
        with self.init_scope():
            self.pi = policies.FCGaussianPolicyWithStateIndependentCovariance(
                obs_size, action_space.low.size,
                n_hidden_layers, n_hidden_channels,
                var_type='diagonal', nonlinearity=F.tanh,
                bound_mean=bound_mean,
                min_action=action_space.low, max_action=action_space.high,
                mean_wscale=1e-2)
            self.v = links.MLP(obs_size, 1, hidden_sizes=hidden_sizes)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


class PPOchianerrl:
    def __init__(self, args, sample_env):
        obs_space = sample_env.observation_space
        action_space = sample_env.action_space

        # Normalize observations based on their empirical mean and variance
        obs_normalizer = chainerrl.links.EmpiricalNormalization(
            obs_space.low.size, clip_threshold=5)

        # Switch policy types accordingly to action space types
        if args.arch == 'FFSoftmax':
            #model = A3CFFSoftmax(obs_space.low.size, action_space.n)
            model = A3CFFSoftmax(obs_space.low.size, sample_env.env_prop.get_softmax_layer_size(),
                                 n_hidden_channels=600, beta=cfg.beta)
        elif args.arch == 'FFMellowmax':
            model = A3CFFMellowmax(obs_space.low.size, action_space.n)
        elif args.arch == 'FFGaussian':
            model = A3CFFGaussian(obs_space.low.size, action_space,
                                  bound_mean=args.bound_mean, n_hidden_channels=cfg.n_hidden_channels)
        elif args.arch == 'FFParamSoftmax':
            model = A3CFFParamSoftmax(obs_space.low.size, sample_env.env_prop.get_pre_output_layer_size(),
                                      sample_env.env_prop.get_parametric_segments(),
                                      sample_env.env_prop.get_parametric_softmax_segments_sizes(),
                                      n_hidden_channels=600, beta=cfg.beta)
        else:
            raise NotImplementedError

        opt = chainer.optimizers.Adam(alpha=args.adam_lr, eps=1e-5)
        opt.setup(model)
        if args.weight_decay > 0:
            opt.add_hook(NonbiasWeightDecay(args.weight_decay))

        # a workaround for saving obs_normalizer
        # see https://github.com/chainer/chainerrl/issues/376
        if 'obs_normalizer' not in PPO.saved_attributes:
            PPO.saved_attributes.append('obs_normalizer')

        agent = PPO(model, opt,
                    obs_normalizer=obs_normalizer,
                    gpu=args.gpu,
                    phi=lambda x: x.astype(np.float32, copy=False),
                    gamma=args.ppo_gamma,
                    lambd=args.ppo_lambda,
                    update_interval=args.ppo_update_interval,
                    minibatch_size=args.batchsize, epochs=args.epochs,
                    clip_eps_vf=None, entropy_coef=args.entropy_coef,
                    standardize_advantages=args.standardize_advantages,
                    )

        if args.load:
            agent.load(args.load)

        self._agent = agent

    @property
    def agent(self):
        return self._agent



def main():
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--env', type=str, default='Hopper-v2')
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--arch', type=str, default='FFGaussian',
                        choices=('FFSoftmax', 'FFMellowmax',
                                 'FFGaussian'))
    parser.add_argument('--bound-mean', action='store_true')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--steps', type=int, default=10 ** 6)
    parser.add_argument('--eval-interval', type=int, default=10000)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--reward-scale-factor', type=float, default=1e-2)
    parser.add_argument('--standardize-advantages', action='store_true')
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--logger-level', type=int, default=logging.DEBUG)
    parser.add_argument('--monitor', action='store_true')
    parser.add_argument('--window-size', type=int, default=100)

    parser.add_argument('--update-interval', type=int, default=2048)
    parser.add_argument('--log-interval', type=int, default=1000)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--entropy-coef', type=float, default=0.0)
    args = parser.parse_args()

    #logging.basicConfig(level=args.logger_level)

    # Set a random seed used in ChainerRL
    misc.set_random_seed(args.seed, gpus=(args.gpu,))

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    args.outdir = experiments.prepare_output_dir(args, args.outdir)

    def make_env(process_idx, test):
        env = gym.make(args.env)
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[process_idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = chainerrl.wrappers.CastObservationToFloat32(env)
        if args.monitor:
            env = gym.wrappers.Monitor(env, args.outdir)
        if not test:
            # Scale rewards (and thus returns) to a reasonable range so that
            # training is easier
            env = chainerrl.wrappers.ScaleReward(env, args.reward_scale_factor)
        if args.render:
            env = chainerrl.wrappers.Render(env)
        return env

    def make_batch_env(test):
        return chainerrl.envs.MultiprocessVectorEnv(
            [(lambda: make_env(idx, test))
             for idx, env in enumerate(range(args.num_envs))])

    # Only for getting timesteps, and obs-action spaces
    sample_env = gym.make(args.env)
    timestep_limit = sample_env.spec.tags.get(
        'wrapper_config.TimeLimit.max_episode_steps')
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space

    # Normalize observations based on their empirical mean and variance
    obs_normalizer = chainerrl.links.EmpiricalNormalization(
        obs_space.low.size, clip_threshold=5)

    # Switch policy types accordingly to action space types
    if args.arch == 'FFSoftmax':
        model = A3CFFSoftmax(obs_space.low.size, action_space.n)
    elif args.arch == 'FFMellowmax':
        model = A3CFFMellowmax(obs_space.low.size, action_space.n)
    elif args.arch == 'FFGaussian':
        model = A3CFFGaussian(obs_space.low.size, action_space,
                              bound_mean=args.bound_mean)

    opt = chainer.optimizers.Adam(alpha=args.lr, eps=1e-5)
    opt.setup(model)
    if args.weight_decay > 0:
        opt.add_hook(NonbiasWeightDecay(args.weight_decay))
    agent = PPO(model, opt,
                obs_normalizer=obs_normalizer,
                gpu=args.gpu,
                update_interval=args.ppo_update_interval,
                minibatch_size=args.batchsize, epochs=args.epochs,
                clip_eps_vf=None, entropy_coef=args.entropy_coef,
                standardize_advantages=args.standardize_advantages,
                )

    if args.load:
        agent.load(args.load)

    # Linearly decay the learning rate to zero
    def lr_setter(env, agent, value):
        agent.optimizer.alpha = value

    lr_decay_hook = experiments.LinearInterpolationHook(
        args.steps, args.lr, 0, lr_setter)

    # Linearly decay the clipping parameter to zero
    def clip_eps_setter(env, agent, value):
        agent.clip_eps = value

    clip_eps_decay_hook = experiments.LinearInterpolationHook(
        args.steps, 0.2, 0, clip_eps_setter)

    experiments.train_agent_batch_with_evaluation(
        agent=agent,
        env=make_batch_env(False),
        eval_env=make_batch_env(True),
        outdir=args.outdir,
        steps=args.steps,
        eval_n_runs=args.eval_n_runs,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        return_window_size=args.window_size,
        max_episode_len=timestep_limit,
        save_best_so_far_agent=False,
        step_hooks=[
            lr_decay_hook,
            clip_eps_decay_hook,
        ],
    )


if __name__ == '__main__':
    main()