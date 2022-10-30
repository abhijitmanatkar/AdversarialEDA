from stable_baselines3.common.policies import ActorCriticPolicy
import torch.nn as nn
import torch
import torch.nn.functional as F
import itertools
import numpy as np
import gym


class CustomNetwork(nn.Module):
    """Class for defining policy and value networks to be used with a custom ActorCriticPolicy class.

    Args:

        feature_dim: Size of the input (observation) space.
        policy_arch: Hidden layer sizes of the policy network given as a list.
                     Example: [100,100,100] for 3 hidden layers of 100 neurons each.
        policy_activation: The activation function to be used for the policy network.
                           Example: torch.nn.Tanh
        value_arch: Hidden layer sizes of the value network given as a list.
        value_activation: The activation function to be used for the vallue network.
    """ 

    def __init__(self, feature_dim, policy_arch, policy_activation, value_arch, value_activation):

        super(CustomNetwork, self).__init__()

        self.latent_dim_pi = policy_arch[-1]
        self.latent_dim_vf = value_arch[-1]

        policy_net_modules = nn.ModuleList(
            [nn.Linear(feature_dim, policy_arch[0]), policy_activation()])
        
        for inp, out in zip(policy_arch[:-2], policy_arch[1:-1]):
            policy_net_modules.append(nn.Linear(inp, out))
            # Add batch norm
            policy_net_modules.append(nn.BatchNorm1d(out))
            # Activation
            policy_net_modules.append(policy_activation())
            #Dropout
            policy_net_modules.append(nn.Dropout(p=0.3))
        
        policy_net_modules.append(nn.Linear(policy_arch[-2], policy_arch[-1]))
        policy_net_modules.append(policy_activation())

        # for inp, out in zip(policy_arch, policy_arch[1:]):
        #     policy_net_modules.append(nn.Linear(inp, out))
        #     policy_net_modules.append(policy_activation())
        
        self.policy_net = nn.Sequential(*policy_net_modules)

        value_net_modules = nn.ModuleList(
            [nn.Linear(feature_dim, value_arch[0]), value_activation()])

        for inp, out in zip(value_arch[:-2], value_arch[1:-1]):
            # Add batch norm
            value_net_modules.append(nn.BatchNorm1d(out))
            # Activation
            value_net_modules.append(value_activation())
            #Dropout
            value_net_modules.append(nn.Dropout(p=0.3))

        value_net_modules.append(nn.Linear(value_arch[-2], policy_arch[-1]))
        value_net_modules.append(policy_activation())  
        
        # for inp, out in zip(value_arch, value_arch[1:]):
        #     value_net_modules.append(nn.Linear(inp, out))
        #     value_net_modules.append(value_activation())
        
        self.value_net = nn.Sequential(*value_net_modules)

    def forward(self, features):
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)


def custom_ac_policy_class_builder(obs_len, policy_arch, policy_activation=nn.Tanh, value_arch=None, value_activation=None):
    """Method for generating a custom subclass of the ActorCriticPolicy class from sb3.
    This class can be used as the `gen_algo` input for the GAIL class from imitation.
    
    Args:
        obs_len: Size of the input (observation) space.
        policy_arch: Hidden layer sizes of the policy network given as a list.
                     Example: [100,100,100] for 3 hidden layers of 100 neurons each.
        policy_activation: The activation function to be used for the policy network.
                           Example: torch.nn.Tanh
        value_arch: Hidden layer sizes of the value network given as a list.
        value_activation: The activation function to be used for the vallue network.
    """
    
    if value_arch is None:
        value_arch = policy_arch
    if value_activation is None:
        value_activation = policy_activation

    # class CustomACPolicy(ActorCriticPolicy):

    #     def __init__(self, *args, **kwargs):
    #         super(CustomACPolicy, self).__init__(*args, **kwargs)
    #         self.ortho_init = False

    #     def _build_mlp_extractor(self):
    #         self.mlp_extractor = CustomNetwork(
    #             obs_len, policy_arch, policy_activation, value_arch, value_activation)
    # return CustomACPolicy
    
    mlp_extractor = CustomNetwork(obs_len, policy_arch, policy_activation, value_arch, value_activation)
    return mlp_extractor
