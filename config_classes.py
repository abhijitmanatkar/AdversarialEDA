from dataclasses import dataclass
from typing import List


@dataclass
class GailConfig:
    episode_length: int
    total_steps: int
    arch: List[int]
    parallel_envs: int
    ppo_n_steps: int
    ppo_batch_size: int
    demo_batch_size: int
    gen_replay_buffer_capacity: int
    n_disc_updates_per_round: int
    n_gen_updates_per_round: int


@dataclass
class BCConfig:
    arch: List[int]
    n_epochs: int
    lr: float


@dataclass
class Config:
    experiment_name: str
    dataset_type: str  # ["NETWORKS", "FLIGHTS"]
    dataset_num: int
    algo: str  # ["GAIL", "BC", "BC_GAIL"]
    back_allowed: bool
    dataset_path: str = None
    trajectories_path: str = None
    gail: GailConfig = None
    bc: BCConfig = None
