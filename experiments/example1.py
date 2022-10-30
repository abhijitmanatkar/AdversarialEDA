from config_classes import GailConfig, BCConfig, Config

config = Config(
    experiment_name="Example1",
    dataset_type="NETWORKS",
    dataset_num=1,
    trajectories_path=None,
    algo="BC_GAIL",
    back_allowed=True,
    gail=GailConfig(
        episode_length=12,
        total_steps=200000,
        arch=[100, 100, 100],
        parallel_envs=8,
        ppo_n_steps=48,
        ppo_batch_size=32,
        demo_batch_size=96*2,
        gen_replay_buffer_capacity=768,
        n_disc_updates_per_round=2,
        n_gen_updates_per_round=1
    ),
    bc=BCConfig(
        arch=[100, 100, 100],
        n_epochs=100,
        lr=1e-4
    )
)
