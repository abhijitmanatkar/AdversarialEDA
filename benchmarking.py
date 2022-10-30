from stable_baselines3 import PPO
import numpy as np
from AnalysisEnv import AnalysisEnv
from collections import defaultdict
import utils
import os
import parse_nb

# Atena related imports
from benchmark.atena.evaluation.metrics import (
    EvalInstance,
    DisplaysTreeBleuMetric,
    PrecisionMetric,
    get_dataframe_all_eval_metrics
)

from benchmark.atena.simulation.actions import (
    GroupAction,
    Column,
    AggregationFunction,
    BackAction,
    FilterAction,
    FilterOperator,
)

from benchmark.atena.simulation.dataset import (
    Dataset,
    DatasetMeta,
    DatasetName,
    CyberDatasetName,
    FlightsDatasetName,
    SchemaName,
)

from benchmark.atena.evaluation.metrics import get_dataframe_all_eval_metrics
from benchmark.atena.simulation.utils import random_action_generator


# These two dictionaries are necessary for converting between our representation and
# the representation used in the ATENA benchmark

benchmark_equiv_filter_ops = {
    'EQ': FilterOperator.EQUAL,
    'NEQ': FilterOperator.NOTEQUAL,
    'GT': FilterOperator.GE,
    'GE': FilterOperator.GE,
    'LT': FilterOperator.LT,
    'LE': FilterOperator.LE,
    'CONTAINS': FilterOperator.CONTAINS,
    'STARTS_WITH': FilterOperator.STARTS_WITH,
    'ENDS_WITH': FilterOperator.ENDS_WITH
}

benchmark_equiv_agg_ops = {
    len: AggregationFunction.COUNT,
    np.sum: AggregationFunction.SUM,
    utils.hack_max: AggregationFunction.MAX,
    utils.hack_min: AggregationFunction.MIN,
    np.mean: AggregationFunction.MEAN
}


def collect_trajectory(policy, env, length=None, deterministic=True):
    """Collect a trajectory from `env` using `policy`

    Args:
        `policy`: The policy used to sample actions in the environment
        `env`: The gym-like environment from which samples will be collected
        `length`: The length of the trajectory to be collected. If `None`, then trajectory is collected till `env` returns done = True.
    """
    itr = 0
    done = False

    if length is not None:
        def stop_condition(done, itr): return (itr == length) or done
    else:
        def stop_condition(done, itr): return done

    obs = env.reset()
    while not stop_condition(done, itr):
        act, _ = policy.predict(obs, deterministic=deterministic)
        obs, rew, done, info = env.step(act)
        itr += 1

    return env.state_history, env.action_history


def trajectory_to_atena_comparable(traj_actions):
    """Convert a list of actions in the form of `AnalysisEnv.action_history` list to an 
    action list as required by the ATENA benchmark.

    Args:
        `traj_actions`: A list of actions similar to `AnalysisEnv.action_history`
    """

    action_list = []
    for action in traj_actions:
        if action['type'] == 'filter':
            action_list.append(
                FilterAction(
                    filtered_column=Column(action['col']),
                    filter_operator=benchmark_equiv_filter_ops[action['operator']],
                    filter_term=action['filter_term']
                )
            )
        elif action['type'] == 'group':
            action_list.append(
                GroupAction(
                    grouped_column=Column(action['col']),
                    aggregated_column=Column(action['agg_option']),
                    aggregation_function=benchmark_equiv_agg_ops[action['agg_func']]
                )
            )
        elif action['type'] == 'back':
            action_list.append(BackAction())

    return action_list


def modify_num_to_packet_num(trajectory):
    """Modify column == 'number' to 'packet_number' to make it better comparablein the atena_benchmark"""

    trajectory_mod = []
    for action in trajectory:
        if action['type'] in ['filter', 'group']:
            if action['col'] == 'number':
                action['col'] = 'packet_number'
            if action['agg_option'] == 'number':
                action['agg_option'] = 'packet_number'
        trajectory_mod.append(action)

    return trajectory_mod


class TrajectoryIncompatibleException(Exception):
    pass


def trajectory_benchmark_score(trajectory, ref_dataset=1, ref_actions=None, dataset_type="NETWORKING"):
    """Get benchmark score for a single trajectory

    Args:
        `traj`: The trajectory in the form of what is returned by `collect_trajectory`
        `ref_dataset`: The dataset number that the trajectory is for.
        `ref_actions`: The reference trajectories that we compare against, provided as an atena benchmark type action list. 
        If `None`, we compare against the default as per the ATENA benchmark.

    Return:
        A dataframe of all benchmark scores for that trajectory
    """

    trajectory = modify_num_to_packet_num(trajectory)
    action_list = trajectory_to_atena_comparable(trajectory)

    return action_list_benchmark_score(action_list, ref_dataset, ref_actions, dataset_type)

    # ref_datasets = [
    #     CyberDatasetName.DATASET1,
    #     CyberDatasetName.DATASET2,
    #     CyberDatasetName.DATASET3,
    #     CyberDatasetName.DATASET4
    # ]
    # # try:
    # dataset_meta = DatasetMeta(SchemaName.CYBER, ref_datasets[ref_dataset-1])
    # if ref_actions is not None:
    #     eval_instance = utils.CustomEvalInstance(dataset_meta, trajectory, ref_actions)
    # else:
    #     eval_instance = EvalInstance(dataset_meta, trajectory)
    # return get_dataframe_all_eval_metrics([eval_instance])
    # # except:
    # #     raise


def action_list_benchmark_score(action_list, ref_dataset=1, ref_actions=None, dataset_type="NETWORKING"):
    """Get benchmark score for a single trajectory passed as an atena compatible action list

        Args:
        `action_list`: The trajectory in the form of an atena benchmark compatible action list
        `dataset_type`: "NETWORKING"/"FLIGHTS" for specifying the dataset type
        `ref_dataset`: The dataset number that the trajectory is for.
        `ref_actions`: The reference trajectories that we compare against, provided as an atena benchmark type action list. 
        If `None`, we compare against the default as per the ATENA benchmark.

    """

    ref_dataset_names = {
        "NETWORKING": [
            CyberDatasetName.DATASET1,
            CyberDatasetName.DATASET2,
            CyberDatasetName.DATASET3,
            CyberDatasetName.DATASET4
        ],
        "FLIGHTS": [
            FlightsDatasetName.DATASET1,
            FlightsDatasetName.DATASET2,
            FlightsDatasetName.DATASET3,
            FlightsDatasetName.DATASET4,
        ]
    }

    schemas = {"NETWORKING": SchemaName.CYBER, "FLIGHTS": SchemaName.FLIGHTS}

    schema = schemas[dataset_type]
    dataset_name = ref_dataset_names[dataset_type][ref_dataset-1]

    dataset_meta = DatasetMeta(schema, dataset_name)
    if ref_actions is not None:
        eval_instance = utils.CustomEvalInstance(
            dataset_meta, action_list, ref_actions)
    else:
        eval_instance = EvalInstance(dataset_meta, action_list)
    return get_dataframe_all_eval_metrics([eval_instance])


def get_benchmark_scores(policy, env, num_trajs, dataset_type="NETWORKING", ref_dataset=1, ref_actions=None, traj_len=None, deterministic=True):
    """Get benchmark scores for `policy` acting in `env`.

    Args:
        `policy`: The policy which will generate action samples
        `env`: The environment in which the policy will be acting. 
        Will be an instance of AnalysisEnv.
        `num_trajs`: The number of trajectories that will be atttempted to be sampled. 
        Final benchmark values will be averaged across the trajectories that satisfy the ATENA benchmark criteria.
        `dataset_type`: "NETWORKING"/"FLIGHTS" for specifying the dataset type
        `ref_dataset`: The dataset number that the trajectory is for.
        `ref_actions`: The reference trajectories that we compare against, provided as an atena benchmark type action list. 
        If `None`, we compare against the default as per the ATENA benchmark.
        `traj_len`: Lengths of trajectories to collect

    """
    scores = {}
    columns = ['Precision', 'T-BLEU-1', 'T-BLEU-2', 'T-BLEU-3', 'EDA-Sim']
    for column in columns:
        scores[column] = []

    for i in range(num_trajs):
        trajectory = collect_trajectory(
            policy, env, traj_len, deterministic=deterministic)[1]
        # try:
        score_df = trajectory_benchmark_score(
            trajectory, ref_dataset, ref_actions, dataset_type)
        # except Exception as e:
        #     print(e)
        #     print(f"Trajectory {i} incompatible with benchmark")
        #     continue
        for column in columns:
            scores[column].append(score_df.iloc[0][column])
        #print(f"Done {i}")

    return scores


def atena_notebook_benchmark_scores(notebooks_dir_path, ref_dataset=1, ref_actions=None, dataset_type="NETWORKING"):
    """Get benchmark scores for notebooks in notebooks_dir_path

    Args:
        `notebooks_dir_path`: Path to directory where notebooks are stored.
        `ref_dataset`: The dataset number that the trajectory is for.
        `ref_actions`: The reference trajectories that we compare against, provided as an atena benchmark type action list. 
        If `None`, we compare against the default as per the ATENA benchmark.
    """

    scores = {}
    columns = ['Precision', 'T-BLEU-1', 'T-BLEU-2', 'T-BLEU-3', 'EDA-Sim']
    for column in columns:
        scores[column] = []

    nbs = os.listdir(notebooks_dir_path)
    for nb in nbs:
        action_list = parse_nb.parse_nb(os.path.join(notebooks_dir_path, nb))
        score_df = action_list_benchmark_score(
            action_list, ref_dataset, ref_actions, dataset_type)
        for column in columns:
            scores[column].append(score_df.iloc[0][column])

    return scores


def print_avg_scores(scores):
    for col in scores:
        print(col, np.mean(scores[col]))

def get_avg_scores(scores):
    return {col:np.mean(scores[col]) for col in scores}