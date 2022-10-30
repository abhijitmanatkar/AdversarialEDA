import pandas as pd
import numpy as np
from react_lib.utilities import Repository
import gym
import utils
from utils import SUPPORTED_OPERATORS, SUPPORTED_AGG_FUNCS
from utils import FilteringTuple, AggregationTuple
from utils import empty_env_state
from utils import CounterWithoutNanKeys
import logging
from scipy.stats import entropy
from atena_basic.gym_atena.reactida.utils.distance import display_distance
from atena_basic.gym_atena.lib.helpers import DisplayTuple

logger = logging.getLogger(__name__)

dataset_dict = {}


class Dataset:
    """Class for representing a dataset. By default, the dataset is a tsv file, loaded as
    a pandas dataframe. 
    Includes methods for modifying the base dataframe (by operations such as filter, group).
    Also has methods for generating the vector representation of a df display.

    Args:
        dataset_path: path to the dataset tsv file.
    """

    def __init__(self, dataset_path):

        print(dataset_path)
        self.dataset_path = dataset_path
        # The base dataframe over which all operations will be performed
        #self.base_df = pd.read_csv(dataset_path, sep='\t', index_col=0)
        dataset_dict[dataset_path] = pd.read_csv(
            dataset_path, sep='\t', index_col=0)

        # Attributes/columns of the base dataframe which can be used for
        # filter/group/agg operations
        self.keys = None
        self.numeric_keys = None

        # A dictionary where keys are the dataset attributes(keys) and the
        # values are the values that these columns can be filtered by
        self.filter_options = {}
        self.non_unique_filter_options = {}

        # List of all possible filters
        self.filter_list = []

        # Frequency bins for action selection
        bins = [0] + [0.1 / 2 ** i for i in range(11, 0, -1)] + [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                                                                 0.5, 0.55,
                                                                 0.6, 0.7, 0.8, 0.9, 1.0]
        self.bins = np.array(bins)

        self.load_metadata()

    @property
    def base_df(self):
        return dataset_dict[self.dataset_path]

    def load_metadata(self):
        """
        Loads and sets metadata properties of the dataset like keys, 
        numeric_keys, filter_options, etc.
        """

        # Drop keys that do not have at least 2 unique values
        column_mask = pd.array(self.base_df.nunique(axis=0)) > 1

        self.keys = self.base_df.loc[:, column_mask].columns.to_list()
        self.numeric_keys = self.base_df[self.keys].select_dtypes(
            np.number).columns.tolist()

        # Unique values in a column are filter options
        for key in self.keys:
            self.non_unique_filter_options[key] = self.base_df[key].dropna(
            ).tolist()
            self.filter_options[key] = self.base_df[key].dropna(
            ).unique().tolist()
            self.filter_list.extend(self.filter_options[key])

    def get_filtered_df(self, dataset_df, filters):
        """Returns a filtered dataframe.

        Args:
            dataset_df: The df to which the filter is applied
            filters: List of filters to apply in the form of utils.FilterTuple objects.
        """

        if not filters:
            return dataset_df

        df = dataset_df.copy()

        for filter in filters:
            field = filter.field
            opr = filter.condition
            value = filter.term

            if opr in ['EQ', 'NEQ', 'GT', 'GE', 'LT', 'LE']:
                opr = utils.OPERATOR_MAP[opr]
                try:
                    if pd.api.types.is_numeric_dtype(df[field]) and value == '<UNK>':
                        value = np.nan
                    value = float(value) if str(df[field].dtype) not in [
                        'object', 'category'] and value != '<UNK>' else value
                    df = df[opr(df[field], value)]
                except:
                    #logger.warning(f"Filter on column {field} with operator {opr} and value {value} is emtpy")
                    return df.truncate(after=-1)
            else:
                # print("***"+field+"\n\n" + str(op_num) +"\n\n" + str(opr)+"\n\n***")
                try:
                    if opr == 'CONTAINS':
                        if df[field].dtype == 'O' or str(df[field].dtype) == 'category':
                            df = df[df[field].str.contains(
                                value, na=False, regex=False)]
                        elif df[field].dtype == 'f8' or df[field].dtype == 'u4' or df[field].dtype == 'int64':
                            df = df[df[field].astype(str).str.contains(
                                str(value), na=False, regex=False)]
                        else:
                            logger.warning(
                                f"Filter on column {field} with operator Contains and value {value} is emtpy")
                            raise NotImplementedError

                    elif opr == 'STARTS_WITH':
                        if df[field].dtype == 'O' or str(df[field].dtype) == 'category':
                            df = df[df[field].str.startswith(value, na=False)]
                        elif df[field].dtype == 'f8' or df[field].dtype == 'u4' or df[field].dtype == 'int64':
                            df = df[df[field].astype(str).str.startswith(
                                str(value), na=False)]
                        else:
                            logger.warning(
                                f"Filter on column {field} with operator Contains and value {value} is emtpy")
                            raise NotImplementedError

                    elif opr == 'ENDS_WITH':
                        if df[field].dtype == 'O' or str(df[field].dtype) == 'category':
                            df = df[df[field].str.endswith(value, na=False)]
                        elif df[field].dtype == 'f8' or df[field].dtype == 'u4' or df[field].dtype == 'int64':
                            df = df[df[field].astype(str).str.endswith(
                                str(value), na=False)]
                        else:
                            logger.warning(
                                f"Filter on column {field} with operator Contains and value {value} is emtpy")
                            raise NotImplementedError
                    else:
                        logger.warning(
                            f"Filter on column {field} with operator {opr} and value {value} raised NotImplementedError and will be emtpy")
                        raise NotImplementedError

                except NotImplementedError:
                    return df.truncate(after=-1)
        return df

    def get_groupby_df(self, df, groupings, aggregations):
        """Returns a grouped and aggregated dataframe

        Args:
            df: The dataframe which is to be grouped
            groupings: List of columns to group by
            aggregations: List of utils.AggregationTuple objects
        """

        if not groupings:
            return None, None

        df_gb = df.groupby(list(groupings), observed=True)

        # agg_dict={'number':len} #all group-by gets the count by default in REACT-UI
        # if aggregations: #Custom aggregations: sum,count,avg,min,max
        agg_dict = {}
        # agg is a namedtuple of type AggregationTuple
        for agg in aggregations:
            agg_dict[agg.field] = agg.type

        try:
            agg_df = df_gb.agg(agg_dict)
        except:
            return None, None
        return df_gb, agg_df

    def get_raw_data(self, dataset_df, state):
        """Returns the raw datafram corresponding to a particular state.

        Args:
            dataset_df: The base dataframe who's filtered, grouped state we want to obtain.
            state: utils.EnvStateTuple object
        """

        filtered_df = self.get_filtered_df(dataset_df, state["filtering"])
        gdf, agg_df = self.get_groupby_df(
            filtered_df, state["grouping"], state["aggregations"])
        return filtered_df, gdf, agg_df

    def get_data_column_measures(self, column):
        """
        for each column, compute its: (1) normalized value entropy (2)Null count (3)Unique values count
        """
        B = 20
        size = len(column)
        if size == 0:
            return {"unique": 0.0, "nulls": 1.0, "entropy": 0.0}
        column_na_value_counts = CounterWithoutNanKeys(column)
        column_na_value_counts_values = column_na_value_counts.values()
        cna_size = sum(column_na_value_counts_values)
        # number of NaNs
        n = size - cna_size
        # u = column.nunique()
        # number of unique non NaNs values
        u = len(column_na_value_counts.keys())
        # normalizing the number of unique non-NaNs values by the total number of non-NaNs
        u_n = u / cna_size if u != 0 else 0

        if column.name not in self.numeric_keys:
            # h=entropy(column_na.value_counts(sort=False, dropna=False).values)
            h = entropy(list(column_na_value_counts_values))
            h = h / np.log(cna_size) if cna_size > 1 else 0.0
        else:  # if numeric data only in column
            h = entropy(np.histogram(column_na_value_counts.non_nan_iterable, bins=B)[0]) / np.log(
                B) if cna_size > 1 else 0.0

        return {"unique": u_n, "nulls": n / size, "entropy": h}

    def calc_data_layer(self, df):
        if len(df) == 0:
            ret_dict = {}
            for k in self.keys:
                ret_dict[k] = {"unique": 0.0, "nulls": 1.0, "entropy": 0.0}
            return ret_dict
        else:
            return {key: self.get_data_column_measures(df[key]) for key in self.keys}

    def get_grouping_measures(self, group_obj, agg_df):
        if group_obj is None or agg_df is None:
            return None

        B = 20
        groups_num = group_obj.ngroups
        if groups_num == 0:
            site_std = 0.0
            inverse_size_mean = 0
            inverse_ngroups = 0
        else:
            sizes = group_obj.size()
            sizes_sum = sizes.sum()
            nsizes = sizes / sizes_sum
            site_std = nsizes.std(ddof=0)
            # size_mean = nsizes.mean()
            sizes_mean = sizes.mean()
            inverse_size_mean = 1 / sizes_mean
            # ngroups=min(len(sizes)/sizes_sum,1)
            if sizes_sum > 0:
                # ngroups = groups_num/sizes_sum
                # Use inverse of ngroups
                inverse_ngroups = 1 / groups_num
            else:
                inverse_ngroups = 0
                inverse_size_mean = 0

        group_keys = group_obj.keys
        agg_keys = list(agg_df.keys())
        agg_nve_dict = {}
        if agg_keys is not None:
            for ak in agg_keys:
                column = agg_df[ak]
                column_na = column.dropna()
                cna_size = len(column_na)
                if cna_size <= 1:
                    agg_nve_dict[ak] = 0.0
                # elif column.name not in NUMERIC_KEYS:
                elif agg_df[ak].dtype == 'O' or str(agg_df[ak].dtype) == 'category':
                    h = entropy(column_na.value_counts().values)
                    agg_nve_dict[ak] = h / np.log(cna_size)
                else:
                    agg_nve_dict[ak] = entropy(np.histogram(
                        column_na, bins=B)[0]) / np.log(B)
        return {"group_attrs": group_keys, "agg_attrs": agg_nve_dict, "inverse_ngroups": inverse_ngroups,
                "site_std": site_std, "inverse_size_mean": inverse_size_mean}

    def calc_gran_layer(self, group_obj, agg_df):
        # print(disp_row.display_id)
        return self.get_grouping_measures(group_obj, agg_df)

    def calculate_state_vector(self, state):
        """Calculate the state vector for a given state.

        Args:
            state: utils.EnvStateTuple object representing the state.
        """
        fdf, gdf, adf = self.get_raw_data(self.base_df, state)
        data_layer = self.calc_data_layer(fdf)
        gran_layer = self.calc_gran_layer(gdf, adf)

        vlist = []
        for d in data_layer.values():
            vlist += [d["unique"], d["nulls"], d["entropy"]]
            # vlist+=list(d.values())

        if gran_layer is None:
            # The first 0 represents the inverse number of groups (i.e. for 0 groups 1/0 is mapped to 0)
            # The second 0 represents the inverse of groups' mean size (i.e. for mean of 0, 1/0 is mapped to 0)
            vlist += [-1 for _ in self.keys] + [0, 0, 0]
        else:
            for k in self.keys:
                if k in gran_layer['agg_attrs'].keys():
                    vlist.append(gran_layer['agg_attrs'][k])
                elif k in gran_layer['group_attrs']:
                    vlist.append(2)
                else:
                    vlist.append(-1)

            vlist += [gran_layer['inverse_ngroups'],
                      gran_layer['inverse_size_mean'], gran_layer['site_std']]

        return np.array(vlist, dtype=np.float32)


class AnalysisEnv(gym.Env):
    """Class representing the analysis environment. 

    Args:
        dataset_path: path to the dataset tsv file
        max_steps: Maximum number of steps that can be taken in one episode.
    """

    def __init__(self, dataset_path, max_steps=12, back_supported=True, stop_supported=True):
        super(AnalysisEnv).__init__()

        self.dataset_path = dataset_path
        self.dataset = Dataset(dataset_path)

        self.max_steps = max_steps
        self.step_num = None

        # Stack of state history, last state is popped when BACK action is taken
        self.state_stack = None

        # State history - not popped when BACK action is taken
        self.state_history = None
        self.action_history = None
        self.obs_history = None
        self.dhist = None

        # Stop/back actions
        if (not back_supported) and (not stop_supported):
            raise Exception("Need to support at least one of back/stop")
        elif back_supported and (not stop_supported):
            self.supported_actions = utils.SUPPORTED_ACTIONS_ONLY_BACK
        elif (not back_supported) and stop_supported:
            self.supported_actions = utils.SUPPORTED_ACTIONS_ONLY_STOP
        else:
            self.supported_actions = utils.SUPPORTED_ACTIONS

        # Action and state spaces
        self.observation_space = self.get_observation_space()
        self.action_space = self.get_action_space()

        # Co-efficients for humanity and diversity scores
        self.diversity_coeff = 6.0
        self.humanity_coeff = 4.8

    def reset(self):
        self.step_num = 0
        self.state_stack = [empty_env_state]
        self.state_history = [empty_env_state]
        self.action_history = []
        self.obs_history = []
        obs = self.get_latest_observation()
        self.obs_history = [obs]
        self.dhist = [self.get_disp(self.state_history[0])]
        return obs

    def get_disp(self, state):
        """Returns display to be appended to self.dhist"""
        fdf, gdf, adf = self.dataset.get_raw_data(self.dataset.base_df, state)
        fs = (fdf, adf)

        data_layer = self.dataset.calc_data_layer(fdf)
        gran_layer = self.dataset.calc_gran_layer(gdf, adf)
        disp_tpl = DisplayTuple(data_layer=data_layer, granularity_layer=gran_layer)

        return disp_tpl

    def compute_diversity_reward(self):
        last_display = self.dhist[-1]
        last_state = self.state_history[-1]
        # sim_vec will contain the similarity scores of the last display and all others
        sim_vec = []

        # (1.e) Compute the diversity-based reward:
        for i, d in enumerate(self.dhist[:-1]):
            i_state = self.state_history[i]
            state1 = i_state
            state2 = last_state
            if str(state1) > str(state2):
                state1, state2 = state2, state1

            display_distance_result_obj = display_distance(d, last_display)

            dist = display_distance_result_obj.display_distance
            # (1.f) Punishment if the exact same display was already seen
            # (2.f) Punishment if the same data layer is seen in the same subsession after a filter action.
            # This means that the two filter action filtered the exact same rows in the same subsession
            if dist == 0 or (
                display_distance_result_obj.data_distance == 0
                and self.action_history[-1]['type'] == 'filter' # checking if last action is filter
                # and len(obs_hist_all) - len(obs_hist) <= i <= len(obs_hist_all)
            ):
                r = -1.0 * self.humanity_coeff
                # print("same display is:" + str(i) + " len(dhist) is:" + str(len(dhist)))
                # reward_info.same_display_seen_already = r
                # reward_info.diversity = r
                break

            else:
                sim_vec.append(dist)
        else:
            # r = sum(sim_vec) / len(sim_vec) * 2
            r = min(sim_vec) * self.diversity_coeff
            # reward_info.diversity = r

        return r

    def take_final_branch(self, actions):
        actions_final = []

        for action in actions:
            action_type = action['type']
            actions_final.append(action)

            if action_type == "back":
                actions_final.pop()
                actions_final.pop()

        return actions_final

    def get_alternate_back_repeating_length(self):
        l = 0
        for i, action in enumerate(reversed(self.action_history)):
            if i % 2 == 1:
                if action['type'] != 'back':
                    l += 1
                else:
                    break
            elif i % 2 == 0:
                if action['type'] != 'back':
                    break
        return l

    def repeat_penalty(self):
        if self.action_history is None or len(self.action_history) < 2:
            return 0
        
        # Penalty for BACK on base state
        # if len(self.take_final_branch(self.action_history[:-1])) == 0:
        #     return -1.0
        
        if self.state_history[-2] == empty_env_state and self.action_history[-1]['type'] == 'back':
            return -1.0

        # if self.action_history[-1] != self.action_history[-2]:
        #     # Penalty for alternating BACK repeats
        #     if self.action_history[-1]['type'] == 'back':
        #         alt_repeat_len = self.get_alternate_back_repeating_length()
        #         return -1.5 * alt_repeat_len
        #     else:
        #         return 0
        
        # Penalty for repeating BACK
        # if self.action_history[-1]['type'] == 'back':
        #     return -0.5
        
        if self.action_history[-1] != self.action_history[-2]:
            # Penalty for alternating BACK repeats
            if self.action_history[-1]['type'] == 'back':
                alt_repeat_len = self.get_alternate_back_repeating_length()
                if alt_repeat_len > 1:
                    return -1.0 * alt_repeat_len
                else:
                    return 0.0
            else:
                return 0.0

        # Penalty for repeating FILTER/GROUP
        if self.action_history[-1] == self.action_history[-2]:
            if self.action_history[-1]['type'] in ['filter', 'group']:
                return -1.0
            else:
                return 0.0
        
        else:
            return 0

    def step(self, action):
        action = self.preprocess_action(action)
        self.action_history.append(action)
        self.take_action(action)

        obs = self.get_latest_observation()
        self.obs_history.append(obs)
        self.dhist.append(self.get_disp(self.state_history[-1]))

        self.step_num += 1
        done = True if (self.step_num >= self.max_steps or action['type'] == 'stop') else False
        # Return 0 reward and empty info dict
        return obs, 0, done, {}

    def get_observation_space(self):
        num_keys = len(self.dataset.keys)

        # Inverse of number of uniques, inverse of number of nulls and
        # normalized entropy of each column
        low_data_layer = np.zeros(3 * num_keys)
        high_data_layer = np.ones(3 * num_keys)

        # Whether an attribute is grouped or aggregated.
        # -1: Neither grouped nor aggregated
        # v \in [0, 1]: Aggregated, where v is normalized entropy of the
        #               of the attribute in the aggregated dataframe
        # 2: Grouped
        low_granularity_layer = np.full(num_keys, -1)
        high_granularity_layer = np.full(num_keys, 2)

        # Global properties: Inverse number of groups, Inverse mean size of
        # group, Inverse variance of group size
        low_global_props = np.zeros(3)
        high_global_props = np.ones(3)

        # Combining all the layers for a single observation
        low_single_obs = np.concatenate(
            [low_data_layer, low_granularity_layer, low_global_props])
        high_single_obs = np.concatenate(
            [high_data_layer, high_granularity_layer, high_global_props])

        # Repeating the combined layers STACK_OBS_NUM times
        low = np.tile(low_single_obs, utils.STACK_OBS_NUM)
        high = np.tile(high_single_obs, utils.STACK_OBS_NUM)

        return gym.spaces.Box(low, high, dtype='float')

    def get_action_space(self):
        return gym.spaces.MultiDiscrete(self.get_neural_network_segments())

    def get_filter_term(self, col, bin):

        last_state = self.state_history[-1]
        df = self.dataset.get_filtered_df(
            self.dataset.base_df, last_state.filtering)

        if len(df) == 0:
            return '<UNK>'

        # term_pos is a position between 0 and 1 used to do nearest neighbour search in
        # a frequency sorted list of filter terms from the queried col of the df
        term_pos = self.dataset.bins[bin] + (
            self.dataset.bins[bin + 1] - self.dataset.bins[bin]) * np.random.random()

        return self.get_nearest_neighbour_filter_term(df, col, term_pos)

    def get_nearest_neighbour_filter_term(self, df, col, term_pos):
        filter_terms = df[col].dropna().tolist()

        freq_dict = {}
        for term in filter_terms:
            if term not in freq_dict:
                freq_dict[term] = 0
            freq_dict[term] += 1

        freq_tups = sorted(list(freq_dict.items()), key=lambda kv: kv[1])
        freq_keys = np.array([kv[0] for kv in freq_tups])
        freq_vals = np.array([kv[1] for kv in freq_tups])
        freq_vals = freq_vals / np.sum(freq_vals)
        cumulative_freqs = np.cumsum(freq_vals)

        return freq_keys[np.abs(cumulative_freqs - term_pos).argmin()]

    def preprocess_action(self, action):

        ret_action = {
            "type": None,
            "col": None,
            "operator": None,
            "filter_term": None,
            "agg_option": None,
            "agg_func": None
        }

        filter_term_bin = action[3]

        # Round off action
        action = np.round(action).astype(int)

        ret_action['type'] = self.supported_actions[action[0]]

        if ret_action['type'] in ['back', 'stop']:
            return ret_action

        ret_action['col'] = self.dataset.keys[action[1]]

        if ret_action['type'] == 'filter':
            ret_action['operator'] = utils.SUPPORTED_OPERATORS[action[2]]
            try:
                ret_action['filter_term'] = self.get_filter_term(
                    ret_action['col'], action[3])
            except:
                ret_action['filter_term'] = '<UNK>'
            return ret_action

        if ret_action['type'] == 'group':
            if action[4] == 0:
                # Check if aggregation is over no column
                ret_action['agg_option'] = 'number'
                ret_action['agg_func'] = len
            else:
                # Aggregation column index is action[4] - 1
                ret_action['agg_option'] = self.dataset.keys[action[4] - 1]
                ret_action['agg_func'] = utils.SUPPORTED_AGG_FUNCS[action[5]]
            return ret_action

    def take_action(self, action):

        if action['type'] == 'back':
            if len(self.state_stack) > 1:
                self.state_stack.pop()
                new_state = self.state_stack[-1]
                self.state_history.append(new_state)
            else:
                new_state = self.state_stack[-1]
                self.state_history.append(new_state)

        elif action['type'] == 'stop':
            self.state_stack = [empty_env_state]
            self.state_history.append(empty_env_state)

        elif action['type'] == 'filter':
            filtering_tuple = FilteringTuple(
                field=action['col'], term=action['filter_term'], condition=action['operator'])
            new_state = self.state_history[-1]
            if filtering_tuple not in new_state.filtering:
                new_state = new_state.append_filtering(filtering_tuple)
            self.state_stack.append(new_state)
            self.state_history.append(new_state)

        elif action['type'] == 'group':
            new_state = self.state_history[-1]
            if action['col'] not in new_state.grouping:
                new_state = new_state.append_grouping(action['col'])
            agg_tuple = AggregationTuple(
                field=action['agg_option'], type=action['agg_func'])
            if agg_tuple not in new_state.aggregations:
                new_state = new_state.append_aggregations(agg_tuple)
            self.state_history.append(new_state)
            self.state_stack.append(new_state)

    def get_latest_observation(self):
        """
        Returns observation vector (based on last 3 displays) corresponding to 
        state_history.
        """
        latest_state_vector = self.dataset.calculate_state_vector(
            self.state_history[-1])
        if not self.obs_history:
            padding = np.zeros(2 * len(latest_state_vector))
            return np.concatenate([padding, latest_state_vector])

        last_obs = self.obs_history[-1]
        return np.concatenate([last_obs[-2 * len(latest_state_vector):], latest_state_vector])

    def get_neural_network_segments(self):
        num_actions = len(self.supported_actions)
        num_cols = len(self.dataset.keys)
        num_operators = len(utils.SUPPORTED_OPERATORS.keys())
        num_bins = len(self.dataset.bins) - 1
        # Extra option for aggregation without any columns
        agg_options = len(self.dataset.keys) + 1
        num_agg_funcs = len(utils.SUPPORTED_AGG_FUNCS.keys())

        return (num_actions, num_cols, num_operators, num_bins, agg_options, num_agg_funcs)

    def describe_action(self, action, preprocessed=False):
        if not preprocessed:
            action = self.preprocess_action(action)
        if action['type'] == 'back':
            return 'BACK'
        if action['type'] == 'stop':
            return 'STOP'
        if action['type'] == 'filter':
            return f'FILTER {action["col"]} {action["operator"]} {action["filter_term"]}'
        if action['type'] == 'group':
            return f'GROUP BY {action["col"]} AGGREGATE {action["agg_option"]} WITH FUNCTION {action["agg_func"]}'

class MultiDatasetEnv():

    def __init__(self, dataset_paths, max_steps=12):

        super(MultiDatasetEnv).__init__()
        
        self.envs = [AnalysisEnv(path, max_steps) for path in dataset_paths]
        self.num_envs = len(self.envs)
        self.curr = 0

    def reset(self):
        self.curr = (self.curr + 1) % self.num_envs
        return self.envs[self.curr].reset()

    def step(self, action):
        return self.envs[self.curr].step(action)

    def __getattr__(self, name):
        return getattr(self.envs[self.curr], name)