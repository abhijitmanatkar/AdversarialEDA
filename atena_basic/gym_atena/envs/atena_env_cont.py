import json
import math
import logging
import random
from copy import deepcopy
from functools import lru_cache

import numpy as np

from cachetools import LRUCache

import gym
from gym import spaces

from scipy.stats import entropy
import scipy



from atena_basic.gym_atena.lib.tokenization import tokenize_column, get_nearest_neighbor_token, gep
#from atena_basic.gym_atena.reactida.utils.utilities import Repository
from atena_basic.gym_atena.reactida.utils.distance import display_distance
import atena_basic.gym_atena.lib.helpers as ATENAUtils
from atena_basic.gym_atena.lib.helpers import (
    normalized_sigmoid_fkt,
    get_aggregate_attributes,
    empty_env_state,
    AggregationTuple,
    FilteringTuple,
)

from atena_basic.arguments import ArchName, FilterTermsBinsSizes
from atena_basic.Utilities.Collections.Counter_Without_Nans import CounterWithoutNanKeys
import atena_basic.Utilities.Configuration.config as cfg

logger = logging.getLogger(__name__)


class ATENAEnvCont(gym.Env):
    """The main Atena environment class


    Attributes:
        max_steps (int): The number of steps in each episode
        repo (Obj): The repository of human session
        observation_space (Obj): The observation space (Box)
        action_space (Obj): The action space (Box)
        ret_df (bool): Wheter to return the dataframes in each observation
        data (obj): Dataframe containing the current dataset of the episode
        history (list): captures all "state" dictionaries in the current episode, each one is equivalent to a query
        ahist (list): a list of actions performed thus far in the episode
        dhist (list): a list of the corresponding result-displays of the actions

    """

    LOG_INTERVAL = 100

    NUM_OF_EPISODES = 0

    metadata = {
        'render.modes': ['human'],
    }

    # cache where key is (dataset_num, state) and value is the tuple (observation, display, dfs)
    STATE_DF_HISTORY = None

    # cache where key is (dataset_num, state, col) and value is the tuple (sorted_token_frequency_pairs_lst, sorted_frequencies_lst)
    COL_TOKENIZATION_HISTORY = None

    # cache where key is (dataset_number, state1, state2) and value is the distance between the displays they are representing
    # Note that you key should be ordered such that str(state1) <= str(state2) since the distance is symmetric and we don't
    # want duplicates
    STATES_DISP_DISTANCE_HISTORY = None

    # architecture
    arch = ArchName(cfg.arch)

    # length of a single display
    len_single_display_vec = None

    # a static variable env for various uses so that we won't have to create a new environment
    # which is expensive
    static_env = None

    def __init__(self, max_steps=cfg.MAX_NUM_OF_STEPS, ret_df=False, gradual_training=False):
        """When initializing the environment class the following happens:
        (1) The data files and the human session repositories are loaded
        (2) The action space and observation space are devined

        Args:
            max_steps (int): The maximum number of steps in an episodes
            env_prop (BasicEnvProp): Environment properties of allowed actions and explored dataset
        """
        self.env_prop = gep.update_global_env_prop_from_cfg()

        # reset caches if needed
        if self._is_caches_reset_needed():
            self.reset_caches()

        # (0) Initialize some attributes, others are initialized in the reset() method
        self.gradual_training = gradual_training
        self.max_steps = max_steps
        self.ret_df = ret_df
        self.env_dataset_prop = self.env_prop.env_dataset_prop

        # (1) Loading data files and session reposiotires. Note that the class Repository is taken from REACT
        self.repo = self.env_dataset_prop.repo

        # (2.a) Define the action space:
        #        0) action_type:            back[0], filter[1], group[2]
        #        1) col_id:                 [0..num_of_columns-1]
        #        2) filter_operator         [LT, GT, etc..]
        #        3) filter_term:          taken from a fixed list of tokens
        #        4) aggregation column_id:  [0..num_of_columns - 1]
        #        5) aggregation function:       [mean, count, etc.]

        # self.action_space = spaces.MultiDiscrete([ACTION_TYPES_NO,COLS_NO,FILTER_OPS,FILTER_TERMS_NO,COLS_NO, AGG_FUNCS_NO])
        # self.action_space=spaces.Box(low=np.zeros(6)-0.49,high=np.array([ACTION_TYPES_NO, COLS_NO, FILTER_OPS, FILTER_TERMS_NO, COLS_NO, AGG_FUNCS_NO])-0.51,dtype='float32')
        # self.action_space = spaces.Box(low=np.zeros(6) - self.env_prop.ACTION_RANGE / 2,
        #                                high=np.zeros(6) + self.env_prop.ACTION_RANGE / 2, dtype='float32')
        # self.action_space = spaces.MultiDiscrete([self.env_prop.ACTION_TYPES_NO,self.env_dataset_prop.COLS_NO,
        #                                           self.env_prop.FILTER_OPS,self.env_dataset_prop.FILTER_TERMS_NO,
        #                                           self.env_dataset_prop.COLS_NO, self.env_prop.AGG_FUNCS_NO])
        self.action_space = spaces.Box(low=np.zeros(6) - 0.49,
                                       high=np.array([self.env_prop.ACTION_TYPES_NO,self.env_dataset_prop.COLS_NO,
                                                     self.env_prop.FILTER_OPS,self.env_dataset_prop.FILTER_TERMS_NO,
                                                     self.env_dataset_prop.AGG_COLS_NO, self.env_prop.AGG_FUNCS_NO]) - 0.51, dtype='float')

        # (2.b) Define the observation space:
        # z_step_number(optional): a binary 1-based(!) vector in size of self.max_steps with binary values
        # s.t. there is a single bit on corresponding to the current step number starting from 1
        # z: {num of Unique, num of nulls, normalized entropy} for each column
        # z2: {Grouped or aggregated state} for each column:
        #     -1 if none, [0-1] if aggregated (value means the NVE), and 2 if grouped
        # z3: {num of groups, mean group size, size variance}
        z = np.zeros(len(self.env_dataset_prop.KEYS) * 3)

        z2 = np.full(len(self.env_dataset_prop.KEYS), -1)
        z3 = np.zeros(3)

        low = np.tile(np.concatenate([z, z2, z3]), cfg.stack_obs_num)
        high = np.tile(np.concatenate([np.ones(len(z)), np.full(len(self.env_dataset_prop.KEYS), 2), np.ones(3)]), cfg.stack_obs_num)

        if cfg.obs_with_step_num:
            z_step_number = np.zeros(self.max_steps)
            low = np.concatenate([z_step_number, low])
            high = np.concatenate([np.ones(len(z_step_number)), high])
        # else:
        #    low = np.concatenate([z, z2, z3])
        #    high = np.concatenate([np.ones(len(z)), np.full(len(self.env_dataset_prop.KEYS), 2), np.ones(3)])

        self.observation_space = spaces.Box(low, high, dtype='float')

        ATENAEnvCont.len_single_display_vec = len(z) + len(z2) + len(z3)

        # Print dataset name if one is chosen
        if cfg.dataset_number is not None:
            print(self.repo.file_list[cfg.dataset_number])

        self._log = True

    @classmethod
    def reset_caches(cls):
        cls.STATE_DF_HISTORY = None
        if cfg.cache_dfs_size > -1:
            cls.STATE_DF_HISTORY = LRUCache(maxsize=cfg.cache_dfs_size)

        cls.COL_TOKENIZATION_HISTORY = None
        if cfg.cache_tokenization_size > -1:
            cls.COL_TOKENIZATION_HISTORY = LRUCache(maxsize=cfg.cache_tokenization_size)

        cls.STATES_DISP_DISTANCE_HISTORY = None
        if cfg.cache_tokenization_size > -1:
            cls.STATES_DISP_DISTANCE_HISTORY = LRUCache(maxsize=cfg.cache_distances_size)

    def _is_caches_reset_needed(self):
        if (self.STATES_DISP_DISTANCE_HISTORY is None
                and self.STATE_DF_HISTORY is None
                and self.COL_TOKENIZATION_HISTORY is None):
            return True
        return False

    ##############################

    # Internal Functions

    ##############################
    def translate_action(self, act_vector, filter_by_field=True, filter_term=None):
        """This function translate an action vector into a human-readable action

        Args:
            act_vector (numpy array): the action vector

        Returns:
            A human readable string that corresponds to the action vector
        """
        if type(act_vector) is not list:
            act_vector = self.cont2dis(act_vector)
        rtype = self.env_prop.OPERATOR_TYPE_LOOKUP.get(act_vector[0])
        if rtype == "back":
            return "Back"
        elif rtype == "filter":
            col = self.data.columns[act_vector[1]]
            cond = self.env_prop.INT_OPERATOR_MAP_ATENA.get(act_vector[2])
            if not cond:
                op_num = act_vector[2]
                if op_num in [6, 7, 8]:  # op_num == 6:
                    cond = 'str.contains'
                elif op_num == 7:
                    cond = 'str.startswith'
                elif op_num == 8:
                    cond = 'str.endswith'
            if not filter_by_field:
                term = self.env_dataset_prop.FILTER_LIST[act_vector[3]]
            else:
                """filter_field_list = FILTER_BY_FIELD_DICT.get(col)
                if filter_field_list and len(filter_field_list)-1 >= act_vector[3]:
                    term = filter_field_list[act_vector[3]]
                else:
                    term = '<UNK>'"""
                term = filter_term
            return "Filter on Column '%s', using condition '%s', with term '%s'" % (col, str(cond), term)
        elif rtype == "group":
            col = self.data.columns[act_vector[1]]
            agg_col = self.data.columns[act_vector[4]]
            agg_func = self.env_prop.AGG_MAP_ATENA.get(act_vector[5])
            return "Group on Column '%s' and aggregate with '%s' on the column '%s'" % (col, str(agg_func), agg_col)
        else:
            raise NotImplementedError

    @lru_cache(maxsize=2048)
    def get_exponential_filter_term_bins(self, num_of_rows, num_of_bins):
        """
        Create `num_of_bins` bins for filter terms such that bins width grows exponentially with frequency of these bins
        Args:
            num_of_rows:
            num_of_bins:

        Returns:

        """
        assert num_of_bins >= 1
        if num_of_rows == 0:
            return [0.0] * num_of_bins + [1.0]
        B_minus1 = num_of_bins - 1
        x = (num_of_rows ** (1 / B_minus1))
        single_row_contribution = 1 / num_of_rows
        bins = [0] + [round(x ** i) / num_of_rows for i in range(num_of_bins)]
        for i in range(1, len(bins)):
            if bins[i] <= bins[i - 1]:
                bins[i] = bins[i - 1] + single_row_contribution
        return bins

    @staticmethod
    def _param_softmax_idx_to_action_helper(idx):
        raise NotImplementedError

    def param_softmax_idx_to_action(self, idx):
        """
        Maps an index that represents one off all possible discrete actions in the environment
        to a legal action in the environment (i.e. a vector of size action_space)
        Args:
            idx (int): index of an entry in the output vector of an architecture
            of type PARAM_SOFTMAX

        Returns:

        """
        result = np.zeros(6, dtype=np.float32)
        (action_type, parameters) = self.env_prop.MAP_PARAMETRIC_SOFMAX_IDX_TO_DISCRETE_ACTION[idx]

        result[0] = action_type
        action_type_string = self.env_prop.OPERATOR_TYPE_LOOKUP[action_type]
        if action_type_string == "back":
            pass
        elif action_type_string == "filter":
            result[1] = parameters[0]

            result[2] = (parameters[1] + 1) * 3 - 1

            filter_terms_bin_sizes = FilterTermsBinsSizes(cfg.bins_sizes)
            if filter_terms_bin_sizes is FilterTermsBinsSizes.EQUAL_WIDTH:
                bin_size = 1 / (self.env_prop.DISCRETE_FILTER_TERM_BINS_NUM - 1)
                result[3] = parameters[2] * bin_size - 0.5 + random.uniform(-bin_size / 2, bin_size / 2)
            elif filter_terms_bin_sizes is FilterTermsBinsSizes.CUSTOM_WIDTH:
                lower_bin_edge = self.env_prop.bins[parameters[2]]
                upper_bin_edge = self.env_prop.bins[parameters[2] + 1]
                bin_size = upper_bin_edge - lower_bin_edge
                result[3] = lower_bin_edge - 0.5 + random.uniform(0, bin_size)
            elif filter_terms_bin_sizes is FilterTermsBinsSizes.EXPONENTIAL:
                num_of_rows = self.num_of_fdf_rows_hist[-1]
                bins = self.get_exponential_filter_term_bins(num_of_rows, cfg.exponential_sizes_num_of_bins)
                lower_bin_edge = bins[parameters[2]]
                upper_bin_edge = bins[parameters[2] + 1]
                bin_size = upper_bin_edge - lower_bin_edge
                result[3] = lower_bin_edge - 0.5 + random.uniform(0, bin_size)
            else:
                raise NotImplementedError

        elif action_type_string == "group":
            result[1] = parameters[0]
        else:
            raise ValueError("action_type should refer to back filter or group")

        return result

    @staticmethod
    def static_param_softmax_idx_to_action(idx):
        """
        Maps an index that represents one off all possible discrete actions in the environment
        to a legal action in the environment (i.e. a vector of size action_space)
        Args:
            idx (int): index of an entry in the output vector of an architecture
            of type PARAM_SOFTMAX

        Returns:

        """
        result, _, _ = ATENAEnvCont._param_softmax_idx_to_action_helper(idx)
        return result


    @staticmethod
    def cont2dis(c_vector):
        """This function discretizes (rounds) a continuous (float) action vector

        Args:
            C_vector (numpy array): a continuous (float) action vector

        Returns:
            A vector of discrete integer representing the actions
        """
        return list(np.array(np.round(c_vector), dtype=np.int))

    @staticmethod
    def _is_empty_display(obs):
        start_of_cur_display_suv_vec = len(obs) - ATENAEnvCont.len_single_display_vec
        return np.array_equal(
            obs[start_of_cur_display_suv_vec:3 + start_of_cur_display_suv_vec],
            [0, 1, 0])

    @staticmethod
    def _is_empty_groupings(obs):
        return np.array_equal(obs[-3:], [0, 0, 0])

    def reset(self, dataset_number=None):
        """This function starts a new episode. It performs the following steps:
            (1) Randomly choose a dataset
            (2) Initialize the history lists: history,ahist,dhist

            :param dataset_number: if set, the dataset with this nubmer will be
            loaded, else a random dataset will be chosen
            :return: The first observation vector describing the chosen dataset
        """
        # resample a seed:
        random.seed()
        scipy.random.seed()

        self.NUM_OF_EPISODES += 1
        self.step_num = 0
        if self.gradual_training:
            self.max_steps = random.randint(2, max(3, int(self.NUM_OF_EPISODES / 2500)))

        self._log = True if self.NUM_OF_EPISODES % self.LOG_INTERVAL == 0 else False

        # (1) Choose a dataset:
        if cfg.dataset_number is not None:
            dataset_number = cfg.dataset_number
        elif dataset_number is None:
            dataset_number = np.random.randint(len(self.repo.data))
        self.dataset_number = dataset_number
        if self._log:
            logger.info(f"Dataset number chosen is {dataset_number}, dataset name is {self.repo.file_list[dataset_number]}")

        self.data = self.repo.data[dataset_number][self.env_dataset_prop.KEYS]

        # Initialize history of states lists:
        empty_state = empty_env_state
        # history is a stack of state (back actions pop elements from list)
        self.history = [empty_state]
        # states_history is a list of all states during session
        self.states_hisotry = [empty_state]
        # stack for history of observations (back actions pop elements from list)
        self.obs_hist = []
        # obs_hist_all is a list of all observations during the session
        self.obs_hist_all = []

        # Calculate the display and the observation vector:
        obs, disp, _ = self.env_prop.calc_display_vector(self.data,
                                           empty_state,
                                           memo=self.STATE_DF_HISTORY,
                                           dataset_number=self.dataset_number,
                                           step_number=self.step_num,
                                           states_hist=self.history,
                                           obs_hist=self.obs_hist,
                                           len_single_display_vec=self.len_single_display_vec
                                           )

        # Display history will contain the first display
        self.dhist = [disp]
        self.ahist = []
        self.obs_hist = [obs]
        self.obs_hist_all = [obs]
        self.filter_terms_hist = []
        self.num_of_rows_hist = [len(self.data)]  # Number of rows or groups (if grouped) in the current display
        self.num_of_fdf_rows_hist = [len(self.data)]  # Number of rows in the current display fdf
        # Number of rows if the current action is group and the previous action is filter if we did not take the
        # filter action and make an immediate group action instead
        self.num_of_immediate_action_rows_lst = [None]

        # determine if we are in the middle of a point inside the session where the
        # a group operation was empty, and the number of operations in this subsession
        self.in_the_middle_of_empty_grouping = False
        self.in_the_middle_of_empty_grouping_steps = 0

        assert self.observation_space.contains(obs)
        return obs

    @property
    def arch(self):
        return ArchName(cfg.arch)

    @staticmethod
    def get_filtered_only_or_grouped_data_frame(dfs):
        '''
        get a tuple of dataframes (dfs=[fdf, adf] as is returned from calc_display_vector())
        and returns the first dataframe if no grouping is indicated, else the second.
        Moreover, a Boolean value indicating whether a grouping was indicated is returned.

        Note: this function asssumed that if dfs[1] is None there is no grouping (which is not
        only the case. Whether a dataframe is grouped can only be determined by the state dictionary)
        :param dfs:
        :return:
        '''
        is_grouping = True if dfs[1] is not None else False
        if is_grouping:
            df_dt = dfs[1]  # df <- adf (df is filtered, grouped and aggregated)
        else:
            df_dt = dfs[0]  # df has filtering only
        return df_dt, is_grouping


    def reward(self, obs, dfs, state, no_history_for_back):
        """This function determines the reward for an observation. It perfoms the following:
         (0) Check if "Done", i.e. the maximum number of steps is reached
         (1) Determines the reward (punishments + diversity based positive reward)

        Args:
            obs (obj): the observation vector

        Returns:
            bool: Is "done"
            float: A reward
            reward_info: dict of details about each reward component
        """
        #reward_info = StepReward()

        # (0) check if done:
        self.step_num += 1
        done = self.step_num >= self.max_steps

        return done, 0, {}

    def step(self, action, compressed=False, filter_by_field=True, continuous_filter_term=True, filter_term=None):
        """This function processes an action:
         (1) deconstruct the action to its parameters
         (2) executes the action: It computes a rolling "state" dictionary, comprising filtering,grouping and aggregations
         (3) Calculate the display vector
         (4) Update the history lists
         (5) Determine the reward

        Args:
            action (obj): Action vector

        Returns:
            obj: observation vector
            float: reward score
            bool: if done
            dict: information dict

        """

        # (1) Deconstruct the action:
        prev_action = action
        action, filter_by_field = self.action_to_vec(action, compressed, continuous_filter_term, filter_by_field)
        operator_type = self.env_prop.OPERATOR_TYPE_LOOKUP.get(action[0])
        col = self.env_dataset_prop.KEYS[action[1]]
        no_history_for_back = False

        # (2) Executing an action by incrementing the state dictionary:

        if operator_type == 'back':
            # If back: pop the last element from the history and use it as the current state
            if len(self.history) > 1:
                self.obs_hist.pop()
                self.history.pop()
                new_state = self.history[-1]
            else:
                new_state = empty_env_state
                no_history_for_back = True

        elif operator_type == 'filter':
            # If filter: add the filter condition to the list of filters in the prev state
            condition = action[2]
            if filter_term is not None:
                pass
            elif not filter_by_field:
                filter_term = self.env_dataset_prop.FILTER_LIST[action[3]]
            else:
                """filter_field_list = FILTER_BY_FIELD_DICT.get(col)
                if filter_field_list and len(filter_field_list)-1 >= action[3]:
                    filter_term = filter_field_list[action[3]]
                else:
                    filter_term = '<UNK>'"""
                filter_term = self.compute_nearest_neighbor_filter_term(action, col)

            filt_tpl = FilteringTuple(field=col, term=filter_term, condition=condition)

            new_state = self.history[-1]
            new_state = new_state.append_filtering(filt_tpl)
            self.history.append(new_state)

        elif operator_type == 'group':
            # add to the grouping and aggregations lists of the prev state:
            new_state = self.history[-1]
            if col not in new_state["grouping"]:
                new_state = new_state.append_grouping(col)
            agg_tpl = AggregationTuple(field=self.env_dataset_prop.AGG_KEYS[action[4]], type=action[5])
            if agg_tpl not in new_state["aggregations"]:
                new_state = new_state.append_aggregations(agg_tpl)
            self.history.append(new_state)
        else:
            raise Exception("unknown operator type: " + operator_type)

        self.states_hisotry.append(new_state)

        # (3) calculate observation and update dictionaries:
        obs, disp, dfs = self.env_prop.calc_display_vector(self.data,
                                             new_state,
                                             memo=self.STATE_DF_HISTORY,
                                             dataset_number=self.dataset_number,
                                             step_number=self.step_num,
                                             states_hist=self.history,
                                             obs_hist=self.obs_hist,
                                             len_single_display_vec=self.len_single_display_vec
                                             )

        # (4) Update the history lists:
        self.dhist.append(disp)
        self.ahist.append(action)
        self.obs_hist_all.append(obs)
        if operator_type != 'back':
            self.obs_hist.append(obs)
        
        # Upadte hists needed for Snorkel
        self.filter_terms_hist.append(filter_term)
        self.num_of_rows_hist.append(len(self.get_previous_df()))
        self.num_of_fdf_rows_hist.append(len(self.get_previous_fdf()))
        self.num_of_immediate_action_rows_lst.append(self.get_num_of_immediate_action_rows_after_filter())

        # (5) Get the reward
        done, reward, reward_info = self.reward(obs, dfs, new_state, no_history_for_back)

        # validate that obs type is np.float32
        assert obs.dtype == np.float32, 'obs.dtype must be np.float32'

        if done and self._log:
            logger.info('actions:%s' % str(self.ahist))
            # logger.info('states:%s' % str(self.dhist))

        if not self.ret_df:
            dfs = None

        return obs, reward, done, {"raw_action": action,
                                   "action": self.translate_action(action, filter_by_field, filter_term),
                                   "raw_display": dfs,
                                   "reward_info": reward_info,
                                   "state": new_state,
                                   "filter_term": filter_term,
                                   }

    def action_to_vec(self, action, compressed=False, continuous_filter_term=True, filter_by_field=True):
        if self.arch is ArchName.FF_PARAM_SOFTMAX or self.arch is ArchName.FF_SOFTMAX:
            compressed = False
            action = self.param_softmax_idx_to_action(action)
        if compressed:
            # if self._log:
            # logger.info('compressed action:%s' % str(action))
            action = self.env_prop.compressed2full_range(action, continuous_filter_term)
        action_filter_term = action[3]
        action = self.cont2dis(action)
        if continuous_filter_term:
            action[3] = action_filter_term + 0.5
        return action, filter_by_field

    def compute_nearest_neighbor_filter_term(self, action, col):
        prev_state = self.history[-1]
        prev_state_without_group_and_agg = prev_state.reset_grouping_and_aggregations()
        if self.COL_TOKENIZATION_HISTORY is None or (
                (self.dataset_number, prev_state_without_group_and_agg,
                 col) not in self.COL_TOKENIZATION_HISTORY):
            prev_fdf = self.get_previous_fdf()
            sorted_by_freq_token_frequency_pairs, frequencies = tokenize_column(prev_fdf, col)

            # saving to cache
            # Note: we use the key prev_state_without_group_and_agg and not prev_state
            # to increase caching hit rate and due to the fact the both cases should have the same
            # column tokenization
            if self.COL_TOKENIZATION_HISTORY is not None:
                self.COL_TOKENIZATION_HISTORY[(self.dataset_number,
                                               prev_state_without_group_and_agg,
                                               col)] = (sorted_by_freq_token_frequency_pairs, frequencies)
        else:
            sorted_by_freq_token_frequency_pairs, frequencies = self.COL_TOKENIZATION_HISTORY[
                (self.dataset_number,
                 prev_state_without_group_and_agg,
                 col)]
        filter_term = get_nearest_neighbor_token(sorted_by_freq_token_frequency_pairs, frequencies, action[3])
        return filter_term

    def get_num_of_immediate_action_rows_after_filter(self):
        if len(self.ahist) < 2:
            return None
        cur_action = self.ahist[-1]
        prev_action = self.ahist[-2]
        cur_action_type_str = self.env_prop.OPERATOR_TYPE_LOOKUP[cur_action[0]]
        prev_action_type_str = self.env_prop.OPERATOR_TYPE_LOOKUP[prev_action[0]]

        if prev_action_type_str == 'filter':
            if cur_action_type_str == 'back':
                return None
            else:
                # Create step before filter but including group
                state_before_filter = self.states_hisotry[-3]
                if cur_action_type_str == 'group':
                    grouped_column = self.env_dataset_prop.GROUP_COLS[cur_action[1]]
                    if grouped_column not in state_before_filter["grouping"]:
                        state_before_filter = state_before_filter.append_grouping(grouped_column)
                    agg_tpl = AggregationTuple(field=self.env_dataset_prop.AGG_KEYS[cur_action[4]], type=cur_action[5])
                    if agg_tpl not in state_before_filter["aggregations"]:
                        state_before_filter = state_before_filter.append_aggregations(agg_tpl)
                elif cur_action_type_str == 'filter':
                    filtered_column = self.env_dataset_prop.FILTER_COLS[cur_action[1]]
                    filt_tpl = FilteringTuple(field=filtered_column, term=self.filter_terms_hist[-1],
                                              condition=cur_action[2])
                    state_before_filter = state_before_filter.append_filtering(filt_tpl)

                dfs = self.env_prop.get_state_dfs(self.data,
                                    state_before_filter,
                                    memo=self.STATE_DF_HISTORY,
                                    dataset_number=self.dataset_number,
                                    )

                df_dt, is_grouping = self.get_filtered_only_or_grouped_data_frame(dfs)
                return len(df_dt)

        else:
            return None


    def get_previous_fdf(self, past_steps=1):
        prev_state = self.states_hisotry[-1 * past_steps]

        dfs = self.env_prop.get_state_dfs(self.data,
                            prev_state,
                            memo=self.STATE_DF_HISTORY,
                            dataset_number=self.dataset_number,
                            )

        return dfs[0]

    def get_previous_df(self, past_steps=1):
        """

        Args:
            past_steps: number of steps to go back in history, starting from 1 (!)

        Returns:

        """
        prev_state = self.states_hisotry[-1 * past_steps]
        dfs = self.env_prop.get_state_dfs(self.data,
                            prev_state,
                            memo=self.STATE_DF_HISTORY,
                            dataset_number=self.dataset_number,
                            )

        df_dt, is_grouping = self.get_filtered_only_or_grouped_data_frame(dfs)

        return df_dt

    def render(self, mode='human', close=False):
        if close:
            return None
        self.ret_df = True
        # print("I will return the df in the info...")
        return None

    @classmethod
    def get_static_env(cls, max_steps):
        """
        Create a static variable env for various uses so that we won't have to create a new environment
        which is expensive
        Args:
            max_steps: Number of steps in session

        Returns:

        """
        if cls.static_env is None:
            cls.static_env = cls(max_steps=max_steps)
        else:
            cls.static_env.max_steps = max_steps
        return cls.static_env

    GET_SESSIONS_HISTS_CACHE = LRUCache(maxsize=300)

    @classmethod
    def get_sessions_hists(cls, actions_lst,
                           dataset_number,
                           compressed=False,
                           filter_by_field=True,
                           continuous_filter_term=True,
                           filter_terms_lst=None
                           ):
        """
        Returns a 2-tuple (`dhist`, `ahist`) where `dhist` is the displays histogram and `ahsit` is a histogram of actions,
        when running a session containing the actions in `actions_lst` on dataset `dataset_number`
        Args:
            actions_lst:
            dataset_number:
            compressed:
            filter_by_field:
            continuous_filter_term:
            filter_terms_lst: Note: len(filter_terms_lst) == len(actions_lst)

        Returns:

        """
        # Change architecture to gaussian
        prev_arch = cfg.arch
        cfg.arch = ArchName.FF_GAUSSIAN.value

        actions_tuple = ATENAUtils.lst_of_actions_to_tuple(actions_lst)
        if (dataset_number, actions_tuple) in cls.GET_SESSIONS_HISTS_CACHE:
            return cls.GET_SESSIONS_HISTS_CACHE[(dataset_number, actions_tuple)]

        if filter_terms_lst is not None:
            assert len(actions_lst) == len(filter_terms_lst)

        env = cls.get_static_env(max_steps=len(actions_lst))
        info_hist = []
        env.render()
        env.reset(dataset_number)
        for i, a in enumerate(actions_lst):
            s_, _, done, info = env.step(a,
                                         compressed=compressed,
                                         filter_by_field=filter_by_field,
                                         continuous_filter_term=continuous_filter_term,
                                         filter_term=None if filter_terms_lst is None else filter_terms_lst[i]
                                         )  # make step in environment
            s = s_
            info_hist.append(info)
            if done:
                break
        dhist = env.dhist
        ahist = env.ahist
        cls.GET_SESSIONS_HISTS_CACHE[(dataset_number, actions_tuple)] = (dhist, ahist, info_hist)

        # Restore architecture
        cfg.arch = prev_arch
        return dhist, ahist, info_hist

    @classmethod
    def get_greedy_max_reward_actions_lst(cls,
                                          dataset_number,
                                          episode_length,
                                          kl_coeff,
                                          compaction_coeff,
                                          diversity_coeff,
                                          humanity_coeff,
                                          verbose=False):
        """
        Returns a 2-tuple ('actions_lst`, `total_reward`), where `actions_lst` is the list of size `epsiode_length`
        of greedy actions on dataset `dataset_nunmber` using the given coefficients for the rewards
        Args:
            dataset_number:
            episode_length:
            kl_coeff:
            compaction_coeff:
            diversity_coeff:
            humanity_coeff:
            verbose:

        Returns:

        """
        cfg.kl_coeff = kl_coeff
        cfg.compaction_coeff = compaction_coeff
        cfg.diversity_coeff = diversity_coeff
        cfg.humanity_coeff = humanity_coeff

        cur_env = cls(max_steps=episode_length)
        cur_env.render()
        cur_env.reset(dataset_number)
        cur_env.arch = ArchName.FF_PARAM_SOFTMAX

        actions_lst = []
        total_reward = 0
        for step in range(episode_length):
            max_reward = -math.inf
            max_action = None
            max_env = None

            for act_idx in cur_env.env_prop.MAP_PARAMETRIC_SOFMAX_IDX_TO_DISCRETE_ACTION.keys():
                next_env = deepcopy(cur_env)
                obs, reward, done, info = next_env.step(act_idx)
                action_vec = info["raw_action"]
                action_vec[3] -= 0.5
                if reward > max_reward:
                    max_reward = reward
                    max_action = action_vec
                    max_env = next_env
            if verbose:
                print(max_reward)
            cur_env = max_env
            actions_lst.append(max_action)
            total_reward += max_reward
        if verbose:
            print(actions_lst)
            print(total_reward)
        return actions_lst, total_reward

    @classmethod
    def debug_actions(cls, actions_lst, dataset_number=None, compressed=False, filter_by_field=False,
                      continuous_filter_term=False, displays=False,
                      kl_coeff=cfg.kl_coeff,
                      compaction_coeff=cfg.compaction_coeff,
                      diversity_coeff=cfg.diversity_coeff,
                      humanity_coeff=cfg.humanity_coeff,
                      ):

        cfg.kl_coeff = kl_coeff
        cfg.compaction_coeff = compaction_coeff
        cfg.diversity_coeff = diversity_coeff
        cfg.humanity_coeff = humanity_coeff
        cfg.analysis_mode = True

        env = cls(max_steps=len(actions_lst))
        info_hist = []
        env.render()
        env.reset()
        for i in range(1):
            # cls.reset_caches()
            if isinstance(env, ATENAEnvCont):
                s = env.reset(dataset_number)
            elif isinstance(env, gym.wrappers.Monitor):
                s = env.env.env.reset(dataset_number)
            else:
                s = env.env.reset(dataset_number)
            r_sum = 0
            for a in actions_lst:
                print(a)
                s_, r, done, info = env.step(a,
                                             compressed=compressed,
                                             filter_by_field=filter_by_field,
                                             continuous_filter_term=continuous_filter_term)  # make step in environment
                print(str(info["action"]))
                print("reward:" + str(r))
                print(str(info["reward_info"]))
                print()
                info_hist.append((info, r))
                s = s_
                r_sum += r
                print("")
                if displays:
                    f_df, a_df = info["raw_display"]
                    if a_df is not None:
                        print(a_df)
                    else:
                        print(f_df)
                print("---------------------------------------------------")
                if done:
                    break
        print(f"Total reward: {r_sum}")
        return info_hist, r_sum


class StepReward(object):
    """class that contains the reward_info for a single step"""

    def __init__(self,
                 empty_display=0,
                 empty_groupings=0,
                 same_display_seen_already=0,
                 back=0,
                 diversity=0,
                 interestingness=0,
                 kl_distance=0,
                 compaction_gain=0,
                 humanity=0
                 ):
        self.empty_display = empty_display
        self.empty_groupings = empty_groupings
        self.same_display_seen_already = same_display_seen_already
        self.back = back
        self.diversity = diversity
        self.interestingness = interestingness
        self.kl_distance = kl_distance
        self.compaction_gain = compaction_gain
        self.humanity = humanity

        self.is_back = False
        self.is_data_driven = False

    @property
    def is_same_display_seen_already(self):
        return self.same_display_seen_already < 0

    def items(self):
        """return (reward_type_str, reward_type_val) pairs"""
        result = deepcopy(self.__dict__)
        result.pop('is_back')
        result.pop('is_data_driven')
        return result.items()

    def values(self):
        """return reward_type_vals"""
        result = deepcopy(self.__dict__)
        result.pop('is_back')
        result.pop('is_data_driven')
        return result.values()

    def __repr__(self):
        return self.items().__repr__()


if __name__ == '__main__':
    '''actions_lst = [
    [1, 4, 3, 9, 9, 2],
    [2, 1, 3, 23, 5, 0],
    [1, 4, 5, 19, 5, 2],
    [1, 6, 1, 15, 7, 0],'''
    actions_lst = [
        [2, 1, 1, 0.3221167325973511, 0, 0],  # 1
        [2, 6, 4, 0.6, 0, 0],  # 2
        [0, 2, 8, 0.21705351769924164, 0, 0],  # 3
        [2, 2, 8, -0.49803635857105255, 0, 0],  # 4
        [2, 6, 7, -0.46546758365631, 0, 0],  # 5
        [1, 6, 4, 0.38086163997650146, 0, 0],  # 6
        [0, 3, 1, -0.40, 0, 0],  # 7
        [0, 6, 8, 0.444186806678772, 0, 0],  # 9
        [0, 6, 8, 0.444186806678772, 0, 0],  # 8
        [2, 5, 4, 0.6415030360221863, 0, 0],  # 10
        [2, 6, 4, 0.5431699156761169, 0, 0],  # 11
        [2, 2, 4, 0.8115951418876648, 0, 0],  # 12
    ]
    actions_lst = [np.array(act) for act in actions_lst]
    ATENAEnvCont.debug_actions(actions_lst, 3, compressed=False, filter_by_field=True, continuous_filter_term=True,
                               displays=True,
                               kl_coeff=2.5,
                               compaction_coeff=2.9,
                               diversity_coeff=6.0,
                               humanity_coeff=4.8,
                               # kl_coeff=1,
                               # compaction_coeff=1,
                               # diversity_coeff=1,
                               # humanity_coeff=1,
                               )
    # ATENAEnvCont.get_greedy_max_reward_actions_lst(dataset_number=1,
    #                                                episode_length=12,
    #                                                kl_coeff=3.2,
    #                                                compaction_coeff=2.0,
    #                                                diversity_coeff=6.5,
    #                                                humanity_coeff=4.5,
    #                                                verbose=True
    #                                                )

    actions_lst = [
        [2, 6, 2, 0.3221167325973511, 0, 0],  # 1
        [2, 5, 1, -0.445, 0, 0],  # 2
    ]

    actions_lst2 = [
        [2, 5, 1, -0.445, 0, 0],  # 1
        [2, 6, 2, 0.3221167325973511, 0, 0],  # 2
    ]

    # actions_lst = [np.array(act) for act in actions_lst]
    # actions_lst2 = [np.array(act) for act in actions_lst2]
    # dhist1, ahist1 = ATENAEnvCont.get_sessions_hists(actions_lst, dataset_number=0)
    # dhist2, ahist2 = ATENAEnvCont.get_sessions_hists(actions_lst, dataset_number=0)
    # print(dhist1[-1])
    # print(dhist2[-1])
    # print(str(dhist1[-1]) == str(dhist2[-1]))
    # print(len(dhist1))
    # d = dict()
    # d[str(dhist1[1])] = 1
    # print(d)