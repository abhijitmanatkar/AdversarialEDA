import operator
from collections import namedtuple
import numpy as np
from collections import Counter
from benchmark.atena.evaluation.metrics import EvalInstance

STACK_OBS_NUM = 3

#-------------------------------------------------------------------------------

class CounterWithoutNanKeys(Counter):
    def __init__(self, iterable):
        self.non_nan_iterable = [elem for elem in iterable if isinstance(elem, str) or not np.isnan(elem)]
        super().__init__(self.non_nan_iterable)

#-------------------------------------------------------------------------------

def hack_min(pd_series):
    return np.min(pd_series.dropna())


def hack_max(pd_series):
    return np.max(pd_series.dropna())

SUPPORTED_ACTIONS = ['back', 'filter', 'group', 'stop']
SUPPORTED_ACTIONS_ONLY_STOP = ['stop', 'filter', 'group']
SUPPORTED_ACTIONS_ONLY_BACK = ['back', 'filter', 'group']


SUPPORTED_OPERATORS = {
    0: 'EQ',
    1: 'NEQ',
    2: 'GT',
    3: 'GE',
    4: 'LT',
    5: 'LE',
    6: 'CONTAINS', # string contains
    7: 'STARTS_WITH',
    8: 'ENDS_WITH'
}

SUPPORTED_OPERATORS_REVERSE = dict()
for key, value in SUPPORTED_OPERATORS.items():
    SUPPORTED_OPERATORS_REVERSE[value] = key

OPERATOR_MAP = {
    'EQ': operator.eq,
    'NEQ': operator.ne,
    'GT': operator.gt,
    'GE': operator.ge,
    'LT': operator.lt,
    'LE': operator.le
}

# AGG_MAP= {
#     0: np.sum,
#     1: len ,
#     2: hack_min,#lambda x:np.nanmin(x.dropna()),
#     3: hack_max,#lambda x:np.nanmax(x.dropna()),
#     4: np.mean
# }

SUPPORTED_AGG_FUNCS = {
    0: len,
    1: np.sum,
    2: hack_min,
    3: hack_max,
    4: np.mean
}

SUPPORTED_AGG_FUNCS_REVERSE = dict()
for key, value in SUPPORTED_AGG_FUNCS.items():
    SUPPORTED_AGG_FUNCS_REVERSE[value] = key

EXPERT_OPERATOR_MAP = {
    2: 'CONTAINS',
    4: 'STARTS_WITH',
    8: 'EQ',
    16: 'ENDS_WITH',
    32: 'GT',
    64: 'GE',
    128: 'LT',
    256: 'LE',
    512: 'NEQ',
}


EXPERT_AGG_MAP = {
    'avg': np.mean,
    'count': len,
    'max': hack_max,
    'min': hack_min,
    'sum': np.sum
}

#-------------------------------------------------------------------------------

FilteringTuple = namedtuple('FilteringTuple', ["field", "term", "condition"])
AggregationTuple = namedtuple('AggregationTuple', ["field", "type"])
SortingTuple = namedtuple("SortingTuple", ["field", "order"])

#-------------------------------------------------------------------------------

class GetItemByStr(tuple):
    __slots__ = ()

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return self.__getattribute__(attr)
        else:
            return tuple.__getitem__(self, attr)


class EnvStateTuple(namedtuple('EnvStateTuple', ["filtering", "grouping", "aggregations"]), GetItemByStr):
    """
    see https://stackoverflow.com/questions/44320382/subclassing-python-namedtuple
    """
    __slots__ = ()

    def reset_filtering(self):
        return self._create_state_tuple(filtering=tuple(),
                                        grouping=self.grouping,
                                        aggregations=self.aggregations)

    def reset_grouping_and_aggregations(self):
        return self._create_state_tuple(filtering=self.filtering,
                                        grouping=tuple(),
                                        aggregations=tuple())

    def append_filtering(self, elem):
        field_lst = self._append_to_field(elem, "filtering")
        return self._create_state_tuple(filtering=field_lst,
                                        grouping=self.grouping,
                                        aggregations=self.aggregations)

    def append_grouping(self, elem):
        field_lst = self._append_to_field(elem, "grouping")
        return self._create_state_tuple(filtering=self.filtering,
                                        grouping=field_lst,
                                        aggregations=self.aggregations)

    def append_aggregations(self, elem):
        field_lst = self._append_to_field(elem, "aggregations")
        return self._create_state_tuple(filtering=self.filtering,
                                        grouping=self.grouping,
                                        aggregations=field_lst)

    def _append_to_field(self, elem, field):
        field_lst = list(self[field])
        field_lst.append(elem)
        return field_lst

    def __getitem__(self, attr):
        return GetItemByStr.__getitem__(self, attr)

    @classmethod
    def _create_state_tuple(cls, filtering, grouping, aggregations):
        return cls(
            filtering=tuple(filtering),
            grouping=tuple(grouping),
            aggregations=tuple(aggregations),
        )

    @classmethod
    def create_empty_state(cls):
        return cls._create_state_tuple((), (), ())


empty_env_state = EnvStateTuple.create_empty_state()

#-------------------------------------------------------------------------------

class CustomEvalInstance(EvalInstance):
    def __init__(self, dataset_meta, actions_lst, references_actions):
        super().__init__(dataset_meta, actions_lst)
        self._references_actions = references_actions
    
    @property
    def references_actions(self):
        return self._references_actions
    

