import asyncio
import random
import itertools
import numpy as np
from collections.abc import Sequence
from pddlsim.parser import (
    parse_domain_problem_pair_from_files,
    parse_domain_from_file,
    parse_problem_from_file,
)
from pddlsim.local import LocalSimulator
from pddlsim.remote.server import SimulatorConfiguration
from pddlsim.remote.client import (
    GiveUpAction,
    SimulationAction,
    SimulationClient,
    SimulationState,
    with_no_initializer,
)
from pddlsim.ast import (
    GroundedAction,
    Domain,
    Problem ,
    Identifier,
    Parameters,
    PredicateDefinition,
    CustomType
)

class IndexManager:
    def __init__(self, start_index):
        self.pred_to_index_mapper = dict()
        self.current_last_index = start_index
        self.start_index = start_index

    def add_pred_to_map(self, pred: PredicateDefinition):
        if str(pred) in self.pred_to_index_mapper:
            return
        self.pred_to_index_mapper[str(pred)] = self.current_last_index
        self.current_last_index+= 3
    
    def get_pred_index(self, pred):
        if str(pred) in self.pred_to_index_mapper:
            return self.pred_to_index_mapper[str(pred)]
        return -1