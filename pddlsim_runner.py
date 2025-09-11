#!/usr/bin/env python3
"""
pddlsim-based logistics simulator runner (local)
Uses your logistics_domain.pddl and logistics_problem.pddl
"""

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

class Domain_sim:

    def __init__(self, DOMAIN_FILE=None,PROBLEM_FILE=None, action_space = 20, pred_offset=0, action_offset=0, token_size=50):
        self.DOMAIN_FILE = DOMAIN_FILE or "./pddlDomains/logistics_domain.pddl"
        self.PROBLEM_FILE = PROBLEM_FILE or "./pddlDomains/logistics_problem.pddl"
        # Parse domain and problem
        try:
            # Preferred combined loader (less boilerplate)
            self.domain, self.problem = parse_domain_problem_pair_from_files(self.DOMAIN_FILE, self.PROBLEM_FILE)
        except Exception:
            # Fallback to individual loaders
            self.domain = parse_domain_from_file(self.DOMAIN_FILE)
            self.problem = parse_problem_from_file(self.PROBLEM_FILE, Domain_sim)
        self.action_space = action_space
        pred = list(self.domain.predicates_section._items.keys())[0]
        pred2 = self.domain.predicates_section._items[pred]
        params = pred2.parameters
        self.traces = []
        self.last_action = None
        self.pred_offset = pred_offset
        self.action_offset = action_offset
        self.token_size = token_size
        self.num_of_tokens = 60
        self.create_token_map()

    def create_token_map(self):
        token_map = {}
        preds = self.domain.predicates_section._items
        for i, pred in enumerate(preds.keys()):
            token_map[pred] = i*3 + self.pred_offset + self.action_space
        self.token_map = token_map

    
    @staticmethod
    def pick_grounded_action( actions: Sequence[GroundedAction]) -> GroundedAction:
        return random.choice(actions)


    async def get_next_action(self, simulation: SimulationClient) -> SimulationAction:
        states = await simulation.get_perceived_state()
        action_prob = [1, 0.4, 0.7, 0.2]
        if self.last_action:
            self.traces.append(self.get_tokens(action=self.last_action, states=states))
        # Log a small preview of options
        options = await simulation.get_grounded_actions()
        if options:
            # tokens = await simulation.get_tokens()
            self.get_tokens(states,options[0])

            # Print the first few options for visibility
            preview = ", ".join(str(a) for a in options[:3])
            # print(f"Applicable actions ({len(options)}): {preview}{' ...' if len(options) > 3 else ''}")
        else:
            print("No applicable actions, giving up.")
        match len(options):
            case 0:
                return GiveUpAction.from_dead_end()
            case 1:
                chosen = options[0]
            case _:
                chosen = Domain_sim.pick_grounded_action(options)
        # print(f"Chosen action: {chosen}")
        self.traces.append(self.get_tokens(action=chosen,states=states))

        self.last_action = chosen
        return chosen

    @staticmethod
    def make_step_limited_policy(base_policy, max_steps: int):
        step_counter = 0
        async def limited(sim):
            nonlocal step_counter
            if step_counter >= max_steps:
                print(f"[LIMIT] Reached step cap {max_steps}, terminating.")
                return GiveUpAction.from_dead_end()
            step_counter += 1
            return await base_policy(sim)
        return limited

    def 
    async def Generate_trace(self, trace_size, prob = False):
        self.traces = []
        # Create a local simulator from the config
        local_simulator = await LocalSimulator.from_configuration(
            SimulatorConfiguration(self.domain, self.problem)
        )
        self.prob = prob
        # print("Starting local simulation with random agent...")
        summary = await local_simulator.simulate(with_no_initializer(Domain_sim.make_step_limited_policy(self.get_next_action, max_steps=trace_size)))
        self.last_action = None
        self.prob = False
        return self.traces

    def get_tokens(self, states: SimulationState, action: SimulationAction):
        tokens = [self.create_action_token(action)]
        tokens.extend(self.get_obj_predicates_Tokens(simulation_states=states, objs=action.grounding))

        # fill all other tokens to the fixed size
        while len(tokens) < self.num_of_tokens:
            tokens.append(np.zeros(self.token_size))
        return tokens
    
    def create_action_token(self, action: SimulationAction):
        actions = list(map(lambda a: a.value, list(self.domain.actions_section._items.keys())))
        idx = actions.index(action.name.value)        

        token = np.zeros(self.token_size)
        token[idx + self.action_offset] = 1
        return token
    
    def get_obj_predicates_Tokens(self, objs, simulation_states: SimulationState):
        tokens = []
        states_props = simulation_states._true_predicates
        types = self.get_all_object_types(objs)
        preds = self.get_types_predicates(types)
        inst_counter = {}
        existing_props= {}
        for pred in preds:
            token = np.zeros(self.token_size)
            pred_name = pred.name
            if pred_name in inst_counter:
                inst_counter[pred_name] +=1
            else:
                inst_counter[pred_name] = 1
            prop = self.pred_to_prop(pred, types, existing_props)
            if not prop:
                continue
            indx = self.token_map[pred_name]
            token[indx] = 1
            truth = len(list(filter(lambda state: prop[0].name == state.name and prop[1]==state.assignment, states_props))) > 0
            if truth:
                i = 1
            token[indx + 1] = 1 if truth else 0
            token[indx + 2] = inst_counter[pred_name]
            tokens.append(token)
        return tokens
    
    def pred_to_prop(self, pred: PredicateDefinition, obj_types: dict, existing_props: dict):
        params = pred.parameters
        pred_name = pred.name
        if pred_name not in existing_props:
            existing_props[pred_name] =[]
        asignment = {}
        param_candidates = []
        for name, type in params._items.items():
            candidates = [obj for obj, types in obj_types.items() if type in types]
            if not candidates:
                return None  # No object of required type
            param_candidates.append(candidates)
        # Try all permutations
        for assignment in itertools.product(*param_candidates):
            if len(set(assignment)) < len(assignment):
                continue  # Skip assignments with duplicate objects
            if assignment not in existing_props[pred_name]:
                existing_props[pred_name].append(assignment)
                return (pred, assignment)
        return None
        

    def get_all_object_types(self, objs):
        collection = {}
        type_to_type =self.domain.types_section._items
        for obj in objs:
            collection[obj] = []
            obj_type = self.problem.objects_section._items[obj]
            queue = [obj_type]
            while queue:
                current = queue.pop(0)
                collection[obj].append(current)
                if current in type_to_type:
                    queue.append(type_to_type[current])
        return collection
     
    def get_types_predicates(self, objs_types_map: dict) -> list[PredicateDefinition]:
        preds = []
        for _,obj_types in objs_types_map.items():
            for _type in obj_types:
                for param in self.domain.predicates_section._items.values():
                    if _type in param.parameters._items.values():
                        preds.append(param)
        return preds


if __name__ == "__main__":
    domain = Domain_sim()
    asyncio.run(domain.Generate_trace(trace_size=10))

#     (at ?obj - object ?loc - location)
#     (in ?pkg - package ?veh - vehicle)
#     (loaded ?pkg - package ?veh - vehicle) 
#     (between ?obj ?loc ?loc)



# drive 
#