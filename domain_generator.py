import numpy as np
import random as rd
class blockWorld:
    def __init__(self):
        self.token_size = 20
        self.num_of_tokens = 30
        self.name = "blockworld"
        self.description = "A simple block world domain where blocks can be stacked."
        self.actions = [("stack", "block", "block"), ("pick", "block"), ("putdown", "block"), ("unstack", "block", "block")]
        self.current_state = {"on": [], "clear": {}, "holding": {}}
        self.on = dict()
        self.objects = {"block": 3}

        for i in range(self.objects["block"]):
            self.on[f"block_{i}"] = None
        self.types = ["block"]
        self.predicates_arity = {"block": {"on": ("block","block"), "clear": ("block"), "holding": ("block")}}
        self.global_predicates = {"hand_free": True}
        self.on = {"on" : lambda block1, block2: (block1, block2) in self.current_state["on"]}

        self.create_objects_to_prop()

        self.action_map = {
            "stack": self.stack,
            "pick": self.pick,
            "putdown": self.putdown
        }
        self.create_token_map()

    def create_objects_to_prop(self):
        self.objects_to_propositions = dict()
        for type in self.types:
            self.objects_to_propositions[type] = dict()
            for obj in range(self.objects[type]):
                self.objects_to_propositions[type][f"{obj}"] = (False, True, False)  # (on, clear, holding)

    def stack(self, block1, block2):
        if self.objects_to_propositions[block1][1] and self.objects_to_propositions[block2][2]:
            self.current_state["on"].append((block1, block2))
            self.on[block2] = block1
            self.objects_to_propositions["block"][block1] = (self.on["on"](block1, block2), False, False)
            self.objects_to_propositions["block"][block2] = (False, True, False)
            self.global_predicates["hand_free"] = True
            return True
        return False

    def pick(self, block):
        if self.objects_to_propositions[block][1]:
            self.current_state["holding"] = block
            if self.on.get(f"block_{block}") != None:
                self.current_state["on"].remove((self.on[block], block))
                self.on[block] = None
                self.global_predicates["hand_free"] = False
            self.objects_to_propositions[block] = (True, False, True)
            return True
        return False
    
    def putdown(self, block):
        if self.objects_to_propositions[block][2]:
            self.objects_to_propositions[block] = (False, True, False)
            self.global_predicates["hand_free"] = True
            return True
        return False

    def unstack(self, block1, block2):
        if self.objects_to_propositions[block1][2] and self.objects_to_propositions[block2][0]:
            self.current_state["on"].remove((block1, block2))
            self.on[block2] = None
            self.objects_to_propositions[block1] = (False, True, False)
            self.objects_to_propositions[block2] = (self.on["on"](block1, block2), False, False)
            self.global_predicates["hand_free"] = True
            return True
        return False
    

    def next_trace(self):
        action = rd.choice(self.actions)
        if action == "stack":
            block1 = rd.choice([f"{i}" for i in range(self.objects["block"]) if self.objects_to_propositions["block"][i][1]])
            block2 = rd.choice([f"{i}" for i in range(self.objects["block"]) if i != block1 and self.objects_to_propositions["block"][1] and
                               self.objects_to_propositions["block"][i][2]] and self.global_predicates["hand_free"])
            if self.stack(block1, block2):
                return action
            

        elif action == "pick":
            block = rd.choice([f"{i}" for i in range(self.objects["block"]) if self.objects_to_propositions["block"][i][1]] and self.global_predicates["hand_free"])
            if self.pick(block):
                return action
            


        elif action == "putdown":
            block = rd.choice([f"block_{i}" for i in range(self.objects["block"]) if self.objects_to_propositions["block"][i][2]])
            if self.putdown(block):
                return action
            

        elif action == "unstack":
            block1 = rd.choice([f"block_{i}" for i in range(self.objects["block"])])
            block2 = rd.choice([f"block_{i}" for i in range(self.objects["block"]) if i != block1])
            if self.unstack(block1, block2):
                return action
            
        return action
    
    def get_tokens(self,action):
        tokens = [self.get_actionTokens(action)]
        for (typ,obj) in action[1:]:
            objects_tokens = self.get_predicate_Tokens(typ,obj,1)
            tokens.extend(objects_tokens)
        global_tokens = self.get_global_tokens()
        tokens.extend(global_tokens)
        # fill all other tokens to the fixed size
        while len(tokens) < self.num_of_tokens:
            tokens.append(np.zeros(self.token_size))
        return tokens
    
    def get_actionTokens(self, action):
        token = np.zeros(self.token_size)
        token[0] = 1  # action token
        return token

    def create_token_map(self):
        token_map = {}
        for i, (pred, _) in enumerate((self.predicates_arity["block"] | self.global_predicates).items()):
            token_map[pred] = i
        self.token_map = token_map


    def get_predicate_Tokens(self, type, obj, instace):
        preds = self.predicates_arity[type].keys()
        props = self.objects_to_propositions[type][str(obj)]
        tokens = []
        for i,pred in enumerate(preds):
            token = np.zeros(self.token_size)
            indx = self.token_map[pred]
            token[indx] = 1
            if props[i]:
                token[indx + 1] = 1
            token[indx + 2] = instace
            tokens.append(token)
        return tokens
    
    def get_global_tokens(self):
        tokens = []
        for pred in self.global_predicates:
            token = np.zeros(self.token_size)
            indx = self.token_map[pred]
            token[indx] = 1
            if self.global_predicates[pred]:
                token[indx + 1] = 1
            tokens.append(token)
        return tokens
    # token - [......,indx,bool value,placement ]