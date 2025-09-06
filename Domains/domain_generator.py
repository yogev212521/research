import numpy as np
import random as rd
class blockWorld:
    def __init__(self):
        self.token_size = 30
        self.num_of_tokens = 30
        self.name = "blockworld"
        self.description = "A simple block world domain where blocks can be stacked."
        self.actions = [("stack", "block", "block"), ("pick", "block"), ("putdown", "block"), ("unstack", "block", "block")]
        self.current_state = {"on": [], "clear": {}, "holding": {}}
        self.on = dict()
        self.objects = {"block": 5}

        for i in range(self.objects["block"]):
            self.on[i] = None
        self.types = ["block"]
        self.predicates_arity = {"block": {"on": ("block","block"), "clear": ("block"), "holding": ("block")}}
        self.global_predicates = {"hand_free": True}
        self.current_state = {"on" : lambda block1, block2: (block1, block2) in self.current_state["on"]}

        self.create_objects_to_prop()

        self.action_map = {
            "stack": self.stack,
            "pick": self.pick,
            "putdown": self.putdown,
            "unstack": self.unstack
        }
        self.create_token_map()

    def create_objects_to_prop(self):
        self.objects_to_propositions = dict()
        for type in self.types:
            self.objects_to_propositions[type] = dict()
            for obj in range(self.objects[type]):
                self.objects_to_propositions[type][obj] = (False, True, False)  # (on, clear, holding)

    def stack(self, block1, block2):
        if self.objects_to_propositions["block"][block1][2] and self.objects_to_propositions["block"][block2][1]:
            self.on[block2] = block1
            self.objects_to_propositions["block"][block1] = (True, True, False)
            self.objects_to_propositions["block"][block2] = (self.objects_to_propositions["block"][block2][0], False, False)
            self.global_predicates["hand_free"] = True
            return True
        return False

    def pick(self, block):
        if self.objects_to_propositions["block"][block][1]:
            if self.on.get(block) != None:
                self.on[block] = None
            self.global_predicates["hand_free"] = False
            self.objects_to_propositions["block"][block] = (True, False, True)
            return True
        return False
    
    def putdown(self, block):
        if self.objects_to_propositions["block"][block][2]:
            self.objects_to_propositions["block"][block] = (False, True, False)
            self.global_predicates["hand_free"] = True
            return True
        return False

    def unstack(self, block1, block2):
        if self.objects_to_propositions["block"][block1][0] and not self.objects_to_propositions["block"][block2][1]:
            self.on[block2] = None
            self.objects_to_propositions["block"][block1] = (False, True, True)
            self.objects_to_propositions["block"][block2] = (self.objects_to_propositions["block"][block2][0], True, False)
            self.global_predicates["hand_free"] = True
            return True
        return False

    def choose_action(self):
        if self.global_predicates["hand_free"]:
            action = rd.choice(["unstack", "pick"])
            if action == "unstack":
                try:
                    block1 = rd.choice([i for i in range(self.objects["block"]) if self.objects_to_propositions["block"][i][1] and
                                        self.on[i] is not None])
                    block2 = self.on[block1]
                    return (action, ("block", block1), ("block", block2))
                except:
                    block = rd.choice([i for i in range(self.objects["block"]) if self.objects_to_propositions["block"][i][1]])
                    return ("pick", ("block", block))
            else:
                block = rd.choice([i for i in range(self.objects["block"]) if self.objects_to_propositions["block"][i][1]])
                return (action, ("block", block))
        else:
            action =  rd.choice(["putdown", "stack"])
            if action == "putdown":
                block = rd.choice([i for i in range(self.objects["block"]) if self.objects_to_propositions["block"][i][2]])
                return (action, ("block", block))
            else:
                block1 = rd.choice([i for i in range(self.objects["block"]) if self.objects_to_propositions["block"][i][2]])
                try:
                    block2 = rd.choice([i for i in range(self.objects["block"]) if i != block1 and self.objects_to_propositions["block"][i][1]])
                except:
                    block = rd.choice([i for i in range(self.objects["block"]) if self.objects_to_propositions["block"][i][2]])
                    return ("putdown", ("block", block1))
                return (action, ("block", block1), ("block", block2))

    def next_trace(self,action):
        suc = False
        action_func = self.action_map.get(action[0])
        params = [obj[1] for obj in action[1:]]
        if action_func and params:
            suc = action_func(*params)
        return suc

    def prob_next_trace(self,action):
        suc = False
        action_func = self.action_map.get(action[0])
        if action[0] == "pick" and rd.random() < 0.3:
            return True
        params = [obj[1] for obj in action[1:]]
        if action_func and params:
            suc = action_func(*params)
        return suc

    def get_tokens_from(self, from_pred, from_action, action):
        tokens = [self.get_actionTokens(action, from_action, from_pred)]
        object_inst = dict()
        for (typ,obj) in action[1:]:
            if typ not in object_inst:
                object_inst[typ] = 1
            else:
                object_inst[typ] += 1
            inst = object_inst[typ]
            objects_tokens = self.get_predicate_Tokens(typ,obj,inst,from_pred)
            tokens.extend(objects_tokens)
        global_tokens = self.get_global_tokens(from_pred=from_pred)
        tokens.extend(global_tokens)
        # fill all other tokens to the fixed size
        while len(tokens) < self.num_of_tokens:
            tokens.append(np.zeros(self.token_size + from_pred))
        return tokens
    
    def get_tokens(self,action):
        tokens = [self.get_actionTokens(action)]
        object_inst = dict()
        for (typ,obj) in action[1:]:
            if typ not in object_inst:
                object_inst[typ] = 1
            else:
                object_inst[typ] += 1
            inst = object_inst[typ]
            objects_tokens = self.get_predicate_Tokens(typ,obj,inst)
            tokens.extend(objects_tokens)
        global_tokens = self.get_global_tokens()
        tokens.extend(global_tokens)
        # fill all other tokens to the fixed size
        while len(tokens) < self.num_of_tokens:
            tokens.append(np.zeros(self.token_size))
        return tokens

    def get_actionTokens(self, action, from_index = 0, added_size=0):
        indx = list(self.action_map.keys()).index(action[0])
        token = np.zeros(self.token_size +added_size)
        token[indx + from_index] = 1  # action token
        return token

    def create_token_map(self):
        token_map = {}
        for i, (pred, _) in enumerate((self.predicates_arity["block"] | self.global_predicates).items()):
            token_map[pred] = i*3 + 4
        self.token_map = token_map

    def get_predicate_Tokens(self, type, obj, instance, from_pred = 0):
        preds = self.predicates_arity[type].keys()
        props = self.objects_to_propositions[type][obj]
        tokens = []
        for i,pred in enumerate(preds):
            token = np.zeros(self.token_size)
            indx = self.token_map[pred]
            token[indx] = 1
            if props[i]:
                token[indx + 1] = 1
            token[indx + 2] = instance
            if from_pred > 0:
                pre_tok = np.zeros(from_pred)
                token = np.concatenate((pre_tok, token))
            tokens.append(token)
        return tokens

    def get_global_tokens(self, from_pred=0):
        tokens = []
        for pred in self.global_predicates:
            token = np.zeros(self.token_size)
            indx = self.token_map[pred]
            token[indx] = 1
            if self.global_predicates[pred]:
                token[indx + 1] = 1
            if from_pred > 0:
                pre_tok = np.zeros(from_pred)
                token = np.concatenate((pre_tok, token))
            tokens.append(token)
        return tokens
    # token - [......,indx,bool value,placement ]

    def get_token_to_pred_map(self, tokens,i):
        token = tokens[i]
        pred = i%3
        token_to_pred_map = {}
        for pred, indx in self.token_map.items():
            token_to_pred_map[indx] = pred
        return token_to_pred_map

    def get_token_to_action_map(self, tokens):
        action_token = tokens[0] if tokens is list else tokens
        action_num = action_token.index(1)
        self.action_map.keys()[action_num]

    def init(self):
        self.__init__()

# (on_peg, clear, smaller)
class HanoiTowers:
    def __init__(self):
        self.token_size = 25
        self.num_of_tokens = 30
        self.name = "hanoi"
        self.description = "Towers of Hanoi domain where discs can be moved between pegs."
        self.actions = [("move", "disc", "peg")]
        self.current_state = {"on": [], "clear": {}}
        self.on = dict()
        self.objects = {"disc": 3, "peg": 3}
        
        # Initialize disc positions (all on peg 0 initially)
        for i in range(self.objects["disc"]):
            self.on[i] = 0  # all discs start on peg 0
        
        # Initialize peg positions (what's on top of each peg)
        self.peg_top = {0: 0, 1: None, 2: None}  # peg 0 has smallest disc on top
        
        self.types = ["disc", "peg"]
        self.predicates_arity = {
            "disc": {"on": ("disc", "peg"), "clear": ("disc"), "smaller": ("disc", "disc")},
            "peg": {"clear": ("peg")}
        }
        self.global_predicates = {}
        
        # Initialize smaller relationships (disc i is smaller than disc j if i < j)
        self.smaller_relations = {}
        for i in range(self.objects["disc"]):
            for j in range(self.objects["disc"]):
                self.smaller_relations[(i, j)] = (i < j)
            # All discs are smaller than all pegs
            for peg in range(self.objects["peg"]):
                self.smaller_relations[(i, peg)] = True
        
        self.create_objects_to_prop()
        
        self.action_map = {
            "move": self.move
        }
        self.create_token_map()

    def create_objects_to_prop(self):
        self.objects_to_propositions = dict()
        
        # Initialize disc propositions
        self.objects_to_propositions["disc"] = dict()
        for disc in range(self.objects["disc"]):
            # (on_peg, clear, smaller) - disc 0 is smallest, disc 2 is largest
            is_clear = (disc == 0)  # only top disc is clear initially
            # smaller predicate is handled separately via smaller_relations
            self.objects_to_propositions["disc"][disc] = (True, is_clear, False)
        
        # Initialize peg propositions  
        self.objects_to_propositions["peg"] = dict()
        for peg in range(self.objects["peg"]):
            # (clear) - peg 0 has discs, pegs 1,2 are clear
            is_clear = (peg != 0)
            self.objects_to_propositions["peg"][peg] = (is_clear,)

    def can_move(self, disc, to_peg):
        """Check if disc can be moved to target peg according to Hanoi rules"""
        # Disc must be clear (on top)
        if not self.objects_to_propositions["disc"][disc][1]:
            return False
        
        # Target peg must be clear OR have a larger disc on top
        if self.peg_top[to_peg] is None:
            return True
        
        top_disc = 200 if self.peg_top[to_peg] == None else self.peg_top[to_peg]
        # Use smaller relations: disc must be smaller than top_disc
        return top_disc > disc

    def move(self, disc, from_peg, to_peg):
        """Move a disc to a target peg"""
        if not self.can_move(disc, to_peg):
            return False
        
        from_peg = self.on[disc]
        
        # Update disc position
        self.on[disc] = to_peg
        
        # Update what's on top of source peg
        # Find what disc is now on top of source peg
        new_top = None
        for d in range(self.objects["disc"]):
            if self.on[d] == from_peg and d != disc:
                if new_top is None or d < new_top:
                    new_top = d
        self.peg_top[from_peg] = new_top
        
        # Update what's on top of target peg
        self.peg_top[to_peg] = disc
        
        # Update disc propositions
        for d in range(self.objects["disc"]):
            peg = self.on[d]
            is_clear = (self.peg_top[peg] == d)
            # (on_peg, clear, smaller) - smaller is handled separately via smaller_relations
            self.objects_to_propositions["disc"][d] = (True, is_clear, False)
        self.objects_to_propositions["disc"][disc]  = (False, True, False)  # moved disc is now clear
        # Update peg propositions
        self.objects_to_propositions["peg"][to_peg] = (False,)
        self.objects_to_propositions["peg"][from_peg] = (self.peg_top[from_peg] is None,)

        return True

    def choose_action(self):
        """Choose a random valid move"""
        valid_moves = []
        disc_shuff = np.random.permutation(self.objects["disc"])
        for disc in disc_shuff:
            if self.objects_to_propositions["disc"][disc][1]:  # if disc is clear
                current_peg = self.on[disc]
                for target_peg in range(self.objects["peg"]):
                    if target_peg != current_peg and self.can_move(disc, target_peg):
                        valid_moves.append(("move", ("disc", disc),("peg", current_peg), ("peg", target_peg)))
        
        if valid_moves:
            return rd.choice(valid_moves)
        else:
            # Fallback - try to move smallest clear disc
            raise Exception("No valid moves available")
            for disc in range(self.objects["disc"]):
                if self.objects_to_propositions["disc"][disc][1]:
                    current_peg = self.on[disc]
                    target_peg = (current_peg + 1) % self.objects["peg"]
                    return ("move", ("disc", disc), ("peg", target_peg))

    def next_trace(self, action):
        suc = False
        action_func = self.action_map.get(action[0])
        params = [obj[1] for obj in action[1:]]
        if action_func and params:
            suc = action_func(*params)
        return suc

    def prob_next_trace(self, action):
        suc = False
        action_func = self.action_map.get(action[0])
        if action[0] == "move" and rd.random() < 0.1:
            return True
        params = [obj[1] for obj in action[1:]]
        if action_func and params:
            suc = action_func(*params)
        return suc

    def get_tokens(self, action):
        tokens = [self.get_actionTokens(action)]
        object_inst = dict()
        for (typ, obj) in action[1:]:
            if typ not in object_inst:
                object_inst[typ] = 1
            else:
                object_inst[typ] += 1
            inst = object_inst[typ]
            objects_tokens = self.get_predicate_Tokens(typ, obj, inst)
            tokens.extend(objects_tokens)
        global_tokens = self.get_global_tokens()
        tokens.extend(global_tokens)
        # fill all other tokens to the fixed size
        while len(tokens) < self.num_of_tokens:
            tokens.append(np.zeros(self.token_size))
        return tokens

    def get_actionTokens(self, action):
        indx = list(self.action_map.keys()).index(action[0])
        token = np.zeros(self.token_size)
        token[indx] = 1  # action token
        return token

    def create_token_map(self):
        token_map = {}
        idx = 1  # start after action tokens
        for type_name in self.types:
            for pred in self.predicates_arity[type_name]:
                token_map[pred] = idx * 3 + 1
                idx += 1
        for pred in self.global_predicates:
            token_map[pred] = idx * 3 + 1
            idx += 1
        self.token_map = token_map

    def get_predicate_Tokens(self, type, obj, instance):
        preds = self.predicates_arity[type].keys()
        props = self.objects_to_propositions[type][obj]
        tokens = []
        for i, pred in enumerate(preds):
            token = np.zeros(self.token_size)
            indx = self.token_map[pred]
            token[indx] = 1
            if props[i]:
                token[indx + 1] = 1
            token[indx + 2] = instance
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

    def get_token_to_pred_map(self, tokens, i):
        token = tokens[i]
        pred = i % 3
        token_to_pred_map = {}
        for pred, indx in self.token_map.items():
            token_to_pred_map[indx] = pred
        return token_to_pred_map

    def get_token_to_action_map(self, tokens):
        action_token = tokens[0] if isinstance(tokens, list) else tokens
        action_num = action_token.index(1)
        return list(self.action_map.keys())[action_num]

    def init(self):
        self.__init__()
