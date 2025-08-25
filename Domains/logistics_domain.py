import numpy as np
import random as rd

class LogisticsDomain:
    def __init__(self):
        self.token_size = 30
        self.num_of_tokens = 30
        self.name = "logistics"
        self.description = "A logistics domain where packages are transported using trucks and airplanes between locations."
        self.actions = [("load", "package", "vehicle", "location"), 
                       ("unload", "package", "vehicle", "location"),
                       ("drive", "truck", "location", "location"),
                       ("fly", "airplane", "location", "location")]
        
        # Define objects in the domain
        self.objects = {"package": 3, "truck": 2, "airplane": 1, "location": 4}
        
        # Initialize locations and object positions
        self.locations = {}  # tracks what objects are at each location
        self.vehicle_contents = {}  # tracks what packages are in each vehicle
        self.object_locations = {}  # tracks where each object is located
        
        # Initialize locations as clear
        for loc in range(self.objects["location"]):
            self.locations[loc] = {"clear": True, "objects": []}
        
        # Initialize vehicle contents as empty
        for truck in range(self.objects["truck"]):
            self.vehicle_contents[("truck", truck)] = []
        for plane in range(self.objects["airplane"]):
            self.vehicle_contents[("airplane", plane)] = []
        
        # Place all objects at random locations initially
        for pkg in range(self.objects["package"]):
            loc = rd.randint(0, self.objects["location"] - 1)
            self.object_locations[("package", pkg)] = loc
            self.locations[loc]["objects"].append(("package", pkg))
            
        for truck in range(self.objects["truck"]):
            loc = rd.randint(0, self.objects["location"] - 1)
            self.object_locations[("truck", truck)] = loc
            self.locations[loc]["objects"].append(("truck", truck))
            
        for plane in range(self.objects["airplane"]):
            loc = rd.randint(0, self.objects["location"] - 1)
            self.object_locations[("airplane", plane)] = loc
            self.locations[loc]["objects"].append(("airplane", plane))
        
        self.types = ["package", "truck", "airplane", "location"]
        self.predicates_arity = {
            "package": {"at": ("package", "location"), "in": ("package", "vehicle"), "loaded": ("package", "vehicle")},
            "truck": {"at": ("truck", "location")},
            "airplane": {"at": ("airplane", "location")},
            "location": {"clear": ("location")}
        }
        self.global_predicates = {}
        
        self.create_objects_to_prop()
        
        self.action_map = {
            "load": self.load,
            "unload": self.unload,
            "drive": self.drive,
            "fly": self.fly
        }
        self.create_token_map()

    def create_objects_to_prop(self):
        self.objects_to_propositions = dict()
        
        # Initialize package propositions (at, in, loaded)
        self.objects_to_propositions["package"] = dict()
        for pkg in range(self.objects["package"]):
            is_at_location = True  # packages start at locations, not in vehicles
            is_in_vehicle = False
            is_loaded = False
            self.objects_to_propositions["package"][pkg] = (is_at_location, is_in_vehicle, is_loaded)
        
        # Initialize truck propositions (at)
        self.objects_to_propositions["truck"] = dict()
        for truck in range(self.objects["truck"]):
            is_at_location = True
            self.objects_to_propositions["truck"][truck] = (is_at_location,)
        
        # Initialize airplane propositions (at)
        self.objects_to_propositions["airplane"] = dict()
        for plane in range(self.objects["airplane"]):
            is_at_location = True
            self.objects_to_propositions["airplane"][plane] = (is_at_location,)
        
        # Initialize location propositions (clear)
        self.objects_to_propositions["location"] = dict()
        for loc in range(self.objects["location"]):
            is_clear = self.locations[loc]["clear"]
            self.objects_to_propositions["location"][loc] = (is_clear,)

    def load(self, pkg, vehicle_type, vehicle_id, loc):
        """Load a package into a vehicle at a specific location"""
        vehicle = (vehicle_type, vehicle_id)
        
        # Check preconditions
        pkg_location = self.object_locations.get(("package", pkg))
        vehicle_location = self.object_locations.get(vehicle)
        
        # Package must be at the location
        if pkg_location != loc:
            return False
        
        # Vehicle must be at the same location
        if vehicle_location != loc:
            return False
        
        # Location must be clear (simplified - always allow for now)
        if not self.locations[loc]["clear"]:
            pass  # Allow loading even if location has other objects
        
        # Package must not already be in a vehicle
        if not self.objects_to_propositions["package"][pkg][0]:  # not at location anymore
            return False
        
        # Execute the action
        # Remove package from location
        self.locations[loc]["objects"].remove(("package", pkg))
        
        # Add package to vehicle
        self.vehicle_contents[vehicle].append(pkg)
        
        # Update object propositions
        self.objects_to_propositions["package"][pkg] = (False, True, True)  # not at location, in vehicle, loaded
        
        # Update package location to be "in vehicle" (special marker)
        self.object_locations[("package", pkg)] = None
        
        return True

    def unload(self, pkg, vehicle_type, vehicle_id, loc):
        """Unload a package from a vehicle at a specific location"""
        vehicle = (vehicle_type, vehicle_id)
        
        # Check preconditions
        vehicle_location = self.object_locations.get(vehicle)
        
        # Package must be in the vehicle
        if pkg not in self.vehicle_contents[vehicle]:
            return False
        
        # Vehicle must be at the location
        if vehicle_location != loc:
            return False
        
        # Location must be clear (simplified)
        if not self.locations[loc]["clear"]:
            pass  # Allow unloading even if location has other objects
        
        # Execute the action
        # Remove package from vehicle
        self.vehicle_contents[vehicle].remove(pkg)
        
        # Add package to location
        self.locations[loc]["objects"].append(("package", pkg))
        self.object_locations[("package", pkg)] = loc
        
        # Update object propositions
        self.objects_to_propositions["package"][pkg] = (True, False, False)  # at location, not in vehicle, not loaded
        
        return True

    def drive(self, truck_id, from_loc, to_loc):
        """Move a truck from one location to another"""
        truck = ("truck", truck_id)
        
        # Check preconditions
        truck_location = self.object_locations.get(truck)
        
        # Truck must be at the starting location
        if truck_location != from_loc:
            return False
        
        # Destination location must be clear (simplified - always allow)
        if not self.locations[to_loc]["clear"]:
            pass  # Allow driving to any location
        
        # Execute the action
        # Remove truck from old location
        self.locations[from_loc]["objects"].remove(truck)
        
        # Add truck to new location
        self.locations[to_loc]["objects"].append(truck)
        self.object_locations[truck] = to_loc
        
        return True

    def fly(self, plane_id, from_loc, to_loc):
        """Move an airplane from one location to another"""
        plane = ("airplane", plane_id)
        
        # Check preconditions
        plane_location = self.object_locations.get(plane)
        
        # Airplane must be at the starting location
        if plane_location != from_loc:
            return False
        
        # Destination location must be clear (simplified - always allow)
        if not self.locations[to_loc]["clear"]:
            pass  # Allow flying to any location
        
        # Execute the action
        # Remove airplane from old location
        self.locations[from_loc]["objects"].remove(plane)
        
        # Add airplane to new location
        self.locations[to_loc]["objects"].append(plane)
        self.object_locations[plane] = to_loc
        
        return True

    def choose_action(self):
        """Choose a random valid action"""
        valid_actions = []
        
        # Try load actions
        for pkg in range(self.objects["package"]):
            if self.objects_to_propositions["package"][pkg][0]:  # package at location
                pkg_loc = self.object_locations[("package", pkg)]
                
                # Try loading into trucks at same location
                for truck in range(self.objects["truck"]):
                    truck_loc = self.object_locations[("truck", truck)]
                    if truck_loc == pkg_loc:
                        valid_actions.append(("load", ("package", pkg), ("truck", truck), ("location", pkg_loc)))
                
                # Try loading into airplanes at same location
                for plane in range(self.objects["airplane"]):
                    plane_loc = self.object_locations[("airplane", plane)]
                    if plane_loc == pkg_loc:
                        valid_actions.append(("load", ("package", pkg), ("airplane", plane), ("location", pkg_loc)))
        
        # Try unload actions
        for truck in range(self.objects["truck"]):
            truck_vehicle = ("truck", truck)
            truck_loc = self.object_locations[truck_vehicle]
            for pkg in self.vehicle_contents[truck_vehicle]:
                valid_actions.append(("unload", ("package", pkg), ("truck", truck), ("location", truck_loc)))
        
        for plane in range(self.objects["airplane"]):
            plane_vehicle = ("airplane", plane)
            plane_loc = self.object_locations[plane_vehicle]
            for pkg in self.vehicle_contents[plane_vehicle]:
                valid_actions.append(("unload", ("package", pkg), ("airplane", plane), ("location", plane_loc)))
        
        # Try drive actions
        for truck in range(self.objects["truck"]):
            truck_loc = self.object_locations[("truck", truck)]
            for to_loc in range(self.objects["location"]):
                if to_loc != truck_loc:
                    valid_actions.append(("drive", ("truck", truck), ("location", truck_loc), ("location", to_loc)))
        
        # Try fly actions
        for plane in range(self.objects["airplane"]):
            plane_loc = self.object_locations[("airplane", plane)]
            for to_loc in range(self.objects["location"]):
                if to_loc != plane_loc:
                    valid_actions.append(("fly", ("airplane", plane), ("location", plane_loc), ("location", to_loc)))
        
        if valid_actions:
            return rd.choice(valid_actions)
        else:
            # Fallback - just drive a truck somewhere
            truck_loc = self.object_locations[("truck", 0)]
            to_loc = (truck_loc + 1) % self.objects["location"]
            return ("drive", ("truck", 0), ("location", truck_loc), ("location", to_loc))

    def next_trace(self, action):
        suc = False
        action_func = self.action_map.get(action[0])
        if action_func:
            if action[0] in ["load", "unload"]:
                # load/unload: (action, (package_type, pkg_id), (vehicle_type, vehicle_id), (location_type, loc_id))
                pkg_id = action[1][1]
                vehicle_type = action[2][0] 
                vehicle_id = action[2][1]
                loc_id = action[3][1]
                suc = action_func(pkg_id, vehicle_type, vehicle_id, loc_id)
            elif action[0] in ["drive", "fly"]:
                # drive/fly: (action, (vehicle_type, vehicle_id), (location_type, from_loc), (location_type, to_loc))
                vehicle_id = action[1][1]
                from_loc = action[2][1]
                to_loc = action[3][1]
                suc = action_func(vehicle_id, from_loc, to_loc)
        return suc

    def prob_next_trace(self, action):
        suc = False
        action_func = self.action_map.get(action[0])
        if action[0] in ["load", "unload"] and rd.random() < 0.2:
            return True
        if action_func:
            if action[0] in ["load", "unload"]:
                pkg_id = action[1][1]
                vehicle_type = action[2][0] 
                vehicle_id = action[2][1]
                loc_id = action[3][1]
                suc = action_func(pkg_id, vehicle_type, vehicle_id, loc_id)
            elif action[0] in ["drive", "fly"]:
                vehicle_id = action[1][1]
                from_loc = action[2][1]
                to_loc = action[3][1]
                suc = action_func(vehicle_id, from_loc, to_loc)
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
        idx = 0
        # Reserve first 4 indices for actions
        current_idx = 4
        breaks = False
        for type_name in self.types:
            for pred in self.predicates_arity[type_name]:
                if pred in token_map:
                    self.predicates_arity[type_name][f"{pred}_{type_name}"] = self.predicates_arity[type_name].pop(pred)
                    self.create_token_map()
                    breaks = True
                    break
                else:
                    token_map[pred] = current_idx
                    current_idx += 3  # Each predicate uses 3 indices (predicate, value, instance)
            if breaks:
                break

        if not breaks:
            for pred in self.global_predicates:
                if current_idx + 2 < self.token_size:
                    token_map[pred] = current_idx
                    current_idx += 3
            
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
        action_indices = np.where(action_token == 1)[0]
        if len(action_indices) > 0:
            action_num = action_indices[0]
            return list(self.action_map.keys())[action_num]
        return None

    def init(self):
        self.__init__()
