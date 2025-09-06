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
        
        # Centralized state management
        self.state = {
            # Track where each package is: either a location_id or ("vehicle_type", vehicle_id)
            "package_locations": {},
            # Track where each vehicle is: location_id
            "vehicle_locations": {},
            # Track which packages are in which vehicles
            "vehicle_contents": {},
            # Track what objects are at each location (excluding packages in vehicles)
            "location_contents": {}
        }
        
        self._initialize_state()
        
        self.types = ["package", "truck", "airplane", "location"]
        self.predicates_arity = {
            "package": {"at": ("package", "location"), "in": ("package", "vehicle"), "loaded": ("package", "vehicle")},
            "truck": {"at": ("truck", "location")},
            "airplane": {"at": ("airplane", "location")},
            "location": {"clear": ("location")}
        }
        self.global_predicates = {}
        
        self.action_map = {
            "load": self.load,
            "unload": self.unload,
            "drive": self.drive,
            "fly": self.fly
        }
        self.create_token_map()

    def _initialize_state(self):
        """Initialize the centralized state management system"""
        # Initialize all data structures
        for pkg in range(self.objects["package"]):
            loc = rd.randint(0, self.objects["location"] - 1)
            self.state["package_locations"][pkg] = loc
            
        for truck in range(self.objects["truck"]):
            loc = rd.randint(0, self.objects["location"] - 1)
            self.state["vehicle_locations"][("truck", truck)] = loc
            self.state["vehicle_contents"][("truck", truck)] = []
            
        for plane in range(self.objects["airplane"]):
            loc = rd.randint(0, self.objects["location"] - 1)
            self.state["vehicle_locations"][("airplane", plane)] = loc
            self.state["vehicle_contents"][("airplane", plane)] = []
        
        # Initialize location contents
        for loc in range(self.objects["location"]):
            self.state["location_contents"][loc] = {"packages": [], "vehicles": []}
        
        # Populate location contents based on initial positions
        for pkg, loc in self.state["package_locations"].items():
            if isinstance(loc, int):  # Package is at a location, not in a vehicle
                self.state["location_contents"][loc]["packages"].append(pkg)
        
        for vehicle, loc in self.state["vehicle_locations"].items():
            self.state["location_contents"][loc]["vehicles"].append(vehicle)

    # Query methods for state management
    def is_package_in_truck(self, pkg_id, truck_id):
        """Check if a specific package is in a specific truck"""
        truck_key = ("truck", truck_id)
        return pkg_id in self.state["vehicle_contents"].get(truck_key, [])
    
    def is_package_in_airplane(self, pkg_id, plane_id):
        """Check if a specific package is in a specific airplane"""
        plane_key = ("airplane", plane_id)
        return pkg_id in self.state["vehicle_contents"].get(plane_key, [])
    
    def is_package_in_vehicle(self, pkg_id, vehicle_type, vehicle_id):
        """Check if a specific package is in a specific vehicle"""
        vehicle_key = (vehicle_type, vehicle_id)
        return pkg_id in self.state["vehicle_contents"].get(vehicle_key, [])
    
    def is_truck_at_location(self, truck_id, location_id):
        """Check if a specific truck is at a specific location"""
        truck_key = ("truck", truck_id)
        return self.state["vehicle_locations"].get(truck_key) == location_id
    
    def is_airplane_at_location(self, plane_id, location_id):
        """Check if a specific airplane is at a specific location"""
        plane_key = ("airplane", plane_id)
        return self.state["vehicle_locations"].get(plane_key) == location_id
    
    def is_vehicle_at_location(self, vehicle_type, vehicle_id, location_id):
        """Check if a specific vehicle is at a specific location"""
        vehicle_key = (vehicle_type, vehicle_id)
        return self.state["vehicle_locations"].get(vehicle_key) == location_id
    
    def get_package_location(self, pkg_id):
        """Get the current location of a package (either location_id or vehicle tuple)"""
        return self.state["package_locations"].get(pkg_id)
    
    def get_vehicle_location(self, vehicle_type, vehicle_id):
        """Get the current location of a vehicle"""
        vehicle_key = (vehicle_type, vehicle_id)
        return self.state["vehicle_locations"].get(vehicle_key)
    
    def is_package_at_location(self, pkg_id, location_id):
        """Check if a package is at a specific location (not in a vehicle)"""
        pkg_location = self.state["package_locations"].get(pkg_id)
        return pkg_location == location_id
    
    def _validate_state_consistency(self):
        """Validate that all state representations are consistent"""
        # This method can be called after actions to ensure state integrity
        for pkg_id in range(self.objects["package"]):
            pkg_location = self.state["package_locations"][pkg_id]
            
            # If package is at a location, it should be in that location's contents
            if isinstance(pkg_location, int):
                assert pkg_id in self.state["location_contents"][pkg_location]["packages"], \
                    f"Package {pkg_id} location inconsistency"
            
            # If package is in a vehicle, it should be in that vehicle's contents
            elif isinstance(pkg_location, tuple):
                vehicle_key = pkg_location
                assert pkg_id in self.state["vehicle_contents"][vehicle_key], \
                    f"Package {pkg_id} vehicle inconsistency"

    def load(self, pkg, vehicle_type, vehicle_id, loc):
        """Load a package into a vehicle at a specific location"""
        # Check preconditions using centralized state
        
        # 1. Package must be at the specified location (not in any vehicle)
        if not self.is_package_at_location(pkg, loc):
            return False
        
        # 2. Vehicle must be at the same location
        if not self.is_vehicle_at_location(vehicle_type, vehicle_id, loc):
            return False
        
        # 3. Package must not already be in any vehicle
        for v_type in ["truck", "airplane"]:
            for v_id in range(self.objects[v_type]):
                if self.is_package_in_vehicle(pkg, v_type, v_id):
                    return False
        
        # Execute the action - update centralized state
        vehicle_key = (vehicle_type, vehicle_id)
        
        # Remove package from location
        self.state["location_contents"][loc]["packages"].remove(pkg)
        
        # Add package to vehicle
        self.state["vehicle_contents"][vehicle_key].append(pkg)
        
        # Update package location to vehicle
        self.state["package_locations"][pkg] = vehicle_key
        
        # Validate state consistency
        self._validate_state_consistency()
        
        return True

    def unload(self, pkg, vehicle_type, vehicle_id, loc):
        """Unload a package from a vehicle at a specific location"""
        # Check preconditions using centralized state
        
        # 1. Package must be in the specified vehicle
        if not self.is_package_in_vehicle(pkg, vehicle_type, vehicle_id):
            return False
        
        # 2. Vehicle must be at the specified location
        if not self.is_vehicle_at_location(vehicle_type, vehicle_id, loc):
            return False
        
        # Execute the action - update centralized state
        vehicle_key = (vehicle_type, vehicle_id)
        
        # Remove package from vehicle
        self.state["vehicle_contents"][vehicle_key].remove(pkg)
        
        # Add package to location
        self.state["location_contents"][loc]["packages"].append(pkg)
        
        # Update package location to the location
        self.state["package_locations"][pkg] = loc
        
        # Validate state consistency
        self._validate_state_consistency()
        
        return True

    def drive(self, truck_id, from_loc, to_loc):
        """Move a truck from one location to another"""
        # Check preconditions using centralized state
        
        # 1. Truck must be at the starting location
        if not self.is_truck_at_location(truck_id, from_loc):
            return False
        
        # 2. From and to locations must be different
        if from_loc == to_loc:
            return False
        
        # 3. To location must be valid
        if to_loc < 0 or to_loc >= self.objects["location"]:
            return False
        
        # Execute the action - update centralized state
        truck_key = ("truck", truck_id)
        
        # Remove truck from old location
        self.state["location_contents"][from_loc]["vehicles"].remove(truck_key)
        
        # Add truck to new location
        self.state["location_contents"][to_loc]["vehicles"].append(truck_key)
        
        # Update truck location
        self.state["vehicle_locations"][truck_key] = to_loc
        
        # Validate state consistency
        self._validate_state_consistency()
        
        return True

    def fly(self, plane_id, from_loc, to_loc):
        """Move an airplane from one location to another"""
        # Check preconditions using centralized state
        
        # 1. Airplane must be at the starting location
        if not self.is_airplane_at_location(plane_id, from_loc):
            return False
        
        # 2. From and to locations must be different
        if from_loc == to_loc:
            return False
        
        # 3. To location must be valid
        if to_loc < 0 or to_loc >= self.objects["location"]:
            return False
        
        # Execute the action - update centralized state
        plane_key = ("airplane", plane_id)
        
        # Remove airplane from old location
        self.state["location_contents"][from_loc]["vehicles"].remove(plane_key)
        
        # Add airplane to new location
        self.state["location_contents"][to_loc]["vehicles"].append(plane_key)
        
        # Update airplane location
        self.state["vehicle_locations"][plane_key] = to_loc
        
        # Validate state consistency
        self._validate_state_consistency()
        
        return True

    def choose_action(self):
        """Choose a random valid action using centralized state"""
        valid_actions = []
        
        # Try load actions - package at location + vehicle at same location
        for pkg in range(self.objects["package"]):
            pkg_location = self.get_package_location(pkg)
            
            # Package must be at a location (not in a vehicle)
            if isinstance(pkg_location, int):
                
                # Try loading into trucks at same location
                for truck in range(self.objects["truck"]):
                    if self.is_truck_at_location(truck, pkg_location):
                        valid_actions.append(("load", ("package", pkg), ("truck", truck), ("location", pkg_location)))
                
                # Try loading into airplanes at same location
                for plane in range(self.objects["airplane"]):
                    if self.is_airplane_at_location(plane, pkg_location):
                        valid_actions.append(("load", ("package", pkg), ("airplane", plane), ("location", pkg_location)))
        
        # Try unload actions - package in vehicle + vehicle at location
        for truck in range(self.objects["truck"]):
            truck_loc = self.get_vehicle_location("truck", truck)
            truck_key = ("truck", truck)
            for pkg in self.state["vehicle_contents"][truck_key]:
                valid_actions.append(("unload", ("package", pkg), ("truck", truck), ("location", truck_loc)))
        
        for plane in range(self.objects["airplane"]):
            plane_loc = self.get_vehicle_location("airplane", plane)
            plane_key = ("airplane", plane)
            for pkg in self.state["vehicle_contents"][plane_key]:
                valid_actions.append(("unload", ("package", pkg), ("airplane", plane), ("location", plane_loc)))
        
        # Try drive actions - move trucks between locations
        for truck in range(self.objects["truck"]):
            truck_loc = self.get_vehicle_location("truck", truck)
            for to_loc in range(self.objects["location"]):
                if to_loc != truck_loc:
                    valid_actions.append(("drive", ("truck", truck), ("location", truck_loc), ("location", to_loc)))
        
        # Try fly actions - move airplanes between locations
        for plane in range(self.objects["airplane"]):
            plane_loc = self.get_vehicle_location("airplane", plane)
            for to_loc in range(self.objects["location"]):
                if to_loc != plane_loc:
                    valid_actions.append(("fly", ("airplane", plane), ("location", plane_loc), ("location", to_loc)))
        
        if valid_actions:
            return rd.choice(valid_actions)
        else:
            # Fallback - just drive a truck somewhere
            truck_loc = self.get_vehicle_location("truck", 0)
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
        # Reserve first 10 indices for actions
        current_idx = 10
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
        """
        Generate predicate tokens by checking actual world state.
        The truth value of each predicate is determined by the specific states of the objects involved.
        For example: at_truck(block, truck) is true if block is currently in the specified truck.
        """
        preds = list(self.predicates_arity[type].keys())
        tokens = []
        
        for pred in preds:
            token = np.zeros(self.token_size)
            indx = self.token_map[pred]
            token[indx] = 1  # Mark predicate as active
            
            # Check actual world state for each predicate using centralized state
            predicate_value = False
            
            if pred == "at":
                if type == "package":
                    # Package "at" predicate: true if package is at a location (not in any vehicle)
                    pkg_location = self.get_package_location(obj)
                    predicate_value = isinstance(pkg_location, int)  # at location, not in vehicle
                    
                elif type == "truck":
                    # Truck "at" predicate: true if truck is at any location (always true since trucks are always somewhere)
                    truck_location = self.get_vehicle_location("truck", obj)
                    predicate_value = truck_location is not None
                    
                elif type == "airplane":
                    # Airplane "at" predicate: true if airplane is at any location (always true since airplanes are always somewhere)
                    plane_location = self.get_vehicle_location("airplane", obj)
                    predicate_value = plane_location is not None
                    
            elif pred == "in":
                if type == "package":
                    # Package "in" predicate: true if package is in any vehicle
                    pkg_location = self.get_package_location(obj)
                    predicate_value = isinstance(pkg_location, tuple)  # in vehicle (location is vehicle tuple)
                    
            elif pred == "loaded":
                if type == "package":
                    # Package "loaded" predicate: true if package is loaded in any vehicle (same as "in")
                    pkg_location = self.get_package_location(obj)
                    predicate_value = isinstance(pkg_location, tuple)  # loaded in vehicle
                    
            elif pred == "clear":
                if type == "location":
                    # Location "clear" predicate: true if location has no packages (vehicles can still be there)
                    # A location is clear if it has no packages at it
                    location_packages = self.state["location_contents"][obj]["packages"]
                    predicate_value = len(location_packages) == 0
            
            # Handle specific predicates that may be created for token differentiation
            elif pred.startswith("at_"):
                # Handle renamed predicates like "at_truck", "at_airplane", etc.
                base_pred = pred.split("_")[0]  # Extract "at" from "at_truck"
                
                if base_pred == "at":
                    if type == "package":
                        pkg_location = self.get_package_location(obj)
                        predicate_value = isinstance(pkg_location, int)
                    elif type == "truck":
                        truck_location = self.get_vehicle_location("truck", obj)
                        predicate_value = truck_location is not None
                    elif type == "airplane":
                        plane_location = self.get_vehicle_location("airplane", obj)
                        predicate_value = plane_location is not None
                        
            elif pred.startswith("in_"):
                # Handle renamed predicates like "in_package", etc.
                base_pred = pred.split("_")[0]  # Extract "in" from "in_package"
                
                if base_pred == "in" and type == "package":
                    pkg_location = self.get_package_location(obj)
                    predicate_value = isinstance(pkg_location, tuple)
                    
            elif pred.startswith("loaded_"):
                # Handle renamed predicates like "loaded_package", etc.
                base_pred = pred.split("_")[0]  # Extract "loaded" from "loaded_package"
                
                if base_pred == "loaded" and type == "package":
                    pkg_location = self.get_package_location(obj)
                    predicate_value = isinstance(pkg_location, tuple)
                    
            elif pred.startswith("clear_"):
                # Handle renamed predicates like "clear_location", etc.
                base_pred = pred.split("_")[0]  # Extract "clear" from "clear_location"
                
                if base_pred == "clear" and type == "location":
                    location_packages = self.state["location_contents"][obj]["packages"]
                    predicate_value = len(location_packages) == 0
            
            # Set the predicate value in the token
            if predicate_value:
                token[indx + 1] = 1
            
            token[indx + 2] = instance  # Set instance number
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
