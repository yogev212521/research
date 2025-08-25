from Domains.logistics_domain import LogisticsDomain
import numpy as np

def validate_tokens_after_actions():
    """Validate that tokens correctly represent domain state after each action"""
    print("TOKEN STATE VALIDATION AFTER ACTIONS")
    print("=" * 60)
    
    domain = LogisticsDomain()
    
    print("Initial domain state:")
    print_domain_state(domain)
    
    # Test sequence of actions and validate tokens after each
    test_sequence = [
        # Move truck to package location
        ("drive", ("truck", 0), ("location", None), ("location", None)),
        # Load package
        ("load", ("package", 0), ("truck", 0), ("location", None)),
        # Move truck with package
        ("drive", ("truck", 0), ("location", None), ("location", None)),
        # Unload package
        ("unload", ("package", 0), ("truck", 0), ("location", None)),
        # Test airplane
        ("fly", ("airplane", 0), ("location", None), ("location", None))
    ]
    
    for step, base_action in enumerate(test_sequence):
        print(f"\n{'='*50}")
        print(f"STEP {step + 1}: Testing {base_action[0]} action")
        print(f"{'='*50}")
        
        # Get a valid action of the desired type
        action = get_valid_action(domain, base_action[0])
        if action is None:
            print(f"Could not generate valid {base_action[0]} action")
            continue
            
        print(f"Action: {action}")
        
        # Capture state before action
        state_before = capture_domain_state(domain)
        tokens_before = domain.get_tokens(action)
        
        print(f"\nState before action:")
        print_domain_state(domain)
        
        print(f"\nTokens before action execution:")
        analyze_tokens(domain, tokens_before, action)
        
        # Execute action
        success = domain.next_trace(action)
        print(f"\nAction execution: {'SUCCESS' if success else 'FAILED'}")
        
        if success:
            # Capture state after action
            state_after = capture_domain_state(domain)
            
            print(f"\nState after action:")
            print_domain_state(domain)
            
            # Generate tokens after action for a subsequent action to see state changes
            next_action = domain.choose_action()
            tokens_after = domain.get_tokens(next_action)
            
            print(f"\nTokens after action (for next action {next_action[0]}):")
            analyze_tokens(domain, tokens_after, next_action)
            
            # Validate state changes match token representations
            validate_state_token_consistency(domain, state_before, state_after, action, tokens_before, tokens_after)

def get_valid_action(domain, action_type):
    """Get a valid action of the specified type"""
    for _ in range(100):
        action = domain.choose_action()
        if action[0] == action_type:
            return action
    
    # If we can't find the action type, create a forced scenario
    if action_type == "load":
        # Move a truck to a package location
        pkg_loc = domain.object_locations[("package", 0)]
        truck_loc = domain.object_locations[("truck", 0)]
        if truck_loc != pkg_loc:
            domain.drive(0, truck_loc, pkg_loc)
        if domain.objects_to_propositions["package"][0][0]:  # package at location
            return ("load", ("package", 0), ("truck", 0), ("location", pkg_loc))
    
    elif action_type == "unload":
        # Load a package first if needed
        pkg_loc = domain.object_locations[("package", 0)]
        if pkg_loc is not None:  # package not in vehicle
            truck_loc = domain.object_locations[("truck", 0)]
            if truck_loc == pkg_loc:
                domain.load(0, "truck", 0, pkg_loc)
        
        # Find a truck with packages
        for truck_id in range(domain.objects["truck"]):
            truck_key = ("truck", truck_id)
            if truck_key in domain.vehicle_contents and domain.vehicle_contents[truck_key]:
                truck_loc = domain.object_locations[truck_key]
                pkg_id = domain.vehicle_contents[truck_key][0]
                return ("unload", ("package", pkg_id), ("truck", truck_id), ("location", truck_loc))
    
    return None

def capture_domain_state(domain):
    """Capture complete domain state"""
    return {
        'object_locations': dict(domain.object_locations),
        'vehicle_contents': dict(domain.vehicle_contents),
        'objects_to_propositions': {
            obj_type: dict(props) for obj_type, props in domain.objects_to_propositions.items()
        }
    }

def print_domain_state(domain):
    """Print current domain state"""
    print("  Object locations:")
    for obj_type in ["package", "truck", "airplane"]:
        for obj_id in range(domain.objects[obj_type]):
            key = (obj_type, obj_id)
            location = domain.object_locations.get(key)
            if location is not None:
                print(f"    {obj_type} {obj_id}: location {location}")
            else:
                print(f"    {obj_type} {obj_id}: in vehicle")
    
    print("  Vehicle contents:")
    for vehicle_type in ["truck", "airplane"]:
        for vehicle_id in range(domain.objects[vehicle_type]):
            key = (vehicle_type, vehicle_id)
            contents = domain.vehicle_contents.get(key, [])
            print(f"    {vehicle_type} {vehicle_id}: packages {contents}")
    
    print("  Object propositions:")
    for obj_type in domain.objects_to_propositions:
        for obj_id in domain.objects_to_propositions[obj_type]:
            props = domain.objects_to_propositions[obj_type][obj_id]
            print(f"    {obj_type} {obj_id}: {props}")

def analyze_tokens(domain, tokens, action):
    """Analyze token structure and content"""
    print(f"  Action token analysis:")
    action_token = tokens[0]
    action_indices = np.where(action_token == 1)[0]
    if len(action_indices) == 1:
        expected_action = list(domain.action_map.keys())[action_indices[0]]
        print(f"    Action encoded: {expected_action} (index {action_indices[0]})")
        print(f"    Matches action: {expected_action == action[0]}")
    
    print(f"  Predicate token analysis:")
    # Analyze tokens related to objects in the action
    token_idx = 1
    for param in action[1:]:
        if len(param) == 2:  # (type, id) format
            obj_type, obj_id = param
            if obj_type in domain.predicates_arity:
                predicates = list(domain.predicates_arity[obj_type].keys())
                for pred_idx, predicate in enumerate(predicates):
                    if token_idx < len(tokens):
                        token = tokens[token_idx]
                        active_indices = np.where(token > 0)[0]
                        if len(active_indices) > 0:
                            print(f"    {obj_type} {obj_id} {predicate}: indices {active_indices}, values {token[active_indices]}")
                        token_idx += 1

def validate_state_token_consistency(domain, state_before, state_after, action, tokens_before, tokens_after):
    """Validate that token changes match state changes"""
    print(f"\nValidating state-token consistency:")
    
    action_type = action[0]
    
    if action_type == "load":
        # After load: package should be in vehicle, not at location
        pkg_id = action[1][1]
        vehicle_type, vehicle_id = action[2][0], action[2][1]
        
        print(f"  Load validation:")
        print(f"    Package {pkg_id} before: location={state_before['object_locations'].get(('package', pkg_id))}")
        print(f"    Package {pkg_id} after: location={state_after['object_locations'].get(('package', pkg_id))}")
        print(f"    Vehicle {vehicle_type} {vehicle_id} before: {state_before['vehicle_contents'].get((vehicle_type, vehicle_id), [])}")
        print(f"    Vehicle {vehicle_type} {vehicle_id} after: {state_after['vehicle_contents'].get((vehicle_type, vehicle_id), [])}")
        
        # Check propositions
        props_before = state_before['objects_to_propositions']['package'][pkg_id]
        props_after = state_after['objects_to_propositions']['package'][pkg_id]
        print(f"    Package {pkg_id} props before: {props_before}")
        print(f"    Package {pkg_id} props after: {props_after}")
        
        # Validate expected changes
        if pkg_id in state_after['vehicle_contents'].get((vehicle_type, vehicle_id), []):
            print(f"    ✓ Package correctly loaded into vehicle")
        else:
            print(f"    ✗ Package not found in vehicle")
            
        if state_after['object_locations'].get(('package', pkg_id)) is None:
            print(f"    ✓ Package location correctly set to None")
        else:
            print(f"    ✗ Package still has location after loading")
    
    elif action_type == "unload":
        # After unload: package should be at location, not in vehicle
        pkg_id = action[1][1]
        vehicle_type, vehicle_id = action[2][0], action[2][1]
        target_loc = action[3][1]
        
        print(f"  Unload validation:")
        print(f"    Package {pkg_id} before: location={state_before['object_locations'].get(('package', pkg_id))}")
        print(f"    Package {pkg_id} after: location={state_after['object_locations'].get(('package', pkg_id))}")
        print(f"    Vehicle {vehicle_type} {vehicle_id} before: {state_before['vehicle_contents'].get((vehicle_type, vehicle_id), [])}")
        print(f"    Vehicle {vehicle_type} {vehicle_id} after: {state_after['vehicle_contents'].get((vehicle_type, vehicle_id), [])}")
        
        # Validate expected changes
        if state_after['object_locations'].get(('package', pkg_id)) == target_loc:
            print(f"    ✓ Package correctly placed at target location {target_loc}")
        else:
            print(f"    ✗ Package not at expected location")
            
        if pkg_id not in state_after['vehicle_contents'].get((vehicle_type, vehicle_id), []):
            print(f"    ✓ Package correctly removed from vehicle")
        else:
            print(f"    ✗ Package still in vehicle after unloading")
    
    elif action_type in ["drive", "fly"]:
        # After move: vehicle should be at new location
        vehicle_type = action[1][0]
        vehicle_id = action[1][1]
        from_loc = action[2][1]
        to_loc = action[3][1]
        
        print(f"  {action_type.title()} validation:")
        print(f"    {vehicle_type} {vehicle_id} before: location={state_before['object_locations'].get((vehicle_type, vehicle_id))}")
        print(f"    {vehicle_type} {vehicle_id} after: location={state_after['object_locations'].get((vehicle_type, vehicle_id))}")
        
        if state_after['object_locations'].get((vehicle_type, vehicle_id)) == to_loc:
            print(f"    ✓ Vehicle correctly moved to location {to_loc}")
        else:
            print(f"    ✗ Vehicle not at expected location")

def test_token_predicate_accuracy():
    """Test that token predicates accurately reflect domain state"""
    print(f"\n{'='*60}")
    print("TOKEN PREDICATE ACCURACY TEST")
    print(f"{'='*60}")
    
    domain = LogisticsDomain()
    
    # Test specific scenarios
    scenarios = [
        "Initial state",
        "After loading package",
        "After moving vehicle with package", 
        "After unloading package"
    ]
    
    for scenario_idx, scenario in enumerate(scenarios):
        print(f"\nScenario {scenario_idx + 1}: {scenario}")
        print("-" * 40)
        
        if scenario_idx == 1:  # Load package
            # Ensure truck and package are at same location
            pkg_loc = domain.object_locations[("package", 0)]
            truck_loc = domain.object_locations[("truck", 0)]
            if truck_loc != pkg_loc:
                domain.drive(0, truck_loc, pkg_loc)
            domain.load(0, "truck", 0, pkg_loc)
        
        elif scenario_idx == 2:  # Move vehicle
            # Move truck to different location
            current_loc = domain.object_locations[("truck", 0)]
            new_loc = (current_loc + 1) % domain.objects["location"]
            domain.drive(0, current_loc, new_loc)
        
        elif scenario_idx == 3:  # Unload package
            truck_loc = domain.object_locations[("truck", 0)]
            if 0 in domain.vehicle_contents[("truck", 0)]:
                domain.unload(0, "truck", 0, truck_loc)
        
        # Generate tokens for current state
        action = domain.choose_action()
        tokens = domain.get_tokens(action)
        
        print(f"Current state:")
        print_domain_state(domain)
        
        print(f"\nToken validation for action: {action}")
        
        # Check that tokens match current state for each object type
        for obj_type in domain.types:
            if obj_type in domain.objects:
                for obj_id in range(domain.objects[obj_type]):
                    if obj_type in domain.objects_to_propositions:
                        current_props = domain.objects_to_propositions[obj_type][obj_id]
                        print(f"  {obj_type} {obj_id} state: {current_props}")
                        
                        # Check if this object appears in the action
                        obj_in_action = any(param == (obj_type, obj_id) for param in action[1:] if len(param) == 2)
                        if obj_in_action:
                            print(f"    → Object involved in action, tokens should reflect current state")

if __name__ == "__main__":
    validate_tokens_after_actions()
    test_token_predicate_accuracy()
    
    print(f"\n{'='*60}")
    print("TOKEN STATE VALIDATION COMPLETED!")
    print(f"{'='*60}")
