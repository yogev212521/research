from Domains.logistics_domain import LogisticsDomain
import numpy as np
import random

def test_token_consistency():
    """Test that tokens are generated consistently and correctly"""
    print("Testing Token Consistency")
    print("-" * 40)
    
    domain = LogisticsDomain()
    
    # Test token map
    print(f"Token map: {domain.token_map}")
    print(f"Token size: {domain.token_size}")
    print(f"Number of tokens per action: {domain.num_of_tokens}")
    
    # Generate many actions and test tokens
    for i in range(50):
        try:
            action = domain.choose_action()
            tokens = domain.get_tokens(action)
            
            # Validate token structure
            assert len(tokens) == domain.num_of_tokens, f"Wrong number of tokens: {len(tokens)}"
            
            for j, token in enumerate(tokens):
                assert len(token) == domain.token_size, f"Token {j} has wrong size: {len(token)}"
                assert isinstance(token, np.ndarray), f"Token {j} is not numpy array"
                
            # Test action token (first token)
            action_token = tokens[0]
            action_indices = np.where(action_token == 1)[0]
            assert len(action_indices) == 1, f"Action token should have exactly one 1, got {len(action_indices)}"
            
            if i % 10 == 0:
                print(f"Test {i+1}: Action {action[0]} - Tokens valid")
                
        except Exception as e:
            print(f"Error in token test {i+1}: {e}")
            break
    
    print("Token consistency tests completed!")
    print()

def test_domain_state_transitions():
    """Test state transitions through many actions"""
    print("Testing Domain State Transitions")
    print("-" * 40)
    
    domain = LogisticsDomain()
    
    print("Initial state:")
    print_domain_state(domain)
    
    successful_actions = 0
    failed_actions = 0
    
    for i in range(100):
        try:
            action = domain.choose_action()
            initial_state = get_state_snapshot(domain)
            
            success = domain.next_trace(action)
            
            if success:
                successful_actions += 1
                if i % 20 == 0:
                    print(f"Step {i+1}: {action} - SUCCESS")
            else:
                failed_actions += 1
                if i % 20 == 0:
                    print(f"Step {i+1}: {action} - FAILED")
            
            # Verify state consistency
            verify_state_consistency(domain)
            
        except Exception as e:
            print(f"Error in state transition test {i+1}: {e}")
            break
    
    print(f"\nState transition results:")
    print(f"Successful actions: {successful_actions}")
    print(f"Failed actions: {failed_actions}")
    print(f"Success rate: {successful_actions/(successful_actions+failed_actions)*100:.1f}%")
    
    print("\nFinal state:")
    print_domain_state(domain)
    print()

def test_action_types():
    """Test each action type extensively"""
    print("Testing Individual Action Types")
    print("-" * 40)
    
    action_counts = {"load": 0, "unload": 0, "drive": 0, "fly": 0}
    action_successes = {"load": 0, "unload": 0, "drive": 0, "fly": 0}
    
    for test_run in range(10):
        domain = LogisticsDomain()
        
        for i in range(50):
            action = domain.choose_action()
            action_type = action[0]
            action_counts[action_type] += 1
            
            success = domain.next_trace(action)
            if success:
                action_successes[action_type] += 1
    
    print("Action type analysis:")
    for action_type in action_counts:
        total = action_counts[action_type]
        successes = action_successes[action_type]
        if total > 0:
            success_rate = successes / total * 100
            print(f"  {action_type}: {successes}/{total} ({success_rate:.1f}% success)")
        else:
            print(f"  {action_type}: 0/0 (no actions generated)")
    print()

def test_probabilistic_actions():
    """Test probabilistic action execution"""
    print("Testing Probabilistic Actions")
    print("-" * 40)
    
    domain = LogisticsDomain()
    
    prob_successes = 0
    prob_total = 0
    
    for i in range(200):
        action = domain.choose_action()
        
        # Test probabilistic version
        prob_success = domain.prob_next_trace(action)
        prob_total += 1
        if prob_success:
            prob_successes += 1
    
    prob_rate = prob_successes / prob_total * 100
    print(f"Probabilistic action success rate: {prob_successes}/{prob_total} ({prob_rate:.1f}%)")
    print()

def test_token_mapping():
    """Test token to predicate mapping functionality"""
    print("Testing Token Mapping Functions")
    print("-" * 40)
    
    domain = LogisticsDomain()
    
    for i in range(20):
        action = domain.choose_action()
        tokens = domain.get_tokens(action)
        
        # Test action token mapping
        try:
            mapped_action = domain.get_token_to_action_map(tokens)
            print(f"Test {i+1}: Action {action[0]} -> Mapped {mapped_action}")
        except Exception as e:
            print(f"Error in action mapping {i+1}: {e}")
        
        # Test predicate token mapping for first few tokens
        for j in range(min(5, len(tokens))):
            try:
                pred_map = domain.get_token_to_pred_map(tokens, j)
                if j == 0:
                    print(f"  Predicate mappings available: {len(pred_map)} predicates")
            except Exception as e:
                print(f"  Error in predicate mapping token {j}: {e}")
    print()

def print_domain_state(domain):
    """Helper function to print current domain state"""
    print("  Object locations:")
    for obj_type in ["package", "truck", "airplane"]:
        for obj_id in range(domain.objects[obj_type]):
            location = domain.object_locations.get((obj_type, obj_id))
            if location is not None:
                print(f"    {obj_type} {obj_id}: location {location}")
            else:
                print(f"    {obj_type} {obj_id}: in vehicle")
    
    print("  Vehicle contents:")
    for vehicle_type in ["truck", "airplane"]:
        for vehicle_id in range(domain.objects[vehicle_type]):
            contents = domain.vehicle_contents.get((vehicle_type, vehicle_id), [])
            print(f"    {vehicle_type} {vehicle_id}: {contents}")

def get_state_snapshot(domain):
    """Get a snapshot of the current domain state"""
    return {
        'object_locations': dict(domain.object_locations),
        'vehicle_contents': dict(domain.vehicle_contents),
        'objects_to_propositions': dict(domain.objects_to_propositions)
    }

def verify_state_consistency(domain):
    """Verify that the domain state is consistent"""
    # Check that packages are either at locations or in vehicles, not both
    for pkg in range(domain.objects["package"]):
        pkg_key = ("package", pkg)
        location = domain.object_locations.get(pkg_key)
        
        # Count how many vehicles contain this package
        in_vehicles = 0
        for vehicle_key in domain.vehicle_contents:
            if pkg in domain.vehicle_contents[vehicle_key]:
                in_vehicles += 1
        
        # Package should be either at a location OR in exactly one vehicle
        if location is not None and in_vehicles > 0:
            raise Exception(f"Package {pkg} is both at location {location} and in {in_vehicles} vehicles")
        if location is None and in_vehicles != 1:
            raise Exception(f"Package {pkg} is not at any location but in {in_vehicles} vehicles")

def run_comprehensive_tests():
    """Run all comprehensive tests"""
    print("COMPREHENSIVE LOGISTICS DOMAIN TESTING")
    print("=" * 60)
    print()
    
    # Set random seed for reproducible tests
    random.seed(42)
    np.random.seed(42)
    
    test_token_consistency()
    test_domain_state_transitions()
    test_action_types()
    test_probabilistic_actions()
    test_token_mapping()
    
    print("ALL COMPREHENSIVE TESTS COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    run_comprehensive_tests()
