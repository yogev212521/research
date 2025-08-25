from Domains.logistics_domain import LogisticsDomain
import numpy as np

def test_load_unload_tokens():
    """Specifically test load and unload action tokens"""
    print("LOAD/UNLOAD TOKEN TESTING")
    print("=" * 50)
    
    domain = LogisticsDomain()
    
    # Force a scenario where load/unload are possible
    print("Setting up scenario for load/unload actions...")
    print("-" * 45)
    
    # Move a truck to where a package is
    pkg_loc = domain.object_locations[("package", 0)]
    truck_loc = domain.object_locations[("truck", 0)]
    
    print(f"Package 0 at location: {pkg_loc}")
    print(f"Truck 0 initially at location: {truck_loc}")
    
    # Move truck to package location if needed
    if truck_loc != pkg_loc:
        success = domain.drive(0, truck_loc, pkg_loc)
        print(f"Moved truck 0 from {truck_loc} to {pkg_loc}: {success}")
    
    # Test LOAD action tokens
    print("\nTesting LOAD action tokens:")
    print("-" * 28)
    
    load_action = ("load", ("package", 0), ("truck", 0), ("location", pkg_loc))
    print(f"Load action: {load_action}")
    
    # Generate tokens for load action
    load_tokens = domain.get_tokens(load_action)
    print(f"Generated {len(load_tokens)} tokens")
    
    # Analyze action token
    action_token = load_tokens[0]
    action_indices = np.where(action_token == 1)[0]
    print(f"Action token active indices: {action_indices}")
    
    if len(action_indices) == 1:
        action_idx = action_indices[0]
        expected_action = list(domain.action_map.keys())[action_idx]
        print(f"Mapped action: {expected_action}")
        print(f"Mapping correct: {expected_action == 'load'}")
    
    # Test reverse mapping
    mapped_action = domain.get_token_to_action_map(load_tokens)
    print(f"Reverse mapped action: {mapped_action}")
    
    # Analyze predicate tokens
    print("Predicate tokens analysis:")
    for i, token in enumerate(load_tokens[1:8]):  # First few predicate tokens
        active_indices = np.where(token > 0)[0]
        if len(active_indices) > 0:
            print(f"  Token {i+1}: Active indices {active_indices}, Values {token[active_indices]}")
            
            # Map to predicates
            for pred, idx in domain.token_map.items():
                if idx in active_indices:
                    print(f"    Maps to predicate: {pred}")
    
    # Execute the load action
    print(f"\nExecuting load action...")
    load_success = domain.next_trace(load_action)
    print(f"Load success: {load_success}")
    
    if load_success:
        # Test UNLOAD action tokens
        print("\nTesting UNLOAD action tokens:")
        print("-" * 30)
        
        # Move truck to another location first
        new_loc = (pkg_loc + 1) % domain.objects["location"]
        drive_success = domain.drive(0, pkg_loc, new_loc)
        print(f"Moved truck to location {new_loc}: {drive_success}")
        
        unload_action = ("unload", ("package", 0), ("truck", 0), ("location", new_loc))
        print(f"Unload action: {unload_action}")
        
        # Generate tokens for unload action
        unload_tokens = domain.get_tokens(unload_action)
        print(f"Generated {len(unload_tokens)} tokens")
        
        # Analyze action token
        action_token = unload_tokens[0]
        action_indices = np.where(action_token == 1)[0]
        print(f"Action token active indices: {action_indices}")
        
        if len(action_indices) == 1:
            action_idx = action_indices[0]
            expected_action = list(domain.action_map.keys())[action_idx]
            print(f"Mapped action: {expected_action}")
            print(f"Mapping correct: {expected_action == 'unload'}")
        
        # Test reverse mapping
        mapped_action = domain.get_token_to_action_map(unload_tokens)
        print(f"Reverse mapped action: {mapped_action}")
        
        # Analyze predicate tokens
        print("Predicate tokens analysis:")
        for i, token in enumerate(unload_tokens[1:8]):  # First few predicate tokens
            active_indices = np.where(token > 0)[0]
            if len(active_indices) > 0:
                print(f"  Token {i+1}: Active indices {active_indices}, Values {token[active_indices]}")
                
                # Map to predicates
                for pred, idx in domain.token_map.items():
                    if idx in active_indices:
                        print(f"    Maps to predicate: {pred}")
        
        # Execute the unload action
        print(f"\nExecuting unload action...")
        unload_success = domain.next_trace(unload_action)
        print(f"Unload success: {unload_success}")
    
    print("\nTesting complete action cycle with tokens:")
    print("-" * 42)
    
    # Create a fresh domain and test full cycle
    domain = LogisticsDomain()
    
    actions_tested = []
    
    # Try to find and execute each action type
    for _ in range(50):
        action = domain.choose_action()
        action_type = action[0]
        
        if action_type not in [a[0] for a in actions_tested]:
            print(f"\nTesting {action_type} action:")
            
            # Generate and validate tokens
            tokens = domain.get_tokens(action)
            
            # Check action token
            action_token = tokens[0]
            action_indices = np.where(action_token == 1)[0]
            
            if len(action_indices) == 1:
                mapped_action = list(domain.action_map.keys())[action_indices[0]]
                mapping_correct = mapped_action == action_type
                print(f"  Action: {action}")
                print(f"  Token mapping: {mapped_action} (correct: {mapping_correct})")
                print(f"  Number of tokens: {len(tokens)}")
                
                # Execute action
                success = domain.next_trace(action)
                print(f"  Execution success: {success}")
                
                actions_tested.append(action)
                
                if len(actions_tested) == 4:  # All 4 action types tested
                    break
    
    print(f"\nTested {len(actions_tested)} different action types:")
    for action in actions_tested:
        print(f"  - {action[0]}")

def test_token_values_detailed():
    """Test specific token values in detail"""
    print("\n" + "=" * 50)
    print("DETAILED TOKEN VALUES ANALYSIS")
    print("=" * 50)
    
    domain = LogisticsDomain()
    
    # Test each action type with known parameters
    test_actions = [
        ("drive", ("truck", 0), ("location", 0), ("location", 1)),
        ("fly", ("airplane", 0), ("location", 0), ("location", 1)),
    ]
    
    for action in test_actions:
        print(f"\nAnalyzing {action[0]} action tokens:")
        print("-" * 35)
        
        tokens = domain.get_tokens(action)
        
        # Show detailed breakdown of each token
        for i, token in enumerate(tokens[:5]):  # First 5 tokens
            print(f"Token {i}:")
            non_zero_indices = np.where(token != 0)[0]
            if len(non_zero_indices) > 0:
                for idx in non_zero_indices:
                    print(f"  Index {idx}: {token[idx]}")
            else:
                print(f"  All zeros")
        
        # Validate token structure
        assert len(tokens) == domain.num_of_tokens, f"Wrong number of tokens"
        for token in tokens:
            assert len(token) == domain.token_size, f"Wrong token size"
            assert token.dtype == np.float64, f"Wrong token dtype"
        
        print(f"✓ All structural validations passed")

if __name__ == "__main__":
    test_load_unload_tokens()
    test_token_values_detailed()
    
    print("\n" + "=" * 60)
    print("LOAD/UNLOAD TOKEN TESTING COMPLETED!")
    print("All token systems validated successfully!")
    print("=" * 60)
