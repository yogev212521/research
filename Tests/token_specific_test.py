from Domains.logistics_domain import LogisticsDomain
import numpy as np
import random

def test_token_structure():
    """Test the detailed structure of tokens"""
    print("TOKEN STRUCTURE ANALYSIS")
    print("=" * 50)
    
    domain = LogisticsDomain()
    
    print(f"Domain configuration:")
    print(f"  Token size: {domain.token_size}")
    print(f"  Number of tokens per action: {domain.num_of_tokens}")
    print(f"  Token map: {domain.token_map}")
    print()
    
    # Test each action type
    action_types = ["load", "unload", "drive", "fly"]
    
    for action_type in action_types:
        print(f"Testing {action_type.upper()} action tokens:")
        print("-" * 30)
        
        # Generate multiple actions of this type
        attempts = 0
        found_action = False
        
        while attempts < 100 and not found_action:
            action = domain.choose_action()
            if action[0] == action_type:
                found_action = True
                tokens = domain.get_tokens(action)
                
                print(f"  Action: {action}")
                print(f"  Number of tokens: {len(tokens)}")
                
                # Analyze action token (first token)
                action_token = tokens[0]
                print(f"  Action token shape: {action_token.shape}")
                
                # Find which index is set to 1 in action token
                action_indices = np.where(action_token == 1)[0]
                print(f"  Action token active indices: {action_indices}")
                
                if len(action_indices) == 1:
                    action_idx = action_indices[0]
                    expected_action = list(domain.action_map.keys())[action_idx]
                    print(f"  Mapped action: {expected_action}")
                    print(f"  Mapping correct: {expected_action == action_type}")
                else:
                    print(f"  ERROR: Action token should have exactly 1 active index")
                
                # Analyze predicate tokens
                print(f"  Predicate tokens analysis:")
                for i, token in enumerate(tokens[1:6]):  # First few predicate tokens
                    active_indices = np.where(token > 0)[0]
                    if len(active_indices) > 0:
                        print(f"    Token {i+1}: Active indices {active_indices}, Values {token[active_indices]}")
                
                print()
            attempts += 1
        
        if not found_action:
            print(f"  Could not generate {action_type} action in 100 attempts")
            print()

def test_token_consistency_across_states():
    """Test that tokens are consistent across different domain states"""
    print("TOKEN CONSISTENCY ACROSS STATES")
    print("=" * 50)
    
    # Test same action in different states
    print("Testing same action type in different domain states:")
    print("-" * 45)
    
    token_patterns = {}
    
    for state_test in range(5):
        domain = LogisticsDomain()
        
        # Modify domain state
        if state_test == 1:
            # Load some packages
            pkg_loc = domain.object_locations[("package", 0)]
            for truck in range(domain.objects["truck"]):
                if domain.object_locations[("truck", truck)] == pkg_loc:
                    domain.load(0, "truck", truck, pkg_loc)
                    break
        
        elif state_test == 2:
            # Move vehicles around
            for truck in range(domain.objects["truck"]):
                current_loc = domain.object_locations[("truck", truck)]
                new_loc = (current_loc + 1) % domain.objects["location"]
                domain.drive(truck, current_loc, new_loc)
        
        # Generate tokens for drive action
        action = ("drive", ("truck", 0), ("location", 0), ("location", 1))
        try:
            tokens = domain.get_tokens(action)
            
            # Analyze action token consistency
            action_token = tokens[0]
            action_pattern = tuple(np.where(action_token == 1)[0])
            
            if "drive" not in token_patterns:
                token_patterns["drive"] = action_pattern
            
            print(f"  State {state_test}: Action token pattern {action_pattern}")
            print(f"    Consistent with previous: {token_patterns['drive'] == action_pattern}")
            
        except Exception as e:
            print(f"  State {state_test}: Error generating tokens - {e}")
    
    print()

def test_token_mapping_functions():
    """Test all token mapping functions thoroughly"""
    print("TOKEN MAPPING FUNCTIONS TEST")
    print("=" * 50)
    
    domain = LogisticsDomain()
    
    # Test action-to-token and token-to-action mapping
    print("Action <-> Token mapping test:")
    print("-" * 30)
    
    for _ in range(20):
        action = domain.choose_action()
        tokens = domain.get_tokens(action)
        
        # Test action token generation
        action_token = domain.get_actionTokens(action)
        print(f"Action: {action[0]}")
        print(f"  Generated action token indices: {np.where(action_token == 1)[0]}")
        
        # Test reverse mapping
        mapped_action = domain.get_token_to_action_map(tokens)
        print(f"  Mapped back to: {mapped_action}")
        print(f"  Mapping correct: {mapped_action == action[0]}")
        
        # Test predicate mapping for first few tokens
        for i in range(min(3, len(tokens))):
            try:
                pred_map = domain.get_token_to_pred_map(tokens, i)
                if i == 0:
                    print(f"  Available predicate mappings: {len(pred_map)}")
            except Exception as e:
                print(f"  Predicate mapping error for token {i}: {e}")
        
        print()
        break  # Just test one for detailed output

def test_token_values_and_ranges():
    """Test that token values are within expected ranges"""
    print("TOKEN VALUES AND RANGES TEST")
    print("=" * 50)
    
    domain = LogisticsDomain()
    
    all_token_values = []
    min_vals = []
    max_vals = []
    
    print("Analyzing token value distributions:")
    print("-" * 35)
    
    for i in range(100):
        action = domain.choose_action()
        tokens = domain.get_tokens(action)
        
        for token in tokens:
            all_token_values.extend(token.flatten())
            min_vals.append(np.min(token))
            max_vals.append(np.max(token))
    
    all_values = np.array(all_token_values)
    unique_values = np.unique(all_values)
    
    print(f"  Total token values analyzed: {len(all_values)}")
    print(f"  Unique values found: {unique_values}")
    print(f"  Overall min value: {np.min(all_values)}")
    print(f"  Overall max value: {np.max(all_values)}")
    print(f"  Value range is valid: {np.min(all_values) >= 0}")
    
    # Check for any unexpected values
    expected_range = set(range(0, max(domain.objects.values()) + 1)) | {0, 1}
    unexpected = set(unique_values) - expected_range
    if unexpected:
        print(f"  WARNING: Unexpected values found: {unexpected}")
    else:
        print(f"  All values within expected range ✓")
    
    print()

def test_token_reproducibility():
    """Test that identical actions produce identical tokens"""
    print("TOKEN REPRODUCIBILITY TEST")
    print("=" * 50)
    
    print("Testing token reproducibility for identical actions:")
    print("-" * 48)
    
    # Create same action multiple times
    test_action = ("drive", ("truck", 0), ("location", 0), ("location", 1))
    
    token_sets = []
    
    for i in range(5):
        domain = LogisticsDomain()
        
        # Set same initial state
        random.seed(42)
        np.random.seed(42)
        
        tokens = domain.get_tokens(test_action)
        token_sets.append([token.copy() for token in tokens])
        
        print(f"  Run {i+1}: Generated {len(tokens)} tokens")
    
    # Compare all token sets
    all_identical = True
    for i in range(1, len(token_sets)):
        for j in range(len(token_sets[0])):
            if not np.array_equal(token_sets[0][j], token_sets[i][j]):
                all_identical = False
                print(f"  Token {j} differs between run 1 and run {i+1}")
    
    if all_identical:
        print(f"  ✓ All token sets are identical - reproducibility confirmed")
    else:
        print(f"  ✗ Token sets differ - reproducibility issue detected")
    
    print()

def test_predicate_token_encoding():
    """Test how predicates are encoded in tokens"""
    print("PREDICATE TOKEN ENCODING TEST")
    print("=" * 50)
    
    domain = LogisticsDomain()
    
    print("Analyzing predicate encoding:")
    print("-" * 28)
    
    # Test with a known state
    print(f"Predicate-to-index mapping: {domain.token_map}")
    
    # Generate action and analyze predicate tokens
    action = domain.choose_action()
    tokens = domain.get_tokens(action)
    
    print(f"Action: {action}")
    print(f"Object proposition states:")
    
    # Show how each object type's predicates are encoded
    for obj_type in domain.types:
        if obj_type in domain.predicates_arity:
            print(f"  {obj_type} predicates: {list(domain.predicates_arity[obj_type].keys())}")
            
            # Show state for first object of this type
            if obj_type in domain.objects and domain.objects[obj_type] > 0:
                obj_props = domain.objects_to_propositions[obj_type][0]
                print(f"    {obj_type} 0 state: {obj_props}")
    
    print()
    
    # Analyze specific predicate tokens
    print("Detailed token analysis:")
    print("-" * 22)
    
    for i, token in enumerate(tokens[:10]):  # First 10 tokens
        active_indices = np.where(token > 0)[0]
        if len(active_indices) > 0:
            print(f"  Token {i}: Active indices {active_indices}")
            print(f"    Values: {token[active_indices]}")
            
            # Try to identify which predicate this represents
            for pred, idx in domain.token_map.items():
                if idx in active_indices:
                    print(f"    Maps to predicate: {pred}")
            print()

def run_token_tests():
    """Run all token-specific tests"""
    print("COMPREHENSIVE TOKEN TESTING")
    print("=" * 60)
    print()
    
    test_token_structure()
    test_token_consistency_across_states()
    test_token_mapping_functions()
    test_token_values_and_ranges()
    test_token_reproducibility()
    test_predicate_token_encoding()
    
    print("=" * 60)
    print("ALL TOKEN TESTS COMPLETED!")

if __name__ == "__main__":
    run_token_tests()
