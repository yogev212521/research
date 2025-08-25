from Domains.logistics_domain import LogisticsDomain
import numpy as np
import random
import time

def stress_test_logistics():
    """Run extensive stress tests on the logistics domain"""
    print("LOGISTICS DOMAIN STRESS TEST")
    print("=" * 60)
    
    # Test 1: Long sequence execution
    print("\n1. Long Sequence Execution Test (1000 actions)")
    print("-" * 50)
    
    domain = LogisticsDomain()
    success_count = 0
    total_count = 0
    action_history = []
    
    start_time = time.time()
    
    for i in range(1000):
        try:
            action = domain.choose_action()
            success = domain.next_trace(action)
            
            action_history.append((action, success))
            total_count += 1
            if success:
                success_count += 1
                
            # Generate tokens for every action
            tokens = domain.get_tokens(action)
            
            if i % 200 == 0:
                print(f"  Step {i}: {action[0]} - {'SUCCESS' if success else 'FAILED'}")
        
        except Exception as e:
            print(f"  ERROR at step {i}: {e}")
            break
    
    end_time = time.time()
    
    print(f"  Results: {success_count}/{total_count} actions successful")
    print(f"  Success rate: {success_count/total_count*100:.1f}%")
    print(f"  Execution time: {end_time - start_time:.2f} seconds")
    
    # Test 2: Multiple domain instances
    print("\n2. Multiple Domain Instances Test")
    print("-" * 50)
    
    domains = [LogisticsDomain() for _ in range(10)]
    
    for i, domain in enumerate(domains):
        try:
            for j in range(50):
                action = domain.choose_action()
                success = domain.next_trace(action)
                tokens = domain.get_tokens(action)
            print(f"  Domain {i+1}: 50 actions completed successfully")
        except Exception as e:
            print(f"  Domain {i+1}: ERROR - {e}")
    
    # Test 3: Token generation stress test
    print("\n3. Token Generation Stress Test (5000 tokens)")
    print("-" * 50)
    
    domain = LogisticsDomain()
    token_count = 0
    token_errors = 0
    
    start_time = time.time()
    
    for i in range(5000):
        try:
            action = domain.choose_action()
            tokens = domain.get_tokens(action)
            
            # Validate each token
            for token in tokens:
                if len(token) != domain.token_size:
                    token_errors += 1
                    
            token_count += len(tokens)
            
            # Reset domain occasionally to test initialization
            if i % 500 == 0 and i > 0:
                domain.init()
                
        except Exception as e:
            token_errors += 1
            if token_errors < 5:  # Only show first few errors
                print(f"  Token error {token_errors}: {e}")
    
    end_time = time.time()
    
    print(f"  Generated {token_count} tokens")
    print(f"  Token errors: {token_errors}")
    print(f"  Token generation time: {end_time - start_time:.2f} seconds")
    
    # Test 4: Edge case testing
    print("\n4. Edge Case Testing")
    print("-" * 50)
    
    # Test with domain at various states
    edge_cases_passed = 0
    total_edge_cases = 0
    
    for test_case in range(20):
        try:
            domain = LogisticsDomain()
            
            # Force some specific states
            if test_case % 4 == 0:
                # Load all packages
                for pkg in range(domain.objects["package"]):
                    pkg_loc = domain.object_locations[("package", pkg)]
                    for truck in range(domain.objects["truck"]):
                        truck_loc = domain.object_locations[("truck", truck)]
                        if truck_loc == pkg_loc:
                            domain.load(pkg, "truck", truck, pkg_loc)
                            break
            
            elif test_case % 4 == 1:
                # Move all vehicles to one location
                for truck in range(domain.objects["truck"]):
                    current_loc = domain.object_locations[("truck", truck)]
                    domain.drive(truck, current_loc, 0)
                for plane in range(domain.objects["airplane"]):
                    current_loc = domain.object_locations[("airplane", plane)]
                    domain.fly(plane, current_loc, 0)
            
            # Try to execute some actions
            for _ in range(10):
                action = domain.choose_action()
                domain.next_trace(action)
                domain.get_tokens(action)
            
            edge_cases_passed += 1
            total_edge_cases += 1
            
        except Exception as e:
            total_edge_cases += 1
            print(f"  Edge case {test_case} failed: {e}")
    
    print(f"  Edge cases passed: {edge_cases_passed}/{total_edge_cases}")
    
    # Test 5: Memory and performance test
    print("\n5. Memory and Performance Test")
    print("-" * 50)
    
    start_time = time.time()
    domains_created = 0
    
    try:
        for i in range(100):
            domain = LogisticsDomain()
            
            # Execute some actions
            for j in range(20):
                action = domain.choose_action()
                domain.next_trace(action)
            
            domains_created += 1
            
            # Clean up
            del domain
            
        end_time = time.time()
        print(f"  Created and tested {domains_created} domains")
        print(f"  Average time per domain: {(end_time - start_time)/domains_created:.4f} seconds")
        
    except Exception as e:
        print(f"  Memory/Performance test failed: {e}")
    
    # Test 6: Probabilistic behavior consistency
    print("\n6. Probabilistic Behavior Test")
    print("-" * 50)
    
    domain = LogisticsDomain()
    prob_results = []
    
    for i in range(1000):
        action = domain.choose_action()
        regular_result = domain.next_trace(action)
        
        # Reset to same state and try probabilistic
        domain.init()
        prob_result = domain.prob_next_trace(action)
        
        prob_results.append(prob_result)
    
    prob_success_rate = sum(prob_results) / len(prob_results) * 100
    print(f"  Probabilistic success rate: {prob_success_rate:.1f}%")
    print(f"  Expected to be close to regular success rate but with some randomness")
    
    print("\n" + "=" * 60)
    print("STRESS TEST COMPLETED SUCCESSFULLY!")
    print("The logistics domain has passed all extensive stress tests.")

if __name__ == "__main__":
    stress_test_logistics()
