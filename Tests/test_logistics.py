from Domains.logistics_domain import LogisticsDomain

# Test the logistics domain
def test_logistics_domain():
    print("Testing Logistics Domain")
    print("=" * 40)
    
    # Initialize the domain
    domain = LogisticsDomain()
    
    print(f"Domain name: {domain.name}")
    print(f"Description: {domain.description}")
    print(f"Objects: {domain.objects}")
    print(f"Actions: {[action[0] for action in domain.actions]}")
    print()
    
    # Print initial state
    print("Initial State:")
    print("-" * 20)
    for obj_type in domain.types:
        if obj_type != "location":
            for obj_id in range(domain.objects[obj_type]):
                key = (obj_type, obj_id)
                location = domain.object_locations.get(key)
                print(f"{obj_type} {obj_id} at location {location}")
    print()
    
    # Test some actions
    print("Testing Actions:")
    print("-" * 20)
    
    for i in range(5):
        try:
            action = domain.choose_action()
            print(f"Action {i+1}: {action}")
            
            success = domain.next_trace(action)
            print(f"Success: {success}")
            
            # Get tokens for this action
            tokens = domain.get_tokens(action)
            print(f"Number of tokens: {len(tokens)}")
            print()
            
        except Exception as e:
            print(f"Error during action {i+1}: {e}")
            break
    
    print("Test completed successfully!")

if __name__ == "__main__":
    test_logistics_domain()
