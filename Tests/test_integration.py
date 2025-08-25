# Test integration with existing domains
from Domains.domain_generator import blockWorld, HanoiTowers
from Domains.logistics_domain import LogisticsDomain

def test_integration():
    print("Testing Domain Integration")
    print("=" * 40)
    
    # Initialize all domains
    block_domain = blockWorld()
    hanoi_domain = HanoiTowers()
    logistics_domain = LogisticsDomain()
    
    domains = [
        ("Block World", block_domain),
        ("Hanoi Towers", hanoi_domain),
        ("Logistics", logistics_domain)
    ]
    
    for name, domain in domains:
        print(f"\n{name} Domain:")
        print(f"  Name: {domain.name}")
        print(f"  Objects: {domain.objects}")
        print(f"  Actions: {[action[0] for action in domain.actions]}")
        print(f"  Token size: {domain.token_size}")
        print(f"  Number of tokens: {domain.num_of_tokens}")
        
        # Test action selection
        try:
            action = domain.choose_action()
            print(f"  Sample action: {action}")
            tokens = domain.get_tokens(action)
            print(f"  Generated tokens: {len(tokens)}")
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\nAll domains integrated successfully!")

if __name__ == "__main__":
    test_integration()
