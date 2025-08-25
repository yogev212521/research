from Domains.domain_generator import HanoiTowers

def test_hanoi_towers():
    # Create a new HanoiTowers instance
    hanoi = HanoiTowers()
    
    print("Initial state:")
    print(f"Disc positions: {hanoi.on}")
    print(f"Peg tops: {hanoi.peg_top}")
    print(f"Disc propositions: {hanoi.objects_to_propositions['disc']}")
    print(f"Peg propositions: {hanoi.objects_to_propositions['peg']}")
    print()
    
    # Test some moves
    print("Testing moves:")
    
    # Move disc 0 from peg 0 to peg 1
    print("Move disc 0 to peg 1:")
    success = hanoi.move(0, 1)
    print(f"Success: {success}")
    print(f"Disc positions: {hanoi.on}")
    print(f"Peg tops: {hanoi.peg_top}")
    print()
    
    # Move disc 1 from peg 0 to peg 2
    print("Move disc 1 to peg 2:")
    success = hanoi.move(1, 2)
    print(f"Success: {success}")
    print(f"Disc positions: {hanoi.on}")
    print(f"Peg tops: {hanoi.peg_top}")
    print()
    
    # Try invalid move - disc 2 on disc 0
    print("Try to move disc 0 to peg 2 (should fail - larger disc there):")
    success = hanoi.move(0, 2)
    print(f"Success: {success}")
    print()
    
    # Test choose_action
    print("Testing random action selection:")
    for i in range(5):
        action = hanoi.choose_action()
        print(f"Chosen action: {action}")
        if action:
            success = hanoi.next_trace(action)
            print(f"Executed successfully: {success}")
            print(f"New state - Disc positions: {hanoi.on}, Peg tops: {hanoi.peg_top}")
        print()

if __name__ == "__main__":
    test_hanoi_towers()
