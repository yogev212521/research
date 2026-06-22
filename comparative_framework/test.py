#!/usr/bin/env python3
"""
Quick test script to validate comparative_framework installation
Run this to verify all components work together
"""

import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported."""
    print("[test_imports] Checking imports...")
    
    try:
        from comparative_framework.config import DOMAINS, TRAINING_CONFIG
        print("  ✓ config.py")
    except Exception as e:
        print(f"  ✗ config.py: {e}")
        return False
    
    try:
        from comparative_framework.visual_extraction import get_visual_extractor
        print("  ✓ visual_extraction.py")
    except Exception as e:
        print(f"  ✗ visual_extraction.py: {e}")
        return False
    
    try:
        from comparative_framework.token_translator import TokenTranslator
        print("  ✓ token_translator.py")
    except Exception as e:
        print(f"  ✗ token_translator.py: {e}")
        return False
    
    try:
        from comparative_framework.trace_simulator import TraceSimulator
        print("  ✓ trace_simulator.py")
    except Exception as e:
        print(f"  ✗ trace_simulator.py: {e}")
        return False
    
    try:
        from comparative_framework.comparative_tester import ComparativeTest
        print("  ✓ comparative_tester.py")
    except Exception as e:
        print(f"  ✗ comparative_tester.py: {e}")
        return False
    
    return True


def test_visual_extraction():
    """Test visual extraction module."""
    print("\n[test_visual_extraction] Testing visual extraction...")
    
    try:
        import numpy as np
        from comparative_framework.visual_extraction import get_visual_extractor
        
        extractor = get_visual_extractor(domain="hanoi", device="cpu")
        print("  ✓ VisualExtractor initialized")
        
        # Test image extraction
        image = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
        preds, confs = extractor.extract_predicates_from_image(image)
        print(f"  ✓ Extracted {len(preds)} predicates from image")
        
        # Test sequence extraction
        images = np.random.randint(0, 256, (3, 28, 28), dtype=np.uint8)
        sequence = extractor.extract_sequence(images)
        print(f"  ✓ Extracted {len(sequence)}-step sequence")
        
        return True
    except Exception as e:
        print(f"  ✗ Visual extraction failed: {e}")
        return False


def test_token_translation():
    """Test token translation module."""
    print("\n[test_token_translation] Testing token translation...")
    
    try:
        import numpy as np
        from comparative_framework.token_translator import TokenTranslator
        from comparative_framework.config import DOMAINS
        
        translator = TokenTranslator(domain="hanoi")
        print("  ✓ TokenTranslator initialized")
        
        # Test action token
        action_token = translator.predicates_to_action_token(0)
        print(f"  ✓ Action token shape: {action_token.shape}")
        
        # Test state tokens
        num_props = DOMAINS["hanoi"]["article_baseline"]["num_propositions"]
        test_state = {f"p_{i}": (i % 2 == 0) for i in range(num_props)}
        pred_names = [f"p_{i}" for i in range(num_props)]
        
        state_seq = translator.state_dict_to_token_sequence(test_state, pred_names)
        print(f"  ✓ State sequence shape: {state_seq.shape}")
        
        return True
    except Exception as e:
        print(f"  ✗ Token translation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trace_simulator():
    """Test trace simulator module."""
    print("\n[test_trace_simulator] Testing trace simulator...")
    
    try:
        from comparative_framework.trace_simulator import TraceSimulator
        
        simulator = TraceSimulator(domain="hanoi")
        print("  ✓ TraceSimulator initialized")
        print(f"    - {simulator.num_propositions} propositions")
        print(f"    - {simulator.num_actions} actions")
        
        return True
    except Exception as e:
        print(f"  ✗ Trace simulator failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comparative_test():
    """Test comparative test module."""
    print("\n[test_comparative_test] Testing comparative test...")
    
    try:
        from comparative_framework.comparative_tester import ComparativeTest
        
        tester = ComparativeTest(domain="hanoi", device="cpu")
        print("  ✓ ComparativeTest initialized")
        
        # Generate small test traces
        pred_seqs, action_seqs = tester.generate_test_traces(num_traces=2, trace_length=3)
        print(f"  ✓ Generated {len(pred_seqs)} test traces")
        print(f"    - Each trace: {len(pred_seqs[0])} steps")
        
        return True
    except Exception as e:
        print(f"  ✗ Comparative test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("COMPARATIVE FRAMEWORK TEST SUITE")
    print("="*70)
    
    results = {
        "imports": test_imports(),
        "visual_extraction": test_visual_extraction(),
        "token_translation": test_token_translation(),
        "trace_simulator": test_trace_simulator(),
        "comparative_test": test_comparative_test(),
    }
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:30s} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All tests passed! Framework is ready to use.")
        print("\nNext steps:")
        print("1. Run: python -m comparative_framework.comparative_tester")
        print("2. Or: from comparative_framework import ComparativeTest")
    else:
        print("\n✗ Some tests failed. Check errors above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
