from Domains.domain_generator import blockWorld
import torch
from pprint import pprint
from Domains.logistics_domain import LogisticsDomain 
import numpy as np
import Model.att_only as att_only

# Create a modified domain class with 60 token size and action buffer 10
class LogisticsDomain60(LogisticsDomain):
    def __init__(self):
        super().__init__()
        self.token_size = 60  # Token size is 60
        self.num_of_tokens = 60  # Number of tokens is 60
        self.action_buffer = 10  # Action buffer is 10

def train_logistics_model(prob=False, model=None, domain=None):
    """Train model on logistics domain"""
    print("Training on Logistics Domain...")
    batch_size = 20
    seq_length = 20
    iterations = 20
    traces = []
    for _ in range(100):  # Generate 100 traces
        trace = generate_trace_sequence(domain, seq_length=seq_length, prob=prob)
        traces.append(trace)
    traces = np.array(traces)
    traces = traces.reshape(traces.shape[0] // batch_size, batch_size, *traces.shape[1:])  # Reshape to match expected input dimensions
    model = att_only.Att_PAM(output_dim=domain.token_size, embed_dim=256, input_dim=domain.token_size, head_number=8) if model is None else model
    model.train(traces, lr=0.0005, iterations=iterations)
    print("Logistics domain training completed.")
    return model

def train_blockworld_model(prob=False, model=None, domain=None, logistics_domain=None):
    """Train model on blockworld domain with predicates starting from index 30"""
    print("Training on Blockworld Domain...")
    batch_size = 20
    seq_length = 8
    iterations = 20
    traces = []
    
    # Calculate action start index based on logistics domain actions
    logistics_action_count = len(logistics_domain.action_map)
    action_start_idx = logistics_action_count
    
    for _ in range(100):  # Generate 100 traces
        trace = generate_trace_sequence_blockworld(domain, seq_length=seq_length, prob=prob, 
                                                 pred_start=30, action_start=action_start_idx)
        traces.append(trace)
    traces = np.array(traces)
    traces = traces.reshape(traces.shape[0] // batch_size, batch_size, *traces.shape[1:])  # Reshape to match expected input dimensions
    model.train(traces, lr=0.0005, iterations=iterations)
    print("Blockworld domain training completed.")
    return model

def generate_trace_sequence(domain, seq_length: int, prob=False):
    """Generate trace sequence for logistics domain"""
    sequence = []
    for _ in range(seq_length):  # Generate sequence of actions
        next_action = domain.choose_action()
        # Get tokens BEFORE executing the action
        tokens_before = domain.get_tokens(next_action)
        tokens_before = np.array(tokens_before)
        
        if prob:
            success = domain.prob_next_trace(next_action)
        else:
            success = domain.next_trace(next_action)
        
        if success:
            # Get tokens AFTER executing the action
            tokens_after = domain.get_tokens(next_action)
            tokens_after = np.array(tokens_after)
            
            sequence.append(tokens_before)  # State before action
            sequence.append(tokens_after)   # State after action
        else:
            raise ValueError(f"Action {next_action} failed in the domain.")
    domain.init()  # Reset the domain state
    return sequence

def generate_trace_sequence_blockworld(domain: blockWorld, seq_length: int, prob=False, pred_start=30, action_start=4):
    """Generate trace sequence for blockworld domain with offset indices"""
    sequence = []
    for _ in range(seq_length):  # Generate sequence of actions
        next_action = domain.choose_action()
        # Get tokens BEFORE executing the action
        tokens_before = domain.get_tokens_from(from_pred=pred_start, from_action=action_start, action=next_action)
        tokens_before = np.array(tokens_before)
        
        if prob:
            success = domain.prob_next_trace(next_action)
        else:
            success = domain.next_trace(next_action)
        
        if success:
            # Get tokens AFTER executing the action
            tokens_after = domain.get_tokens_from(from_pred=pred_start, from_action=action_start, action=next_action)
            tokens_after = np.array(tokens_after)
            
            sequence.append(tokens_before)  # State before action
            sequence.append(tokens_after)   # State after action
        else:
            raise ValueError(f"Action {next_action} failed in the domain.")
    domain.init()  # Reset the domain state
    return sequence

def test_logistics_model(domain, model, trace_length=8, prob=True):
    """Test the model on logistics domain"""
    print("=" * 80)
    print("LOGISTICS DOMAIN TEST")
    print("=" * 80)
    
    # Generate a fresh trace for testing
    test_trace = generate_trace_sequence(domain, seq_length=trace_length, prob=prob)
    
    # Get domain-specific token mapping
    token_map = domain.token_map
    
    print(f"Domain: {domain.name}")
    print(f"Token map: {token_map}")
    print(f"Testing {trace_length} actions...\n")
    
    correct_predictions = 0
    total_predictions = 0
    
    for step in range(0, min(len(test_trace) - 1, trace_length * 2), 2):
        current_state = test_trace[step]
        expected_next_state = test_trace[step + 1]
        
        # Get model prediction
        prediction = model.test(trace=np.array(current_state))
        predicted_state = prediction[0]
        
        # Extract action from expected state
        action_idx = np.where(expected_next_state[0] == 1)[0][0]
        action_name = list(domain.action_map.keys())[action_idx]
        
        print(f"Action {step//2 + 1}: {action_name}")
        print("-" * 40)
        
        # For logistics domain, analyze relevant predicates
        if hasattr(domain, 'token_map'):
            for pred_name, base_idx in token_map.items():
                print(f"\n{pred_name.upper()} predicate (index {base_idx}):")
                
                # Look at tokens that use this predicate
                for token_idx in range(1, len(current_state)):
                    token_current = current_state[token_idx]
                    token_expected = expected_next_state[token_idx]
                    token_predicted = predicted_state[token_idx]
                    
                    # Check if this token uses the current predicate
                    if base_idx < len(token_current) and token_current[base_idx] == 1:
                        relevant_slice = slice(base_idx, base_idx + 3)
                        
                        current_vals = token_current[relevant_slice]
                        expected_vals = token_expected[relevant_slice]
                        predicted_vals = np.round(token_predicted[relevant_slice], 2)
                        
                        print(f"  Token {token_idx}:")
                        print(f"    Current:   {current_vals}")
                        print(f"    Expected:  {expected_vals}")
                        print(f"    Predicted: {predicted_vals}")
                        
                        # Check if prediction matches expectation
                        match = np.allclose(expected_vals[1:], predicted_vals[1:], atol=0.3)
                        status = "✓" if match else "✗"
                        print(f"    Match: {status}")
                        
                        total_predictions += 1
                        if match:
                            correct_predictions += 1
        
        print("\n" + "=" * 60 + "\n")
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Logistics domain accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})")
    return accuracy

def test_blockworld_model(domain, model, logistics_domain, trace_length=8, prob=True):
    """Test the model on blockworld domain"""
    print("=" * 80)
    print("BLOCKWORLD DOMAIN TEST")
    print("=" * 80)
    
    # Calculate action start index based on logistics domain actions
    logistics_action_count = len(logistics_domain.action_map)
    action_start_idx = logistics_action_count
    
    # Generate a fresh trace for testing
    test_trace = generate_trace_sequence_blockworld(domain, seq_length=trace_length, prob=prob,
                                                  pred_start=30, action_start=action_start_idx)
    
    print(f"Domain: {domain.name}")
    print(f"Predicates start at index: 30")
    print(f"Actions start at index: {action_start_idx}")
    print(f"Token map: {domain.token_map}")
    print(f"Testing {trace_length} actions...\n")
    
    correct_predictions = 0
    total_predictions = 0
    
    for step in range(0, min(len(test_trace) - 1, trace_length * 2), 2):
        current_state = test_trace[step]
        expected_next_state = test_trace[step + 1]
        
        # Get model prediction
        prediction = model.test(trace=np.array(current_state))
        predicted_state = prediction[0]
        
        # Extract action from expected state (accounting for action offset)
        action_indices = np.where(expected_next_state[0] == 1)[0]
        if len(action_indices) > 0:
            action_idx = action_indices[0] - action_start_idx  # Adjust for offset
            if action_idx >= 0 and action_idx < len(domain.action_map):
                action_name = list(domain.action_map.keys())[action_idx]
            else:
                action_name = f"Unknown action (idx: {action_indices[0]})"
        else:
            action_name = "No action detected"
        
        print(f"Action {step//2 + 1}: {action_name}")
        print("-" * 40)
        
        # Analyze blockworld predicates using the domain's token map
        # Look for tokens that have predicates active (predicate value = 1)
        for token_idx in range(1, len(current_state)):
            token_current = current_state[token_idx]
            token_expected = expected_next_state[token_idx]
            token_predicted = predicted_state[token_idx]
            
            # Check if any blockworld predicates are active in this token
            predicate_found = False
            for pred_name, base_idx in domain.token_map.items():
                # Adjust index for the 30-index offset (predicates start at 30)
                adjusted_idx = base_idx + 30
                
                if adjusted_idx < len(token_current) and token_current[adjusted_idx] == 1:
                    predicate_found = True
                    relevant_slice = slice(adjusted_idx, adjusted_idx + 3)
                    
                    current_vals = token_current[relevant_slice]
                    expected_vals = token_expected[relevant_slice]
                    predicted_vals = np.round(token_predicted[relevant_slice], 2)
                    
                    print(f"  Token {token_idx} - {pred_name.upper()} predicate (index {adjusted_idx}):")
                    print(f"    Current:   {current_vals}")
                    print(f"    Expected:  {expected_vals}")
                    print(f"    Predicted: {predicted_vals}")
                    
                    # Check if prediction matches expectation
                    match = np.allclose(expected_vals[1:], predicted_vals[1:], atol=0.3)
                    status = "✓" if match else "✗"
                    print(f"    Match: {status}")
                    
                    total_predictions += 1
                    if match:
                        correct_predictions += 1
            
            # If no predicates found in this token, check if it's a global predicate
            if not predicate_found and token_idx < 10:  # Check first few tokens for global predicates
                # Check global predicates like hand_free
                for pred_name in domain.global_predicates:
                    if pred_name in domain.token_map:
                        base_idx = domain.token_map[pred_name] + 30
                        if base_idx < len(token_current) and token_current[base_idx] == 1:
                            relevant_slice = slice(base_idx, base_idx + 3)
                            
                            current_vals = token_current[relevant_slice]
                            expected_vals = token_expected[relevant_slice]
                            predicted_vals = np.round(token_predicted[relevant_slice], 2)
                            
                            print(f"  Token {token_idx} - {pred_name.upper()} global predicate (index {base_idx}):")
                            print(f"    Current:   {current_vals}")
                            print(f"    Expected:  {expected_vals}")
                            print(f"    Predicted: {predicted_vals}")
                            
                            match = np.allclose(expected_vals[1:], predicted_vals[1:], atol=0.3)
                            status = "✓" if match else "✗"
                            print(f"    Match: {status}")
                            
                            total_predictions += 1
                            if match:
                                correct_predictions += 1
        
        print("\n" + "=" * 60 + "\n")
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Blockworld domain accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})")
    return accuracy

def main():
    """Main training and testing flow"""
    print("=" * 80)
    print("SEQUENTIAL DOMAIN TRAINING PIPELINE")
    print("=" * 80)
    
    # Initialize domains
    logistics_domain = LogisticsDomain60()
    blockworld_domain = blockWorld()
    
    prob = True
    train_mode = True
    
    # Step 1: Train on Logistics Domain
    print("\nStep 1: Training on Logistics Domain")
    print("-" * 40)
    model = None
    if train_mode:
        model = train_logistics_model(prob=prob, model=model, domain=logistics_domain)
        torch.save(model.state_dict(), "./Parameters/sequential_combined_60_tokens.pth")
    else:
        model = att_only.Att_PAM(output_dim=logistics_domain.token_size, embed_dim=256, 
                               input_dim=logistics_domain.token_size, head_number=8)
        model.load_state_dict(torch.load("./Parameters/sequential_combined_60_tokens.pth"))
    
    logistics_accuracy = test_logistics_model(logistics_domain, model, trace_length=6, prob=prob)

    # Step 2: Train on Blockworld Domain (using the same model)
    print("\nStep 2: Training on Blockworld Domain")
    print("-" * 40)
    if train_mode:
        model = train_blockworld_model(prob=prob, model=model, domain=blockworld_domain, 
                                     logistics_domain=logistics_domain)
        torch.save(model.state_dict(), "./Parameters/sequential_combined_60_tokens.pth")
    else:
        model.load_state_dict(torch.load("./Parameters/sequential_combined_60_tokens.pth"))
    
    # Step 3: Test model correctness on both domains
    print("\nStep 3: Testing Model Correctness")
    print("-" * 40)
    
    # Test on logistics domain
    
    # Test on blockworld domain
    blockworld_accuracy = test_blockworld_model(blockworld_domain, model, logistics_domain, 
                                               trace_length=6, prob=prob)
    
    # Summary
    print("\n" + "=" * 80)
    print("TRAINING AND TESTING SUMMARY")
    print("=" * 80)
    print(f"Logistics Domain Accuracy: {logistics_accuracy:.2%}")
    print(f"Blockworld Domain Accuracy: {blockworld_accuracy:.2%}")
    print(f"Overall Performance: {(logistics_accuracy + blockworld_accuracy) / 2:.2%}")
    
    # Save results to file
    with open("sequential_training_results.txt", "w") as f:
        f.write("Sequential Domain Training Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Logistics Domain Accuracy: {logistics_accuracy:.2%}\n")
        f.write(f"Blockworld Domain Accuracy: {blockworld_accuracy:.2%}\n")
        f.write(f"Overall Performance: {(logistics_accuracy + blockworld_accuracy) / 2:.2%}\n")
        f.write(f"\nModel Configuration:\n")
        f.write(f"- Token size: 60\n")
        f.write(f"- Action buffer: 10\n")
        f.write(f"- Blockworld predicates start: index 30\n")
        f.write(f"- Blockworld actions start: index {len(logistics_domain.action_map)}\n")
    
    print("Results saved to 'sequential_training_results.txt'")

if __name__ == "__main__":
    main()
