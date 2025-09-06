from Domains.domain_generator import blockWorld
import torch
from pprint import pprint
from Domains.logistics_domain import LogisticsDomain 
import numpy as np
import Model.att_only as att_only



# Create a modified domain class with 60 token size and 60 tokens
class LogisticsDomain60(LogisticsDomain):
    def __init__(self):
        super().__init__()
        self.token_size = 60  # Token size is 60
        self.num_of_tokens = 60  # Number of tokens is 60

def train_model(prob=False, model=None, domain=None):
    batch_size = 20
    seq_length = 20
    iterations = 15
    traces = []
    for _ in range(100):  # Generate 100 traces
        trace = generate_trace_sequence(domain, seq_length=seq_length, prob=prob)
        traces.append(trace)
    traces = np.array(traces)
    traces = traces.reshape(traces.shape[0] // batch_size, batch_size, *traces.shape[1:])  # Reshape to match expected input dimensions
    model = att_only.Att_PAM(output_dim=domain.token_size, embed_dim=256, input_dim=domain.token_size, head_number=8) if model is None else model
    model.train(traces, lr=0.0005, iterations=iterations)
    return model

def train_sequential(prob=False, model=None, domain=None):
    batch_size = 20
    seq_length = 20
    iterations = 15
    traces = []
    for _ in range(100):  # Generate 100 traces
        trace = generate_trace_sequence_suffix(blocksDom, seq_length=8, prob=prob)
        traces.append(trace)
    traces = np.array(traces)
    traces = traces.reshape(traces.shape[0] // batch_size, batch_size, *traces.shape[1:])  # Reshape to match expected input dimensions
    model.train(traces, lr=0.0005, iterations=iterations)
    return model

def generate_trace_sequence(domain, seq_length: int, prob=False):
    sequence = []
    for _ in range(seq_length):  # Generate 20 actions
        next_action = domain.choose_action()
        tokens = domain.get_tokens(next_action)
        tokens = np.array(tokens)
        if prob:
            success = domain.prob_next_trace(next_action)
        else:
            success = domain.next_trace(next_action)
        if success:
            sequence.append(tokens)
            sequence.append(np.array(domain.get_tokens(next_action)))
        else:
            raise ValueError(f"Action {next_action} failed in the domain.")
    domain.init()  # Reset the domain state
    return sequence


def generate_trace_sequence_suffix(domain: blockWorld, seq_length: int, prob=False ):
    sequence = []
    for _ in range(seq_length):  # Generate 20 actions
        next_action = domain.choose_action()
        tokens = domain.get_tokens_from(from_pred=30, from_action=4, action=next_action)
        tokens = np.array(tokens)
        if prob:
            success = domain.prob_next_trace(next_action)
        else:
            success = domain.next_trace(next_action)
        if success:
            sequence.append(tokens)
            sequence.append(np.array(domain.get_tokens_from(from_pred=30, from_action=4, action=next_action)))
        else:
            raise ValueError(f"Action {next_action} failed in the domain.")
    domain.init()  # Reset the domain state
    return sequence

def test_attention_model(domain, model, trace_length=8, prob=True):
    """
    Test the attention model by comparing current state, expected state, and predicted state.
    Prints only the relevant indices from each token.
    """
    print("=" * 80)
    print("ATTENTION MODEL STATE COMPARISON TEST")
    print("=" * 80)
    
    # Generate a fresh trace for testing
    test_trace = generate_trace_sequence(domain, seq_length=trace_length, prob=prob)
    
    # Get domain-specific token mapping
    token_map = domain.token_map
    
    print(f"Domain: {domain.name}")
    print(f"Token map: {token_map}")
    print(f"Testing {trace_length} actions...\n")
    
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
        
        print("\n" + "=" * 60 + "\n")
    
    print("Test completed.")



domain = LogisticsDomain60()
batch_size = 10
prob = True
train = True
# Skip loading pre-trained model due to dimension mismatch (30 vs 60 tokens)
if train: 
    model = train_model(prob=prob, model=model, domain=domain)
    torch.save(model.state_dict(), "./Parameters/logistics_domain_60_tokens.pth")
else:
    model = att_only.Att_PAM(output_dim=domain.token_size, embed_dim=256, input_dim=domain.token_size, head_number=8)
    model.load_state_dict(torch.load("./Parameters/logistics_domain_60_tokens.pth"))

trace = generate_trace_sequence(domain, seq_length=8, prob=prob)
prediction = model.test(trace=np.array(trace[0]))
f = open("output_60_tokens.txt", "w")

test_attention_model(domain, model, trace_length=8, prob=prob)

blocksDom = blockWorld()

# train_sequential(prob=prob, model=model, domain=blocksDom)

# trace = generate_trace_sequence_suffix(blocksDom, seq_length=8, prob=prob)
# for m in range(0,10,2):
#     next_state = trace[m+1]
#     prediction = model.test(trace=np.array(trace[m]))

#     together = [(trace[m][i], next_state[i], np.round(prediction[0][i], 1)) for i in range(len(prediction[0]))]
#     action = list(domain.action_map.keys())[list(next_state[0]).index(1)]
#     f.write(f"Action: {action}\n\n")
#     print(f"Action: {action}\n\n")
#     if domain.name == "blockworld":
#         prefixes = ["on", "clear", "in hand","on", "clear", "in hand", "hand_free"]
#         last_pred = 7
#     else:
#         prefixes = ["on peg", "clear", "smaller","clear from", "clear to"]
#         last_pred = 5
#     for i in range(1,len(together)):
            
#             prefix = prefixes[i-1]
#             prefix += f" {int((i-1)/3)}"
#             if i>3:
#                 m = i%3 if i%3> 0 else 3
#                 j = (m-1)*3 +30
#             else:
#                 j = (i-1)*3 + 30  # Offset by 30 for the larger token size

#             f.write(f"{j} {prefix} -  was: {together[i][0][j:j+3]}\n     expected: {together[i][1][j:j+3]}\n     predicted: {together[i][2][j:j+3]}\n\n")
#             print(f"{j} {prefix} -  was: {together[i][0][j:j+3]}\n     expected: {together[i][1][j:j+3]}\n     predicted: {together[i][2][j:j+3]}\n")
#         #pprint(f"s0: {together[i][0]}\n s1: {together[i][1]}\n P: {together[i][2]}\n")