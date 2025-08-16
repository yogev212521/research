from domain_generator import blockWorld
import numpy as np
import att_only


def generate_trace_sequence(domain: blockWorld, seq_length: int):
    sequence = []
    for _ in range(5):  # Generate 5 actions
        next_action = domain.choose_action()
        tokens = domain.get_tokens(next_action)
        tokens = np.array(tokens)
        t = tokens.shape
        success = domain.next_trace(next_action)
        if success:
            sequence.append(tokens)
        else:
            raise ValueError(f"Action {next_action} failed in the domain.")
    domain.init()  # Reset the domain state
    return sequence

domain = blockWorld()
batch_size = 5
traces = []
for _ in range(20):  # Generate 5 traces
    trace = generate_trace_sequence(domain, seq_length=5)
    traces.append(trace)
traces = np.array(traces)
traces = traces.reshape(traces.shape[0] // batch_size, batch_size, *traces.shape[1:])  # Reshape to match expected input dimensions
model = att_only.Att_PAM(output_dim=domain.token_size, embed_dim=128, input_dim=domain.token_size, head_number=8)
model.train(traces)
