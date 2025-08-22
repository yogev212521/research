import sys
import torch
from pprint import pprint
from domain_generator import blockWorld
import numpy as np
import att_only

def train_model():
    batch_size = 20
    seq_length = 20
    iterations = 5
    traces = []
    for _ in range(100):  # Generate 100 traces
        trace = generate_trace_sequence(domain, seq_length=seq_length)
        traces.append(trace)
    traces = np.array(traces)
    traces = traces.reshape(traces.shape[0] // batch_size, batch_size, *traces.shape[1:])  # Reshape to match expected input dimensions
    model = att_only.Att_PAM(output_dim=domain.token_size, embed_dim=256, input_dim=domain.token_size, head_number=8)
    model.train(traces, lr=0.001, iterations=iterations)
    return model

def generate_trace_sequence(domain: blockWorld, seq_length: int):
    sequence = []
    for _ in range(seq_length):  # Generate 20 actions
        next_action = domain.choose_action()
        tokens = domain.get_tokens(next_action)
        tokens = np.array(tokens)
        success = domain.next_trace(next_action)
        if success:
            sequence.append(tokens)
            sequence.append(np.array(domain.get_tokens(next_action)))
        else:
            raise ValueError(f"Action {next_action} failed in the domain.")
    domain.init()  # Reset the domain state
    return sequence

domain = blockWorld()
batch_size = 10
# model = att_only.Att_PAM(output_dim=domain.token_size, embed_dim=256, input_dim=domain.token_size, head_number=8)
# model.load_state_dict(torch.load("trained_model_Gelu.pth"))
model = train_model()
torch.save(model.state_dict(), "trained_model_Gelu.pth")

trace = generate_trace_sequence(domain, seq_length=8)
prediction = model.test(trace=np.array(trace[0]))
f = open("output.txt", "w")

for m in range(0,10,2):
    next_state = trace[m+1]
    prediction = model.test(trace=np.array(trace[m]))

    together = [(trace[m][i], next_state[i], np.round(prediction[0][i], 1)) for i in range(len(prediction[0]))]
    action = list(domain.action_map.keys())[list(next_state[0]).index(1)]
    f.write(f"Action: {action}\n\n")
    for i in range(1,7):
            
            prefix = ["on", "clear", "in hand"][(i-1)%3]
            prefix += f" {int((i-1)/3)}"
            if i>3:
                 m = i%3 if i%3> 0 else 3
                 j = (m-1)*3 + 4
                 if action in ["putdown", "pick"]:
                     continue
            else:
                 j = (i-1)*3 + 4

            f.write(f"{j} {prefix} -  was: {together[i][0][j:j+3]}\n     expected: {together[i][1][j:j+3]}\n     predicted: {together[i][2][j:j+3]}\n\n")
            print(f"{j} {prefix} -  was: {together[i][0][j:j+3]}\n         expected: {together[i][1][j:j+3]}\n      predicted: {together[i][2][j:j+3]}\n")
        #pprint(f"s0: {together[i][0]}\n s1: {together[i][1]}\n P: {together[i][2]}\n")
    print(action)

# Test the trained model