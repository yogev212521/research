import torch
import numpy as np
import Model.att_only as att_only
from tqdm import tqdm
from IndexManager import IndexManager
from pddlsim_runner import Domain_sim
import asyncio
import matplotlib.pyplot as plt

def devider(n):
    for i in range(3,n):
        if n%i == 0 and i != 1:
            return i 
    return 2
def test_attention_model(domain, model, trace_length=20, prob=True):
    """
    Test the attention model by comparing current state, expected state, and predicted state.
    Prints only the relevant indices from each token.
    """
    print("=" * 80)
    print("ATTENTION MODEL STATE COMPARISON TEST")
    print("=" * 80)
    try:
        # Generate a fresh trace for testing
        # If Generate_trace is async, use asyncio.run; otherwise, call directly
        if asyncio.iscoroutinefunction(domain.Generate_trace):
            test_trace = asyncio.run(domain.Generate_trace(trace_length))
        else:
            test_trace = domain.Generate_trace(trace_length)

        # Get domain-specific token mapping
        token_map = getattr(domain, "token_map", {})

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
            # Get action name from domain definition
            try:
                actions = list(map(lambda a: a.value, list(domain.domain.actions_section._items.keys())))
                action_offset = getattr(domain, "action_offset", 0)
                real_idx = action_idx - action_offset
                if 0 <= real_idx < len(actions):
                    action_name = actions[real_idx]
                else:
                    action_name = f"INVALID_IDX_{action_idx}"
            except Exception:
                action_name = str(action_idx)

            print(f"Action {step//2 + 1} (index {action_idx}, name: {action_name}):")
            print("-" * 40)

            # For logistics domain, analyze relevant predicates
            if hasattr(domain, 'token_map'):
                for pred_name, base_idx in token_map.items():
                    pred_str = str(pred_name)
                    print(f"\n{pred_str.upper()} predicate (index {base_idx}):")

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

        print("Test completed successfully.")
    except Exception as e:
        print("Test failed with exception:", e)
    finally:
        print("test_attention_model execution finished.")
    
def train_multiple_domain(domains:list, prob = False, model = None, iterations = None ):
    batch_size = 10
    default_iter = 5
    iterations = iterations if iterations is not None else default_iter
    traces = []
    for domain in domains:
        for _ in tqdm(range(40)):  # Generate 100 traces
            trace = asyncio.run(domain.Generate_trace(20))
            print(np.shape(trace))
            while np.shape(trace)[0] != 39:
                trace = asyncio.run(domain.Generate_trace(20))
                print(np.shape(trace))
            traces.append(trace)
    traces = np.array(traces)
    np.random.shuffle(traces)
    traces = traces.reshape(traces.shape[0] // batch_size, batch_size, *traces.shape[1:])  # Reshape to match expected input dimensions
    model = att_only.Att_PAM(output_dim=domain.token_size, embed_dim=512, input_dim=domain.token_size, head_number=8) if model is None else model
    model.train(traces, lr=0.0005, iterations=iterations)
    return model

def train_and_evaluate_domains(domains: list[Domain_sim], trace_length=20, prob=False, model=None, iterations=None, output_file="domain_evaluation_summary.txt"):
    default_iter = 10
    iterations = iterations if iterations is not None else default_iter
    # Prepare metric history per domain
    metric_history = {domain.name: {'identifier': [], 'truth': [], 'instance': [], 'MSE': []} for domain in domains}
    for m in range(1,8):
        print(f"\nstarting itertaion {m}\n")
        traces = []
        batch_size = devider(m)
        trace = None
        for domain in domains:
            for _ in range(m*2):
                while not trace or np.shape(trace)[0] != 19:
                    trace = asyncio.run(domain.Generate_trace(10))
                traces.append(trace)
        traces = np.array(traces)
        np.random.shuffle(traces)
        traces = traces.reshape(traces.shape[0] // batch_size, batch_size, *traces.shape[1:])
        model = att_only.Att_PAM(output_dim=domains[0].token_size, embed_dim=512, input_dim=domains[0].token_size, head_number=8)
        model.train(traces, lr=0.00001, iterations=iterations, delta= 0.0001/m)

        summary = []
        for domain in domains:
            if asyncio.iscoroutinefunction(domain.Generate_trace):
                test_trace = asyncio.run(domain.Generate_trace(trace_length))
            else:
                test_trace = domain.Generate_trace(trace_length)
            N = 0
            identifier, truth, instance, mse = 0,0,0,0
            for step in range(0, min(len(test_trace) - 1, trace_length * 2), 2):
                current_state = test_trace[step]
                expected_next_state = test_trace[step + 1]
                prediction = model.test(trace=np.array(current_state))[0]
                # Compute mean absolute error for each output index
                for i, token in enumerate(expected_next_state, 1):
                    for j in range(len(token)):
                        if token[j] == 1:
                            identifier += abs(prediction[i][j] - expected_next_state[i][j])
                            if domain.prob and domain.prob[j]:
                                truth += 0
                            else:
                                truth += abs(prediction[i][j+1] - expected_next_state[i][j+1])
                            instance += abs(prediction[i][j+2] - expected_next_state[i][j+2])
                            N +=1
                            mse += np.sqrt(instance**2 + truth**2 +identifier**2)
            # Save metrics for this iteration
            metric_history[domain.name]['identifier'].append(identifier/N if N else 0)
            metric_history[domain.name]['truth'].append(truth/N if N else 0)
            metric_history[domain.name]['instance'].append(instance/N if N else 0)
            metric_history[domain.name]['MSE'].append(mse/N if N else 0)
            summary.append((domain.name ,identifier/N,truth/N, instance/N, mse/N))

    with open(output_file, "w") as f:
        f.write(f"Training summary:\n")
        f.write(f"- Number of traces used for training: {len(traces)}\n")
        f.write(f"- Length of each trace: 20\n")
        f.write(f"- Number of training iterations: {iterations}\n\n")
        for domain_name, identifier, truth, instance , MSE in summary:
            f.write(f"Domain: {domain_name}, distences - identifier: {identifier}, truth: {truth}, instance: {instance}\n")

    import os
    # Plot mse and other metrics as separate subplots in the same figure for each domain
    os.makedirs("graphs", exist_ok=True)
    for domain_name, metrics in metric_history.items():
        m_range = [(10 + 5*m)*10 for m in range(len(metrics['identifier']))]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(m_range, metrics['identifier'], marker='o', label='identifier')
        ax1.plot(m_range, metrics['truth'], marker='o', label='truth')
        ax1.plot(m_range, metrics['instance'], marker='o', label='instance')
        ax1.set_ylabel("loss - MSE")
        ax1.set_xlabel("number of states")
        ax1.legend(loc='upper left')
        ax1.set_title(f"{domain_name} - Identifier, Truth, Instance")
        ax1.grid(True)
        from matplotlib.ticker import MaxNLocator
        ax1.yaxis.set_major_locator(MaxNLocator(nbins=15))

        # ax2.plot(m_range, metrics['mse'], marker='o', color='red', label='mse')
        # ax2.set_xlabel("number of states")
        # ax2.set_ylabel("loss - MSE")
        # ax2.legend(loc='upper left')
        # ax2.set_title(f"{domain_name} - MSE")
        # ax2.grid(True)
        # ax2.yaxis.set_major_locator(MaxNLocator(nbins=15))

        plt.tight_layout()
        plt.savefig(os.path.join("graphs", f"solo_{domain_name}_metrics_subplots.png"))
        plt.close(fig)

    print(f"Summary saved to {output_file} and subplot figures saved as graphs/<domain_name>_metrics_subplots.png")

def train_model(domain, prob = False, model = None, iterations = None ):
    batch_size = 10
    default_iter = 10
    iterations = iterations if iterations is not None else default_iter
    traces = []
    for _ in tqdm(range(40)):  # Generate 100 traces
        trace = asyncio.run(domain.Generate_trace(20))
        print(np.shape(trace))
        while np.shape(trace)[0] != 39:
            trace = asyncio.run(domain.Generate_trace(20))
        traces.append(trace)
    traces = np.array(traces)
    traces = traces.reshape(traces.shape[0] // batch_size, batch_size, *traces.shape[1:])  # Reshape to match expected input dimensions
    model = att_only.Att_PAM(output_dim=domain.token_size, embed_dim=512, input_dim=domain.token_size, head_number=8) if model is None else model
    model.train(traces, lr=0.0001, iterations=iterations)
    return model

indexMn = IndexManager(start_index=20)
logistics = Domain_sim(indexManager=indexMn,DOMAIN_FILE="./pddlDomains/logistics_domain.pddl",PROBLEM_FILE="./pddlDomains/logistics_problem.pddl")
blocksworld = Domain_sim(indexManager=indexMn ,DOMAIN_FILE="./pddlDomains/blockworld_domain.pddl",PROBLEM_FILE="./pddlDomains/blockworld_problem.pddl",pred_offset=logistics.pred_size, action_offset=4)
gripper = Domain_sim(indexManager=indexMn ,DOMAIN_FILE="./pddlDomains/gripper_domain.pddl",PROBLEM_FILE="./pddlDomains/gripper_problem.pddl", action_offset=8, pred_offset= logistics.pred_size + blocksworld.pred_size)
rooms = Domain_sim(indexManager=indexMn ,DOMAIN_FILE="./pddlDomains/rooms_domain.pddl",PROBLEM_FILE="./pddlDomains/rooms_problem.pddl", action_offset=11, pred_offset= logistics.pred_size + blocksworld.pred_size + gripper.pred_size)

if True:
    model = train_and_evaluate_domains([logistics])
    # torch.save(model.state_dict(), "./Parameters/test.pth")
else:
    model = att_only.Att_PAM(output_dim=logistics.token_size, embed_dim=512, input_dim=logistics.token_size, head_number=8)
    model.load_state_dict(torch.load("./Parameters/pddl_4_domains_trained.pth"))

# test_attention_model(domain=blocksworld, model=model, prob=False)

# test_attention_model(domain=logistics, model=model, prob=False)

# test_attention_model(domain=gripper, model=model, prob=False)

# test_attention_model(domain=rooms, model=model, prob=False)