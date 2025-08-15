from domain_generator import blockWorld
import att_only
domain = blockWorld()


def generate_trace_sequence(domain: blockWorld, seq_length: int):
    sequence = []
    for _ in range(5):  # Generate 5 actions
        next_action = domain.choose_action()
        tokens = domain.get_tokens(next_action)
        success = domain.next_trace(next_action)
        if success:
            sequence.extend(tokens)
        else:
            raise ValueError(f"Action {next_action} failed in the domain.")
    domain.init()  # Reset the domain state
    return sequence


action = ("stack", ("block",0),("block",1))
trace = generate_trace_sequence(domain, seq_length=5)

model = att_only.Att_PAM(output_dim=domain.output_dim, embed_dim=128, input_dim=domain.token_size)
x = model(trace)

