import domain_generator
import att_only
domain = domain_generator.blockWorld()
action = ("stack", ("block",0),("block",1))
action = domain.next_trace()
tokens = domain.get_tokens(action)


model = att_only.Att_PAM(output_dim=domain.output_dim, embed_dim=128, input_dim=domain.token_size)
x = model(tokens)