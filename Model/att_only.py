from torch import nn
from torch import optim
import numpy as np
import torch

class Att_PAM(nn.Module):

    def __init__(self, output_dim, embed_dim,input_dim, head_number):
        super(Att_PAM, self).__init__()
        self.embed_dim = embed_dim
        self.embedding_in1 = nn.Linear(input_dim, int(embed_dim/2))
        self.embedding_in2 = nn.Linear(int(embed_dim/2), embed_dim)

        self.att1 = nn.MultiheadAttention(embed_dim, num_heads=head_number, dropout=0.1 , batch_first=True)
        self.att2 = nn.MultiheadAttention(embed_dim, num_heads=head_number, dropout=0.1 , batch_first=True)
        self.att3 = nn.MultiheadAttention(embed_dim, num_heads=head_number, dropout=0.1 , batch_first=True)
        self.att4 = nn.MultiheadAttention(embed_dim, num_heads=head_number, dropout=0.1 , batch_first=True)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)

        self.ff1 = nn.Linear(embed_dim, embed_dim)
        self.ff2 = nn.Linear(embed_dim, embed_dim)
        self.ff3 = nn.Linear(embed_dim, embed_dim)
        self.ff4 = nn.Linear(embed_dim, embed_dim)

        self.embedding_out1 = nn.Linear(embed_dim, int(embed_dim/2))
        self.embedding_out2 = nn.Linear(int(embed_dim/2), output_dim)


    def forward(self,x):
        x = self.embedding_in1(x)
        x = self.embedding_in2(x)
        x1, _ = self.att1(x, x, x)
        x = nn.Dropout(0.1)(x)
        x = self.norm1(x1 + x)
        f1 = self.ff1(x)
        f1 = nn.GELU()(f1)
        x = nn.LayerNorm(self.embed_dim)(f1 + x)


        x2, _ = self.att2(x, x, x)
        x = nn.Dropout(0.1)(x)
        x = self.norm2(x2 + x)
        f2 = self.ff2(x)
        x = nn.GELU()(f2)
        x = nn.LayerNorm(self.embed_dim)(f2 + x)

        x3, _ = self.att3(x, x, x)
        x = nn.Dropout(0.1)(x)
        x = self.norm3(x3 + x)
        f3 = self.ff3(x)
        f3 = nn.GELU()(f3)
        x = nn.LayerNorm(self.embed_dim)(f3 + x)

        x3, _ = self.att3(x, x, x)
        x = nn.Dropout(0.1)(x)
        x = self.norm3(x3 + x)
        f3 = self.ff3(x)
        f3 = nn.GELU()(f3)
        x = nn.LayerNorm(self.embed_dim)(f3 + x)

        x4, _ = self.att4(x, x, x)
        x = nn.Dropout(0.1)(x)
        x = self.norm4(x4 + x)
        f4 = self.ff4(x)
        f4 = nn.GELU()(f3)
        x = nn.LayerNorm(self.embed_dim)(f4 + x)

        x = self.embedding_out1(x)
        x = nn.GELU()(x)
        x = self.embedding_out2(x)
        x = nn.GELU()(x)
        return x
    
    def train(self, All_traces , lr=0.00001, iterations = 10):
        criterion = nn.MSELoss()
    
        optimizer = optim.Adam(self.parameters(), lr)
        for j in range(iterations):
            for epoch, traces in enumerate(All_traces):
                tr_traces = np.transpose(traces, (1, 0, 2, 3))
                running_loss = 0.0
                for i, next_state in enumerate(tr_traces):
                    if i%2 == 0:
                        inital_state = torch.from_numpy(next_state).float()
                        continue
                    next_state = torch.from_numpy(next_state).float()
                    optimizer.zero_grad()
                    outputs = self(inital_state)
                    loss = criterion(outputs, next_state)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    
                print(f"Epoch {epoch + 1}, Loss: {running_loss / len(All_traces)}")
            print(f"finished {j+1}/{iterations}.")
        print("Training complete.")

    def test(self, trace):
        with torch.no_grad():
            trace = torch.from_numpy(np.array([trace])).float()
            prediction = self(trace)
            return prediction.numpy()