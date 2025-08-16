from torch import nn
from torch import optim
import numpy as np
import torch

class Att_PAM(nn.Module):

    def __init__(self, output_dim, embed_dim,input_dim, head_number):
        super(Att_PAM, self).__init__()
        self.embedding_in = nn.Embedding(input_dim, embed_dim)

        self.att1 = nn.MultiheadAttention(embed_dim, num_heads=head_number, dropout=0.1 , batch_first=True)
        self.att2 = nn.MultiheadAttention(embed_dim, num_heads=head_number, dropout=0.1 , batch_first=True)
        self.att3 = nn.MultiheadAttention(embed_dim, num_heads=head_number, dropout=0.1 , batch_first=True)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.ff1 = nn.Linear(embed_dim, embed_dim)
        self.ff2 = nn.Linear(embed_dim, embed_dim)
        self.ff3 = nn.Linear(embed_dim, embed_dim)
        
        self.embedding_out = nn.Embedding(output_dim, embed_dim)
     
    def forward(self,x):
        x = self.embedding_in(x)

        x1, _ = self.att1(x, x, x)
        x = nn.Dropout(0.1)(x)
        x = self.norm1(x1 + x)
        f1 = self.ff1(x)
        f1 = nn.LeakyReLU()(f1)
        x = nn.LayerNorm(self.embed_dim)(f1 + x)


        x2, _ = self.att2(x, x, x)
        x = nn.Dropout(0.1)(x)
        x = self.norm2(x2 + x)
        f2 = self.ff2(x)
        x = nn.LeakyReLU()(f2)
        x = nn.LayerNorm(self.embed_dim)(f2 + x)

        x3, _ = self.att3(x, x, x)
        x = nn.Dropout(0.1)(x)
        x = self.norm3(x3 + x)
        f3 = self.ff3(x)
        f3 = nn.LeakyReLU()(f3)
        x = nn.LayerNorm(self.embed_dim)(f3 + x)

        x = self.embedding_out(x)
        return x
    
    def train(self, All_traces , lr=0.001, iterations = 10):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr)
        for j in range(iterations):
            for epoch, traces in enumerate(All_traces):
                tr_traces = np.transpose(traces, (1, 0, 2, 3))
                inital_states = tr_traces[0]
                inital_states = torch.from_numpy(inital_states).float()
                running_loss = 0.0
                for i, ground_action in enumerate(tr_traces, 0):
                    ground_action = torch.from_numpy(ground_action).float()
                    optimizer.zero_grad()
                    outputs = self(inital_states)
                    loss = criterion(outputs, ground_action)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    inital_states = ground_action
                    
                print(f"Epoch {epoch + 1}, Loss: {running_loss / len(All_traces)}")
            print(f"finished {j+1}/{iter}.")
        print("Training complete.")

