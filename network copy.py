from torch import nn
from torch import optim
import torch

class Att_PAM(nn.Module):

    def __init__(self, output_dim, embed_dim, hidden_dim=400, latent_dim=200,input_dim = None):
        super(Att_PAM, self).__init__()
        self.embed_dim = embed_dim
        self.embedding_in = nn.Embedding(input_dim, embed_dim)
        self.embedding_out = nn.Embedding(output_dim, embed_dim)

        self.att1 = nn.MultiheadAttention(embed_dim, num_heads=10, dropout=0.1 , batch_first=True)
        self.att2 = nn.MultiheadAttention(embed_dim, num_heads=10, dropout=0.1 , batch_first=True)
        self.att3 = nn.MultiheadAttention(embed_dim, num_heads=10, dropout=0.1 , batch_first=True)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.ff1 = nn.Linear(embed_dim, embed_dim)
        self.ff2 = nn.Linear(embed_dim, embed_dim)
        self.ff3 = nn.Linear(embed_dim, embed_dim)
        self.flat = nn.Linear(embed_dim*embed_dim,embed_dim)

        self.create_vae(output_dim,hidden_dim, latent_dim)

    def create_vae(self,output_dim,hidden_dim, latent_dim):
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
            )
        
        # latent mean and variance 
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
            )
     
    def attention(self, x):
        x, _ = self.att1(x, x, x)
        x = self.norm1(x)
        x = self.ff1(x)
        x = nn.ReLU()(x)
        x, _ = self.att2(x, x, x)
        x = self.norm2(x)
        x = self.ff2(x)
        x = nn.ReLU()(x)
        x, _ = self.att3(x, x, x)
        x = self.norm3(x)
        x = self.ff3(x)
        x = nn.ReLU()(x)
        p = x.shape
        return x

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to("cpu")      
        z = mean + var*epsilon
        return z

    def decode(self, x):
        return self.decoder(x)


    def forward(self,x ,test=False, batch = 100):
        x = self.attention(x)
        # if  test:
        #     x = torch.flatten(x, start_dim=0)
        # else:
        #     x = torch.flatten(x,start_dim=1)
        # x = self.flat(x)

        # s = x.shape
        # mean, logvar = self.encode(x)

        # z = self.reparameterization(mean, logvar)
        # x_hat = self.decode(z)
        return x
    

    def klvLoss(self,x,y):
        mean, logvar = self.encode(x)
        return torch.mean(-0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp(), dim = 1), dim = 0)

    def changed_prop(self, o, y, x):
        yy = y.detach().cpu().numpy()
        xx = x.detach().cpu().numpy()
        oo = o.detach().cpu().numpy()
        s = xx.shape

        sums = 0
        for i in range(len(xx)):
            for j in range(len(oo[0])):
                if xx[i][j] != yy[i][j]:
                    sums += np.absolute(yy[i][j] - oo[i][j])
        return sums / len(xx)
    
    def train(self, All_traces ,batch , lr=0.001):
        vae = self
        criterion = nn.MSELoss()
        optimizer = optim.Adam(vae.parameters(), lr=0.001)
        iter = 4
        for j in range(iter):
            for epoch, traces in enumerate(All_traces):
                tr_traces = np.transpose(traces, (1, 0, 2, 3))
                inital_states = tr_traces[0]
                inital_states = torch.from_numpy(inital_states).float()
                running_loss = 0.0
                for i, ground_action in enumerate(tr_traces, 0):
                    if i %2 == 1:
                        inital_states = torch.from_numpy(ground_action).float()
                        continue
                    ground_action = torch.from_numpy(ground_action).float()
                    optimizer.zero_grad()

                    outputs,_,_ = vae(inital_states)
                    ground_action_flat = torch.reshape(ground_action,(batch,-1))
                    inital_states = torch.reshape(inital_states,(batch,-1))
                    ground_action_flat1 = torch.tensor([[ground_action[j][i][i] for i in range(100)] for j in range(100)])
                    #ground_action = np.delete(ground_action, [end-3, end-2, end-1], axis=1)
                    loss = criterion(outputs, ground_action_flat1) + vae.klvLoss(outputs, ground_action_flat) #0.1*vae.changed_prop(outputs, ground_action_flat, inital_states)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    inital_states = ground_action
                    
                print(f"Epoch {epoch + 1}, Loss: {running_loss / len(All_traces)}")
            print(f"finished {j+1}/{iter}.")
        print("Training complete.")

