
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import  *
from layer_norm import  *



class SelfAttention(nn.Module):

    def __init__(self, apperture=-1, ignore_itself=False, input_size=1024, output_size=1024):
        super(SelfAttention, self).__init__()

        self.apperture = apperture
        self.ignore_itself = ignore_itself

        self.m = input_size
        self.output_size = output_size

        self.ofx = nn.Linear(in_features=self.m, out_features=self.output_size)
        self.ofy = nn.Linear(in_features=self.m, out_features=self.output_size)
        self.K = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.Q = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.V = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.output_linear = nn.Linear(in_features=self.output_size, out_features=self.m, bias=False)

        self.drop50 = nn.Dropout(0.5)



    def forward(self, x):
        
        H,W = x.shape[0],x.shape[1]
        n = x.shape[0]  # sequence length
        Q = self.Q(x)  # ENC (n x m) => (n x H) H= hidden size

        ofx = self.ofx(Q)
        ofy = self.ofy(Q)
        off = torch.stack((ofx, ofy), -1)
        
        pos = np.indices((H,W)).transpose(1,2,0)
        pos = torch.from_numpy(pos).to('cuda')
        pos = pos.to(torch.float)
        pos[:,:,0] = 2*pos[:,:,0]/H - 1
        pos[:,:,1] = 2*pos[:,:,1]/W - 1

        pos = (pos + off).clamp(-1., +1.)
        pos = torch.reshape(pos,(1,H,W,2))

        x_new = F.grid_sample(
            input = x.reshape(1,1,H,W),
            grid = pos,
            mode='bilinear', align_corners=True)
        
        x_new = x_new.reshape(H,W)
        
        K = self.K(x_new)
        V = self.V(x_new)
        
        Q *= 0.06
        logits = torch.matmul(Q, K.transpose(1,0))

        if self.ignore_itself:
            # Zero the diagonal activations (a distance of each frame with itself)
            logits[torch.eye(n).byte()] = -float("Inf")

        if self.apperture > 0:
            # Set attention to zero to frames further than +/- apperture from the current one
            onesmask = torch.ones(n, n)
            trimask = torch.tril(onesmask, -self.apperture) + torch.triu(onesmask, self.apperture)
            logits[trimask == 1] = -float("Inf")

        att_weights_ = nn.functional.softmax(logits, dim=-1)
        weights = self.drop50(att_weights_)
        y = torch.matmul(V.transpose(1,0), weights).transpose(1,0)
        y = self.output_linear(y)

        return y, att_weights_



class VASNet(nn.Module):

    def __init__(self):
        super(VASNet, self).__init__()

        self.m = 1024 # cnn features size
        self.hidden_size = 1024

        self.att = SelfAttention(input_size=self.m, output_size=self.m)
        self.ka = nn.Linear(in_features=self.m, out_features=1024)
        self.kb = nn.Linear(in_features=self.ka.out_features, out_features=1024)
        self.kc = nn.Linear(in_features=self.kb.out_features, out_features=1024)
        self.kd = nn.Linear(in_features=self.ka.out_features, out_features=1)

        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.drop50 = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=0)
        self.layer_norm_y = LayerNorm(self.m)
        self.layer_norm_ka = LayerNorm(self.ka.out_features)


    def forward(self, x, seq_len):

        m = x.shape[2] # Feature size

        # Place the video frames to the batch dimension to allow for batch arithm. operations.
        # Assumes input batch size = 1.
        x = x.view(-1, m)
        y, att_weights_ = self.att(x)

        y = y + x
        y = self.drop50(y)
        y = self.layer_norm_y(y)

        # Frame level importance score regression
        # Two layer NN
        y = self.ka(y)
        y = self.relu(y)
        y = self.drop50(y)
        y = self.layer_norm_ka(y)

        y = self.kd(y)
        y = self.sig(y)
        y = y.view(1, -1)

        return y, att_weights_



if __name__ == "__main__":
    pass
