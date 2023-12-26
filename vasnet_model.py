
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import  *

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class DeformableAttention(nn.Module):
    def __init__(self, dim, offset_dim):
        super(DeformableAttention, self).__init__()
        self.dim = dim
        self.offset_dim = offset_dim
        self.conv_offset = nn.Conv1d(1, 2 * offset_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x, mask=None):
        L = x.shape[0]  # Account for unbatched 1D input
        C = x.shape[1]
        # Project queries, keys, and values
        offset = self.conv_offset(Q)
        x_sampled = F.grid_sample(
                input=x
                grid=offset[..., (1, 0)], # y, x -> x, y
                mode='bilinear', align_corners=True) # B * g, Cg, Hg, Wg

        return x_sampled, offset


class DeformableSelfAttention(nn.Module):
   def __init__(self, apperture=-1, ignore_itself=False, input_size=1024, output_size=1024):
        super(DeformableSelfAttention, self).__init__()

        self.apperture = apperture
        self.ignore_itself = ignore_itself

        self.m = input_size
        self.output_size = output_size

        self.D = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.K = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.Q = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.V = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.output_linear = nn.Linear(in_features=self.output_size, out_features=self.m, bias=False)

        self.drop50 = nn.Dropout(0.5)



    def forward(self, x):
        n = x.shape[0]  # sequence length

        Q = self.Q(x)  # ENC (n x m) => (n x H) H= hidden size
        offset = self.D(Q)
        x_sampled = F.grid_sample(
                input=x.u
                grid=offset[..., (1, 0)], 
                mode='bilinear', align_corners=True)
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




class DeformableVASNet(nn.Module):
    def __init__(self):
        super(DeformableVASNet, self).__init__()

        self.m = 1024  # cnn features size
        self.hidden_size = 1024

        self.att = DeformableSelfAttention(input_size=self.m, output_size=self.m, offset_dim=1)
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
        m = x.shape[2]  # Feature size

        x = x.view(-1, m)
        y, att_weights_,offset = self.att(x)

        y = y + x
        y = self.drop50(y)
        y = self.layer_norm_y(y)

        y = self.ka(y)
        y = self.relu(y)
        y = self.drop50(y)
        y = self.layer_norm_ka(y)

        y = self.kd(y)
        y = self.sig(y)
        y = y.view(1, -1)

        return y, att_weights_,offset


if __name__ == "__main__":
    pass
