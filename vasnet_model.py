
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

        self.proj_qkv = nn.Linear(dim, 3 * dim)
        self.conv_offset = nn.Conv1d(1, 2 * offset_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x, mask=None):
        B = x.shape[0]

        # Adjusted the chunking along the last dimension
        qkv = self.proj_qkv(x.unsqueeze(1))
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Reshaped x for conv_offset
        x_offset = x.unsqueeze(1).unsqueeze(3)

        offset = self.conv_offset(x_offset).squeeze(3).permute(0, 2, 1)

        # Reshape offset to (B, 1, N, offset_dim)
        offset = offset.view(B, 1, -1, self.offset_dim)

        # Calculate normalized attention scores with offsets
        q, k, v = q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
        q, k, v = map(lambda t: t.view(B, 1, -1, -1), (q, k, v))

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dim ** 0.5)
        attn_scores += offset[:, :, :, :2]  # Incorporate offsets

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.masked_fill(mask == 0, 0.0) if mask is not None else attn_weights

        # Apply attention to values with offsets
        output = torch.matmul(attn_weights, v) + offset[:, :, :, 2:]  # Incorporate offsets in values

        # Reshape and concatenate to get the final output
        output = output.transpose(1, 2).contiguous().view(B, -1, v.size(-1))

        return output, attn_weights, offset



class DeformableSelfAttention(nn.Module):
    def __init__(self, input_size, output_size, offset_dim, attn_drop=0.0, proj_drop=0.0):
        super(DeformableSelfAttention, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.offset_dim = offset_dim
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.deformable_attn = DeformableAttention(output_size, offset_dim)
        self.proj = nn.Linear(output_size, output_size)

    def forward(self, x, mask=None):
        x_total = x.unsqueeze(1)
        attn_output, att_weights_, offset = self.deformable_attn(x_total, mask)
        attn_output = attn_output.squeeze(1)

        # Linear projection and dropout
        x = self.proj(attn_output)
        x = self.proj_drop(x)

        return x, att_weights_, offset



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
