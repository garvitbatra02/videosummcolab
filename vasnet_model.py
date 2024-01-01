import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

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

class SwinDeformableAttention(nn.Module):
    def __init__(self, dim, offset_dim):
        super(SwinDeformableAttention, self).__init__()
        self.dim = dim
        self.offset_dim = offset_dim
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)
        self.conv_offset = nn.Conv1d(1, 2 * offset_dim, kernel_size=3, stride=1, padding=1)
        self.scale = dim ** -0.5
        self.attn_drop = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        B, C, H, W = x.size()
        Q = self.proj_q(x).reshape(B, -1, H * W).transpose(1, 2)
        offset = self.conv_offset(Q)
        x_sampled = F.grid_sample(
            input=x,
            grid=offset[..., (1, 0)].reshape(B, -1, H, W),
            mode='bilinear',
            align_corners=True
        )
        x_sampled = x_sampled.reshape(B, C, -1).transpose(1, 2)
        attn = torch.einsum('b m c, b n c -> b m n', Q * self.scale, self.proj_k(x_sampled))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = torch.einsum('b m n, b n c -> b m c', attn, self.proj_v(x_sampled))
        x = self.proj_out(x)
        return x, offset

class DeformableAttention(nn.Module):
    def __init__(self, dim, offset_dim):
        super(DeformableAttention, self).__init__()
        self.dim = dim
        self.offset_dim = offset_dim
        self.conv_offset = nn.Conv1d(1, 2 * offset_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x, mask=None):
        B, C, H, W = x.size()
        Q = self.conv_offset(x).transpose(1, 2)
        x_sampled = F.grid_sample(
            input=x,
            grid=Q[..., (1, 0)].reshape(B, -1, H, W),
            mode='bilinear',
            align_corners=True
        )
        x_sampled = x_sampled.reshape(B, C, -1).transpose(1, 2)
        offset = Q
        attn = torch.einsum('b m c, b n c -> b m n', Q, Q)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = torch.einsum('b m n, b n c -> b m c', attn, x_sampled)
        return x, offset

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
        Q = self.Q(x) * 0.06
        offset = self.D(Q)
        x_sampled = F.grid_sample(
            input=x,
            grid=offset[..., (1, 0)],
            mode='bilinear',
            align_corners=True
        )
        logits = torch.matmul(Q, self.K(x_sampled).transpose(1, 0))

        if self.ignore_itself:
            logits[torch.eye(n).byte()] = float('-inf')

        if self.apperture > 0:
            onesmask = torch.ones(n, n)
            trimask = torch.tril(onesmask, -self.apperture) + torch.triu(onesmask, self.apperture)
            logits[trimask == 1] = float('-inf')

        att_weights_ = F.softmax(logits, dim=-1)
        weights = self.drop50(att_weights_)
        y = torch.matmul(self.V(x_sampled).transpose(1, 0), weights).transpose(1, 0)
        y = self.output_linear(y)
        return y, att_weights_, offset

# Usage example
if __name__ == "__main__":
    dim = 1024
    offset_dim = 1

    swin_deformable_attention = SwinDeformableAttention(dim, offset_dim)
    deformable_attention = DeformableAttention(dim, offset_dim)

    input_tensor = torch.rand((2, dim, 16, 16))

    # Swin Deformable Attention
    swin_output, swin_offset = swin_deformable_attention(input_tensor)

    # Your existing Deformable Attention
    your_output, your_offset = deformable_attention(input_tensor)

    print("Swin Deformable Attention Output:", swin_output.shape)
    print("Your Deformable Attention Output:", your_output.shape)
