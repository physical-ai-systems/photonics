import torch
from torch import nn
from einops import rearrange

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()


        self.n_heads = config['num_heads']
        self.hidden_size = config['hidden_size']
        self.head_dim = config['head_dim']

        self.attn_dim = self.n_heads * self.head_dim

        self.qkv_proj = nn.Linear(self.hidden_size, 3 * self.attn_dim, bias=True)
        self.output_proj = nn.Sequential(nn.Linear(self.attn_dim, self.hidden_size))

        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.constant_(self.qkv_proj.bias, 0)
        nn.init.xavier_uniform_(self.output_proj[0].weight)
        nn.init.constant_(self.output_proj[0].bias, 0)


    def forward(self, x):
        B, N = x.shape[:2]

        q, k, v = self.qkv_proj(x).split(self.attn_dim, dim=-1)

        q = rearrange(q, 'b n (n_h h_dim) -> b n_h n h_dim',
                      n_h=self.n_heads, h_dim=self.head_dim)
        k = rearrange(k, 'b n (n_h h_dim) -> b n_h n h_dim',
                      n_h=self.n_heads, h_dim=self.head_dim)
        v = rearrange(v, 'b n (n_h h_dim) -> b n_h n h_dim',
                      n_h=self.n_heads, h_dim=self.head_dim)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** (-0.5))
        attn = nn.functional.softmax(attn, dim=-1)


        out = torch.matmul(attn, v)

        out = rearrange(out, 'b n_h n h_dim -> b n (n_h h_dim)')

        out = self.output_proj(out)

        return out