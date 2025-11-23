import torch
from torch import nn
from Models.attention import Attention

class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.hidden_size = config['hidden_size']

        ff_hidden_dim = 4 * self.hidden_size

        self.attn_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1E-6)

        self.attn_block = Attention(config)

        self.ff_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1E-6)

        self.mlp_block = nn.Sequential(
            nn.Linear(self.hidden_size, ff_hidden_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ff_hidden_dim, self.hidden_size)
        )


        self.adaptive_norm_mlp = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(self.hidden_size, 6 * self.hidden_size, bias=True)
        )

        nn.init.xavier_uniform_(self.mlp_block[0].weight)
        nn.init.constant_(self.mlp_block[0].bias, 0)
        nn.init.xavier_uniform_(self.mlp_block[-1].weight)
        nn.init.constant_(self.mlp_block[-1].bias, 0)

        nn.init.constant_(self.adaptive_norm_mlp[-1].bias, 0)
        nn.init.constant_(self.adaptive_norm_mlp[-1].weight, 0)

    def forward(self, x, condition):
        scale_shift_params = self.adaptive_norm_mlp(condition).chunk(6, dim=1)
        (pre_attn_shift, pre_attn_scale, post_attn_scale,
        pre_mlp_shift, pre_mlp_scale, post_mlp_scale) = scale_shift_params
        
        out = x 
        attn_norm_output = (self.attn_norm(out) * (1 + pre_attn_scale.unsqueeze(1)) + pre_attn_shift.unsqueeze(1))
        out = out + post_attn_scale.unsqueeze(1) * self.attn_block(attn_norm_output)

        mlp_norm_output = (self.ff_norm(out) * (1 + pre_mlp_scale.unsqueeze(1)) + pre_mlp_shift.unsqueeze(1))

        out = out + post_mlp_scale.unsqueeze(1) * self.mlp_block(mlp_norm_output)
        return out




