import torch 
import torch.nn as nn
from Models.patch_embed import PatchEmbedding
from Models.transformer_block import TransformerBlock
from einops import rearrange

def get_time_embeddings(time_steps, temb_dim):


    assert temb_dim % 2 == 0

    factor = 10000 ** ((torch.arange(
        start=0,
        end=temb_dim // 2,
        dtype=torch.float32,
        device=time_steps.device) / (temb_dim // 2))
        )
    
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb



class DiT(nn.Module):
    def __init__(self, img_Size, img_channels, config):
        super().__init__()

        num_layers = config["num_layers"]
        self.image_height = img_Size
        self.image_width = img_Size
        self.img_channels = img_channels
        self.hidden_size = config['hidden_size']
        self.patch_height = config['patch_size']
        self.patch_width = config['patch_size']


        self.timestep_emb_dim = config['timestep_emb_dim']
        self.spectrum_len = config['spectrum_len']


        self.nh = self.image_height // self.patch_height
        self.nw = self.image_width // self.patch_width

        self.patch_embed_layer = PatchEmbedding(image_height=self.image_height,
                                                image_width=self.image_width,
                                                img_channels=self.img_channels,
                                                patch_height=self.patch_height,
                                                patch_width=self.patch_width,
                                                hidden_size=self.hidden_size)
        

        self.t_proj = nn.Sequential(
            nn.Linear(self.timestep_emb_dim, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        self.spectrum_proj = nn.Sequential(
            nn.Linear(self.spectrum_len, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )


        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1E-6)

        self.adaptive_norm_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=True)
        )

        self.proj_out = nn.Linear(self.hidden_size, self.patch_height * self.patch_width * self.img_channels)


        nn.init.normal_(self.t_proj[0].weight, std=.02)
        nn.init.normal_(self.t_proj[2].weight, std=.02)

        nn.init.normal_(self.spectrum_proj[0].weight, std=.02)
        nn.init.normal_(self.spectrum_proj[2].weight, std=.02)

        nn.init.constant_(self.adaptive_norm_mlp[-1].weight, 0)
        nn.init.constant_(self.adaptive_norm_mlp[-1].bias, 0)

        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)


    def forward(self, x, t, spectrum):

        out = self.patch_embed_layer(x)


        t_emb = get_time_embeddings(torch.as_tensor(t).long(), self.timestep_emb_dim)

        t_emb = self.t_proj(t_emb)
        spectrum_emb = self.spectrum_proj(spectrum)
        c = t_emb + spectrum_emb

        for layer in self.layers:
            out = layer(out, c)


        pre_mlp_shift, pre_mlp_scale = self.adaptive_norm_mlp(c).chunk(2, dim=1)
        out = (self.norm(out) * (1 + pre_mlp_scale.unsqueeze(1)) + pre_mlp_shift.unsqueeze(1))


        out = self.proj_out(out)
        out = rearrange(out, 'b (nh nw) (ph pw c) -> b c (nh ph) (nw pw)',
                        ph=self.patch_height,
                        pw=self.patch_width,
                        nw=self.nw,
                        nh=self.nh)
        
        return out

