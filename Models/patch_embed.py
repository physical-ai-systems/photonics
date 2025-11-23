import torch
from torch import nn
from einops import rearrange


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_patch_position_embedding(pos_emp_dim, grid_size, device):
    grid_size_h, grid_size_w = grid_size
    grid_h = torch.arange(grid_size_h, dtype=torch.float32, device=device)
    grid_w = torch.arange(grid_size_w, dtype=torch.float32, device=device)

    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
    grid = torch.stack(grid, dim = 0)

    grid_h_positions = grid[0].reshape(-1)
    grid_w_positions = grid[1].reshape(-1)

    factor = 10000 ** ((torch.arange(
        start=0, 
        end= pos_emp_dim // 4,
        dtype=torch.float32,
        device=device) / (pos_emp_dim // 4)
        ))
    
    grid_h_emb = grid_h_positions[:, None].repeat(1, pos_emp_dim // 4) / factor
    grid_h_emb = torch.cat([torch.sin(grid_h_emb), torch.cos(grid_h_emb)], dim=-1)

    grid_w_emb = grid_w_positions[:, None].repeat(1, pos_emp_dim // 4) / factor
    grid_w_emb = torch.cat([torch.sin(grid_w_emb), torch.cos(grid_w_emb)], dim=-1)
    
    pos_emb = torch.cat([grid_h_emb, grid_w_emb], dim=-1)

    return pos_emb



class PatchEmbedding(nn.Module):


    def __init__(self,
                 image_height,
                 image_width,
                 img_channels,
                 patch_height,
                 patch_width,
                 hidden_size):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.img_channels = img_channels
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.hidden_size = hidden_size

        patch_dim  = self.img_channels * self.patch_height * self.patch_width
        self.patch_embed = nn.Linear(patch_dim, self.hidden_size)


        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.constant_(self.patch_embed.bias, 0)


    def forward(self, x):

        grid_size_h = self.image_height // self.patch_height
        grid_size_w = self.image_width // self.patch_width

        out = rearrange(x, 'b c (nh ph) (nw pw) -> b (nh nw) (ph pw c)',
                        ph=self.patch_height,
                        pw=self.patch_width)  
        
        out = self.patch_embed(out)

        pos_embed = get_patch_position_embedding(pos_emp_dim=self.hidden_size,
                                                 grid_size=(grid_size_h, grid_size_w),
                                                 device = x.device)
        
        out += pos_embed
        return out

