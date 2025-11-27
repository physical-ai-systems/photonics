import torch
import torch.nn as nn
from Models.Layers.Transformer_block import TransformerBlock

class DirectEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.structure_layers = config.get('structure_layers', 20) 
        self.hidden_size = config['hidden_size']
        self.spectrum_len = config['spectrum_len']
        self.num_materials = config.get('num_materials', 2)
        self.transformer_depth = config['num_layers'] 

       
        self.layer_queries = nn.Parameter(torch.randn(1, self.structure_layers, self.hidden_size))

        self.spectrum_proj = nn.Sequential(
            nn.Linear(self.spectrum_len, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(self.transformer_depth)
        ])

        self.norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1E-6)
        self.adaptive_norm_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=True)
        )

       
        self.thickness_head = nn.Linear(self.hidden_size, 1)
        self.material_head = nn.Linear(self.hidden_size, self.num_materials)

        nn.init.normal_(self.layer_queries, std=0.02)
        nn.init.normal_(self.spectrum_proj[0].weight, std=0.02)
        nn.init.normal_(self.spectrum_proj[2].weight, std=0.02)
        
        nn.init.constant_(self.adaptive_norm_mlp[-1].weight, 0)
        nn.init.constant_(self.adaptive_norm_mlp[-1].bias, 0)
        
        nn.init.xavier_uniform_(self.thickness_head.weight)
        nn.init.constant_(self.thickness_head.bias, 0.5) 
        
        nn.init.xavier_uniform_(self.material_head.weight)
        nn.init.constant_(self.material_head.bias, 0)

    def forward(self, spectrum):
        """
        Args:
            spectrum: (Batch_Size, Spectrum_Len) - The 1D Reflectance vector
        Returns:
            thickness: (Batch_Size, Structure_Layers) - Predicted thicknesses
            material_logits: (Batch_Size, Structure_Layers, Num_Materials) - Scores for each material
        """
        spectrum = spectrum.float() 
        batch_size = spectrum.shape[0]

        x = self.layer_queries.repeat(batch_size, 1, 1)

        dtype = self.spectrum_proj[0].weight.dtype
        c = self.spectrum_proj(spectrum.to(dtype))

        for layer in self.layers:
            x = layer(x, c)
        
        pre_mlp_shift, pre_mlp_scale = self.adaptive_norm_mlp(c).chunk(2, dim=1)
        x = (self.norm(x) * (1 + pre_mlp_scale.unsqueeze(1)) + pre_mlp_shift.unsqueeze(1))

        thickness = self.thickness_head(x).squeeze(-1) 
        
        material_logits = self.material_head(x)

        return thickness, material_logits
