import torch
import torch.nn as nn
from Models.Layers.blocks import ResidualAttentionBlock, _expand_token

class SimpleEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.name = config.get('name', 'SimpleEncoder')
        self.structure_layers = config.get('structure_layers', 20) 
        self.spectrum_len = config['spectrum_len']
        self.num_materials = config.get('num_materials', 2)
        
        # TiTok-inspired architecture parameters
        self.num_latent_tokens = int(self.structure_layers ** 1.7)
        self.token_size = config.get('token_size', 2 + self.num_materials)  # thickness + material logits per layer
        
        # Model dimensions from config
        self.width = config.get('hidden_size', 768)
        self.num_layers = config.get('num_layers', 12)
        self.num_heads = config.get('num_heads', 12)
        
        # Spectrum embedding (analogous to patch_embed for images)
        self.spectrum_embed = nn.Linear(self.spectrum_len, self.width, bias=True)
        
        # Learnable embeddings
        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(2, self.width))  # For class token + spectrum
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width))
        
        # Pre-transformer normalization
        self.ln_pre = nn.LayerNorm(self.width)
        
        # Transformer blocks
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))
        
        # Post-transformer normalization
        self.ln_post = nn.LayerNorm(self.width)
        
        # Output heads
        self.thickness_head = nn.Linear(self.width, 1)
        self.material_head = nn.Linear(self.width, self.num_materials)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.thickness_head.weight)
        nn.init.constant_(self.thickness_head.bias, 0.5)
        nn.init.xavier_uniform_(self.material_head.weight)
        nn.init.constant_(self.material_head.bias, 0)

    def forward(self, spectrum, latent_tokens=None):
        """
        Args:
            spectrum: (Batch_Size, Spectrum_Len) - The 1D Reflectance vector
            latent_tokens: Optional (Batch_Size, Num_Latent_Tokens, Width) - Pre-initialized latent tokens
        Returns:
            thickness: (Batch_Size, Structure_Layers) - Predicted thicknesses
            material_logits: (Batch_Size, Structure_Layers, Num_Materials) - Scores for each material
        """
        spectrum = spectrum.float()
        batch_size = spectrum.shape[0]
        
        # Embed spectrum (analogous to image patches)
        x = self.spectrum_embed(spectrum).unsqueeze(1)  # (B, 1, Width)
        
        # Add class embedding and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, batch_size).to(x.dtype), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)  # (B, 2, Width)
        
        # Initialize or use provided latent tokens
        if latent_tokens is None:
            latent_tokens = torch.zeros(batch_size, self.num_latent_tokens, self.width, 
                                       device=spectrum.device, dtype=x.dtype)
        
        latent_tokens = latent_tokens + self.latent_token_positional_embedding.to(x.dtype)
        x = torch.cat([x, latent_tokens], dim=1)  # (B, 2 + num_latent_tokens, Width)
        
        # Apply transformer
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        # Extract and process latent tokens
        latent_tokens = x[:, 2:]  # Skip class token and spectrum token
        latent_tokens = self.ln_post(latent_tokens)
        
        # Generate predictions from latent tokens
        thickness = self.thickness_head(latent_tokens).squeeze(-1)  # (B, Structure_Layers)
        material_logits = self.material_head(latent_tokens)  # (B, Structure_Layers, Num_Materials)
        
        return thickness, material_logits
