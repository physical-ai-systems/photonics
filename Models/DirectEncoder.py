import torch
import torch.nn as nn
from Models.Transformer_block import TransformerBlock

class DirectEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Configuration
        self.output_layers = config.get('output_layers', 20) # Number of layers to predict (matches Dataset num_layers)
        self.hidden_size = config['hidden_size']
        self.spectrum_len = config['spectrum_len']
        self.num_materials = config.get('num_materials', 2) # e.g. Air, SiO2
        self.transformer_depth = config['num_layers'] # Depth of the transformer

        # 1. Learnable "Layer Queries"
        # These are blank templates that the model will fill with information.
        # We create one query for each layer we need to output.
        # Shape: (1, Output_Layers, Hidden)
        self.layer_queries = nn.Parameter(torch.randn(1, self.output_layers, self.hidden_size))

        # 2. Spectrum Encoder
        # Projects the 1D Reflectance vector into a latent conditioning vector
        self.spectrum_proj = nn.Sequential(
            nn.Linear(self.spectrum_len, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        # 3. Transformer Backbone
        # We reuse the existing TransformerBlock.
        # It uses 'adaLN' (Adaptive Layer Norm) where the 'condition' modulates the 'input'.
        # Here: Input = Layer Queries, Condition = Spectrum.
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(self.transformer_depth)
        ])

        # Final Normalization (matching the adaLN style of the blocks)
        self.norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1E-6)
        self.adaptive_norm_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=True)
        )

        # 4. Prediction Heads
        
        # Head A: Thickness Prediction (Regression)
        # Outputs a single continuous value for thickness
        self.thickness_head = nn.Linear(self.hidden_size, 1)
        
        # Head B: Material Prediction (Classification)
        # Outputs logits for each material type (e.g., 2 values: score for Air, score for SiO2)
        self.material_head = nn.Linear(self.hidden_size, self.num_materials)

        # Initialization
        nn.init.normal_(self.layer_queries, std=0.02)
        nn.init.normal_(self.spectrum_proj[0].weight, std=0.02)
        nn.init.normal_(self.spectrum_proj[2].weight, std=0.02)
        
        nn.init.constant_(self.adaptive_norm_mlp[-1].weight, 0)
        nn.init.constant_(self.adaptive_norm_mlp[-1].bias, 0)
        
        nn.init.constant_(self.thickness_head.weight, 0)
        nn.init.constant_(self.thickness_head.bias, 0)
        nn.init.constant_(self.material_head.weight, 0)
        nn.init.constant_(self.material_head.bias, 0)

    def forward(self, spectrum):
        """
        Args:
            spectrum: (Batch_Size, Spectrum_Len) - The 1D Reflectance vector
        Returns:
            thickness: (Batch_Size, Output_Layers) - Predicted thicknesses
            material_logits: (Batch_Size, Output_Layers, Num_Materials) - Scores for each material
        """
        spectrum = spectrum.float() # Ensure float32
        batch_size = spectrum.shape[0]

        # 1. Prepare Input (Queries)
        # Replicate the learnable queries for every item in the batch
        # x shape: (Batch, Output_Layers, Hidden)
        x = self.layer_queries.repeat(batch_size, 1, 1)

        # 2. Prepare Condition (Spectrum)
        # Encode the spectrum into a hidden vector
        # c shape: (Batch, Hidden)
        # Explicitly cast weights to float if needed, though usually not necessary if model is float
        # But if input is float and we get error, maybe weights are double?
        # Let's try to force input to same dtype as weights
        dtype = self.spectrum_proj[0].weight.dtype
        c = self.spectrum_proj(spectrum.to(dtype))

        # 3. Pass through Transformer
        # The queries 'x' attend to themselves, conditioned by the spectrum 'c'
        for layer in self.layers:
            x = layer(x, c)
        
        # 4. Final Normalization
        pre_mlp_shift, pre_mlp_scale = self.adaptive_norm_mlp(c).chunk(2, dim=1)
        x = (self.norm(x) * (1 + pre_mlp_scale.unsqueeze(1)) + pre_mlp_shift.unsqueeze(1))

        # 5. Predict Outputs
        
        # Thickness: Remove the last dimension since it's just 1 value
        thickness = self.thickness_head(x).squeeze(-1) 
        
        # Material: Keep dimensions (Batch, Layers, Materials)
        material_logits = self.material_head(x)

        return thickness, material_logits
