import torch
import torch.nn as nn
from Models.Layers.blocks import ResidualAttentionBlock

class SimpleEncoderNextLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.name = config.get('name', 'SimpleEncoderNextLayer')
        self.structure_layers = config.get('structure_layers', 32) 
        self.spectrum_len = config['spectrum_len']
        self.thickness_range = config['thickness_range']
        self.thickness_steps = config.get('thickness_steps', 1)
        self.kernel_size = config.get('kernel_size', 16)
        
        # Calculate vocab size
        self.thickness_vocab_size = int((self.thickness_range[1] - self.thickness_range[0]) / self.thickness_steps) + 1
        
        # Model dimensions
        self.width = config.get('hidden_size', 768)
        self.num_layers = config.get('num_layers', 12)
        self.num_heads = config.get('num_heads', 12)
        scale = self.width ** -0.5
        
        # Embeddings
        self.thickness_embed = nn.Embedding(self.thickness_vocab_size, self.width)
        self.pos_embed_thickness = nn.Parameter(scale * torch.randn(self.structure_layers, self.width))
        
        # Spectrum embedding
        self.spectrum_embed = nn.Conv1d(1, self.width, kernel_size=self.kernel_size, stride=self.kernel_size, bias=True)
        self.spectrum_num_tokens = self.spectrum_len // self.kernel_size
        self.pos_embed_spectrum = nn.Parameter(scale * torch.randn(self.spectrum_num_tokens, self.width))
        
        # Transformer
        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList([
            ResidualAttentionBlock(self.width, self.num_heads, mlp_ratio=4.0, causal=True)
            for _ in range(self.num_layers)
        ])
        self.ln_post = nn.LayerNorm(self.width)
        
        # Head
        self.head = nn.Linear(self.width, self.thickness_vocab_size)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, spectrum, layer_thickness=None):
        """
        Args:
            spectrum: (Batch_Size, Spectrum_Len)
            layer_thickness: Optional (Batch_Size, Sequence_Len) - Indices of thicknesses
        Returns:
            logits: (Batch_Size, Sequence_Len + 1, Vocab_Size)
        """
        spectrum = spectrum.float()
        B = spectrum.shape[0]
        
        # 1. Embed Spectrum
        x_spec = spectrum.unsqueeze(1) # (B, 1, S_len)
        x_spec = self.spectrum_embed(x_spec) # (B, D, S_tokens)
        x_spec = x_spec.permute(0, 2, 1) # (B, S_tokens, D)
        x_spec = x_spec + self.pos_embed_spectrum.unsqueeze(0)
        
        # 2. Embed Thickness (if provided)
        if layer_thickness is not None:
            # Convert normalized float to indices if necessary
            if layer_thickness.is_floating_point():
                # layer_thickness is normalized [0, 1]
                # Convert to indices: idx = round(val * (vocab_size - 1))
                layer_thickness = torch.round(layer_thickness * (self.thickness_vocab_size - 1)).long()
            
            x_thick = self.thickness_embed(layer_thickness) # (B, L, D)
            # Add positional embedding
            L = layer_thickness.shape[1]
            # Ensure L does not exceed structure_layers
            if L > self.structure_layers:
                 x_thick = x_thick[:, :self.structure_layers, :]
                 L = self.structure_layers
            
            x_thick = x_thick + self.pos_embed_thickness[:L].unsqueeze(0)
            
            # Concatenate
            x = torch.cat([x_spec, x_thick], dim=1) # (B, S_tokens + L, D)
        else:
            x = x_spec
            
        # 3. Transformer
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2) # (Seq, B, D)
        
        for block in self.transformer:
            x = block(x)
            
        x = x.permute(1, 0, 2) # (B, Seq, D)
        x = self.ln_post(x)
        
        # 4. Prediction
        # We return logits for the positions that are supposed to predict thickness.
        # Start index: self.spectrum_num_tokens - 1 (The last spectrum token predicts the first thickness)
        start_idx = self.spectrum_num_tokens - 1
        x_out = x[:, start_idx:, :]
        
        logits = self.head(x_out) # (B, L+1, Vocab)
        
        return logits

    @torch.no_grad()
    def generate(self, spectrum):
        """
        Args:
            spectrum: (Batch_Size, Spectrum_Len)
        Returns:
            thickness: (Batch_Size, Structure_Layers) - Normalized thickness values
        """
        spectrum = spectrum.float()
        B = spectrum.shape[0]
        device = spectrum.device
        
        # 1. Embed Spectrum
        x_spec = spectrum.unsqueeze(1) # (B, 1, S_len)
        x_spec = self.spectrum_embed(x_spec) # (B, D, S_tokens)
        x_spec = x_spec.permute(0, 2, 1) # (B, S_tokens, D)
        x_spec = x_spec + self.pos_embed_spectrum.unsqueeze(0)
        
        # Start with spectrum embeddings
        x = x_spec
        
        generated_indices = []
        
        for i in range(self.structure_layers):
            # Transformer
            x_in = self.ln_pre(x)
            x_in = x_in.permute(1, 0, 2) # (Seq, B, D)
            
            for block in self.transformer:
                x_in = block(x_in)
                
            x_out = x_in.permute(1, 0, 2) # (B, Seq, D)
            x_out = self.ln_post(x_out)
            
            # Predict next token from the last embedding
            last_token_embed = x_out[:, -1, :] # (B, D)
            logits = self.head(last_token_embed) # (B, Vocab)
            
            # Greedy decoding
            next_token_idx = torch.argmax(logits, dim=-1) # (B,)
            generated_indices.append(next_token_idx)
            
            # Prepare input for next step
            if i < self.structure_layers - 1:
                x_thick = self.thickness_embed(next_token_idx).unsqueeze(1) # (B, 1, D)
                x_thick = x_thick + self.pos_embed_thickness[i].unsqueeze(0).unsqueeze(0)
                x = torch.cat([x, x_thick], dim=1)
        
        # Stack indices
        generated_indices = torch.stack(generated_indices, dim=1) # (B, Structure_Layers)
        
        # Convert indices to normalized thickness
        # idx = round(val * (vocab_size - 1)) => val = idx / (vocab_size - 1)
        thickness = generated_indices.float() / (self.thickness_vocab_size - 1)
        
        return thickness

