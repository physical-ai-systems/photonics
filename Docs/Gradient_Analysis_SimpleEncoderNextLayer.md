# Gradient & Weight Analysis: SimpleEncoderNextLayer

## Experimental Setup
### Configuration
| Parameter | Value |
| :--- | :--- |
| `spectrum_len` | `301` |
| `thickness_range` | `[0, 50]` |
| `structure_layers` | `32` |
| `name` | `SimpleEncoderNextLayer` |
| `hidden_size` | `768` |
| `num_layers` | `12` |
| `num_heads` | `12` |

- **Batch Size**: 4
- **Initialization**: Random (trunc_normal_ std=0.02)

## Potential Issues Detected
No obvious vanishing (Var < 1e-8) or exploding (Var > 10) gradients detected in this pass.

## Detailed Layer Statistics

| Layer Name | Weight Mean | Weight Var | Grad Mean | Grad Var | Grad Norm | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `pos_embed_thickness` | 9.56e-05 | 1.30e-03 | 2.43e-11 | 5.05e-05 | 1.11e+00 |  |
| `pos_embed_spectrum` | 6.53e-04 | 1.30e-03 | -4.45e-12 | 2.59e-06 | 1.89e-01 |  |
| `thickness_embed.weight` | 3.19e-05 | 4.02e-04 | 1.14e-11 | 3.09e-05 | 1.10e+00 |  |
| `spectrum_embed.weight` | 4.04e-04 | 4.02e-04 | 3.14e-11 | 4.54e-05 | 7.47e-01 |  |
| `spectrum_embed.bias` | 0.00e+00 | 0.00e+00 | -1.41e-10 | 3.40e-04 | 5.10e-01 |  |
| `ln_pre.weight` | 1.00e+00 | 0.00e+00 | 3.90e-05 | 3.08e-06 | 4.86e-02 |  |
| `ln_pre.bias` | 0.00e+00 | 0.00e+00 | -9.70e-12 | 7.56e-06 | 7.61e-02 |  |
| `transformer.0.ln_1.weight` | 1.00e+00 | 0.00e+00 | -6.43e-06 | 9.29e-08 | 8.44e-03 |  |
| `transformer.0.ln_1.bias` | 0.00e+00 | 0.00e+00 | -6.56e-05 | 9.64e-07 | 2.73e-02 |  |
| `transformer.0.attn.in_proj_weight` | 1.29e-05 | 6.51e-04 | 6.50e-14 | 6.24e-08 | 3.32e-01 |  |
| `transformer.0.attn.in_proj_bias` | 0.00e+00 | 0.00e+00 | 3.70e-06 | 6.60e-07 | 3.90e-02 |  |
| `transformer.0.attn.out_proj.weight` | -2.80e-05 | 4.00e-04 | 1.45e-13 | 2.56e-07 | 3.88e-01 |  |
| `transformer.0.attn.out_proj.bias` | 0.00e+00 | 0.00e+00 | -2.43e-11 | 6.29e-06 | 6.95e-02 |  |
| `transformer.0.ln_2.weight` | 1.00e+00 | 0.00e+00 | -3.36e-05 | 3.33e-07 | 1.60e-02 |  |
| `transformer.0.ln_2.bias` | 0.00e+00 | 0.00e+00 | -2.21e-08 | 5.45e-07 | 2.05e-02 |  |
| `transformer.0.mlp.c_fc.weight` | -3.83e-06 | 4.00e-04 | 4.34e-14 | 3.05e-07 | 8.48e-01 |  |
| `transformer.0.mlp.c_fc.bias` | 0.00e+00 | 0.00e+00 | -2.13e-05 | 4.93e-07 | 3.89e-02 |  |
| `transformer.0.mlp.c_proj.weight` | -5.21e-07 | 4.00e-04 | -2.83e-12 | 3.19e-07 | 8.67e-01 |  |
| `transformer.0.mlp.c_proj.bias` | 0.00e+00 | 0.00e+00 | -9.70e-12 | 5.53e-06 | 6.51e-02 |  |
| `transformer.1.ln_1.weight` | 1.00e+00 | 0.00e+00 | -2.97e-06 | 9.11e-08 | 8.36e-03 |  |
| `transformer.1.ln_1.bias` | 0.00e+00 | 0.00e+00 | -1.23e-05 | 7.15e-07 | 2.34e-02 |  |
| `transformer.1.attn.in_proj_weight` | 2.51e-05 | 6.51e-04 | -7.29e-14 | 5.84e-08 | 3.22e-01 |  |
| `transformer.1.attn.in_proj_bias` | 0.00e+00 | 0.00e+00 | -3.16e-07 | 4.77e-07 | 3.32e-02 |  |
| `transformer.1.attn.out_proj.weight` | -2.87e-05 | 4.00e-04 | -2.78e-13 | 2.46e-07 | 3.81e-01 |  |
| `transformer.1.attn.out_proj.bias` | 0.00e+00 | 0.00e+00 | -4.85e-12 | 4.85e-06 | 6.10e-02 |  |
| `transformer.1.ln_2.weight` | 1.00e+00 | 0.00e+00 | 2.74e-05 | 3.20e-07 | 1.57e-02 |  |
| `transformer.1.ln_2.bias` | 0.00e+00 | 0.00e+00 | 1.87e-06 | 5.01e-07 | 1.96e-02 |  |
| `transformer.1.mlp.c_fc.weight` | -7.02e-06 | 4.00e-04 | -8.45e-14 | 2.75e-07 | 8.05e-01 |  |
| `transformer.1.mlp.c_fc.bias` | 0.00e+00 | 0.00e+00 | 1.00e-05 | 4.31e-07 | 3.64e-02 |  |
| `transformer.1.mlp.c_proj.weight` | -5.44e-06 | 4.00e-04 | -2.43e-12 | 2.79e-07 | 8.11e-01 |  |
| `transformer.1.mlp.c_proj.bias` | 0.00e+00 | 0.00e+00 | -2.43e-11 | 4.37e-06 | 5.79e-02 |  |
| `transformer.2.ln_1.weight` | 1.00e+00 | 0.00e+00 | -3.60e-06 | 8.30e-08 | 7.98e-03 |  |
| `transformer.2.ln_1.bias` | 0.00e+00 | 0.00e+00 | -4.81e-05 | 5.84e-07 | 2.12e-02 |  |
| `transformer.2.attn.in_proj_weight` | 1.09e-05 | 6.51e-04 | -6.97e-14 | 5.70e-08 | 3.18e-01 |  |
| `transformer.2.attn.in_proj_bias` | 0.00e+00 | 0.00e+00 | 1.20e-05 | 4.24e-07 | 3.12e-02 |  |
| `transformer.2.attn.out_proj.weight` | -1.20e-05 | 4.01e-04 | -2.27e-13 | 2.41e-07 | 3.77e-01 |  |
| `transformer.2.attn.out_proj.bias` | 0.00e+00 | 0.00e+00 | -9.70e-12 | 3.99e-06 | 5.53e-02 |  |
| `transformer.2.ln_2.weight` | 1.00e+00 | 0.00e+00 | 1.73e-05 | 3.12e-07 | 1.55e-02 |  |
| `transformer.2.ln_2.bias` | 0.00e+00 | 0.00e+00 | -1.14e-05 | 4.40e-07 | 1.84e-02 |  |
| `transformer.2.mlp.c_fc.weight` | -6.23e-06 | 4.00e-04 | -1.34e-13 | 2.43e-07 | 7.57e-01 |  |
| `transformer.2.mlp.c_fc.bias` | 0.00e+00 | 0.00e+00 | 4.73e-06 | 3.56e-07 | 3.31e-02 |  |
| `transformer.2.mlp.c_proj.weight` | -1.51e-05 | 4.00e-04 | 0.00e+00 | 2.46e-07 | 7.62e-01 |  |
| `transformer.2.mlp.c_proj.bias` | 0.00e+00 | 0.00e+00 | -2.43e-11 | 3.66e-06 | 5.30e-02 |  |
| `transformer.3.ln_1.weight` | 1.00e+00 | 0.00e+00 | 1.54e-05 | 7.61e-08 | 7.65e-03 |  |
| `transformer.3.ln_1.bias` | 0.00e+00 | 0.00e+00 | -1.76e-05 | 4.65e-07 | 1.89e-02 |  |
| `transformer.3.attn.in_proj_weight` | 1.29e-05 | 6.51e-04 | -5.79e-15 | 4.90e-08 | 2.94e-01 |  |
| `transformer.3.attn.in_proj_bias` | 0.00e+00 | 0.00e+00 | 1.97e-05 | 3.04e-07 | 2.65e-02 |  |
| `transformer.3.attn.out_proj.weight` | 1.87e-05 | 3.99e-04 | 1.26e-13 | 2.11e-07 | 3.53e-01 |  |
| `transformer.3.attn.out_proj.bias` | 0.00e+00 | 0.00e+00 | 4.85e-12 | 3.32e-06 | 5.05e-02 |  |
| `transformer.3.ln_2.weight` | 1.00e+00 | 0.00e+00 | -4.30e-05 | 3.18e-07 | 1.57e-02 |  |
| `transformer.3.ln_2.bias` | 0.00e+00 | 0.00e+00 | 8.48e-06 | 3.67e-07 | 1.68e-02 |  |
| `transformer.3.mlp.c_fc.weight` | -4.05e-06 | 4.00e-04 | 1.37e-13 | 2.21e-07 | 7.22e-01 |  |
| `transformer.3.mlp.c_fc.bias` | 0.00e+00 | 0.00e+00 | -7.07e-06 | 3.13e-07 | 3.10e-02 |  |
| `transformer.3.mlp.c_proj.weight` | -2.57e-06 | 4.00e-04 | -4.04e-13 | 2.23e-07 | 7.25e-01 |  |
| `transformer.3.mlp.c_proj.bias` | 0.00e+00 | 0.00e+00 | -1.46e-11 | 3.06e-06 | 4.84e-02 |  |
| `transformer.4.ln_1.weight` | 1.00e+00 | 0.00e+00 | -3.33e-06 | 6.55e-08 | 7.09e-03 |  |
| `transformer.4.ln_1.bias` | 0.00e+00 | 0.00e+00 | 3.31e-05 | 4.35e-07 | 1.83e-02 |  |
| `transformer.4.attn.in_proj_weight` | -2.05e-05 | 6.51e-04 | -2.99e-14 | 4.94e-08 | 2.96e-01 |  |
| `transformer.4.attn.in_proj_bias` | 0.00e+00 | 0.00e+00 | 1.11e-05 | 3.08e-07 | 2.67e-02 |  |
| `transformer.4.attn.out_proj.weight` | 1.23e-05 | 4.00e-04 | 1.26e-13 | 2.18e-07 | 3.59e-01 |  |
| `transformer.4.attn.out_proj.bias` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 2.78e-06 | 4.61e-02 |  |
| `transformer.4.ln_2.weight` | 1.00e+00 | 0.00e+00 | -2.14e-05 | 2.29e-07 | 1.33e-02 |  |
| `transformer.4.ln_2.bias` | 0.00e+00 | 0.00e+00 | 2.90e-05 | 3.27e-07 | 1.59e-02 |  |
| `transformer.4.mlp.c_fc.weight` | -1.88e-05 | 4.00e-04 | -3.71e-14 | 1.99e-07 | 6.85e-01 |  |
| `transformer.4.mlp.c_fc.bias` | 0.00e+00 | 0.00e+00 | -1.08e-05 | 2.68e-07 | 2.87e-02 |  |
| `transformer.4.mlp.c_proj.weight` | 1.37e-05 | 4.00e-04 | -1.41e-12 | 1.96e-07 | 6.81e-01 |  |
| `transformer.4.mlp.c_proj.bias` | 0.00e+00 | 0.00e+00 | -1.46e-11 | 2.56e-06 | 4.43e-02 |  |
| `transformer.5.ln_1.weight` | 1.00e+00 | 0.00e+00 | 1.12e-05 | 6.73e-08 | 7.19e-03 |  |
| `transformer.5.ln_1.bias` | 0.00e+00 | 0.00e+00 | -1.48e-05 | 4.09e-07 | 1.77e-02 |  |
| `transformer.5.attn.in_proj_weight` | 2.97e-05 | 6.51e-04 | 3.16e-15 | 4.26e-08 | 2.75e-01 |  |
| `transformer.5.attn.in_proj_bias` | 0.00e+00 | 0.00e+00 | 2.44e-06 | 2.32e-07 | 2.31e-02 |  |
| `transformer.5.attn.out_proj.weight` | 8.92e-06 | 4.00e-04 | 6.32e-15 | 1.84e-07 | 3.29e-01 |  |
| `transformer.5.attn.out_proj.bias` | 0.00e+00 | 0.00e+00 | -2.43e-11 | 2.31e-06 | 4.21e-02 |  |
| `transformer.5.ln_2.weight` | 1.00e+00 | 0.00e+00 | -3.17e-05 | 2.14e-07 | 1.28e-02 |  |
| `transformer.5.ln_2.bias` | 0.00e+00 | 0.00e+00 | -2.75e-05 | 2.80e-07 | 1.47e-02 |  |
| `transformer.5.mlp.c_fc.weight` | 3.95e-06 | 4.00e-04 | 1.80e-13 | 1.82e-07 | 6.56e-01 |  |
| `transformer.5.mlp.c_fc.bias` | 0.00e+00 | 0.00e+00 | -1.83e-05 | 2.31e-07 | 2.66e-02 |  |
| `transformer.5.mlp.c_proj.weight` | 9.50e-06 | 4.00e-04 | -2.02e-12 | 1.77e-07 | 6.46e-01 |  |
| `transformer.5.mlp.c_proj.bias` | 0.00e+00 | 0.00e+00 | -1.46e-11 | 2.23e-06 | 4.14e-02 |  |
| `transformer.6.ln_1.weight` | 1.00e+00 | 0.00e+00 | 2.49e-05 | 6.22e-08 | 6.94e-03 |  |
| `transformer.6.ln_1.bias` | 0.00e+00 | 0.00e+00 | 1.55e-05 | 3.25e-07 | 1.58e-02 |  |
| `transformer.6.attn.in_proj_weight` | 1.49e-05 | 6.51e-04 | -5.50e-14 | 4.14e-08 | 2.71e-01 |  |
| `transformer.6.attn.in_proj_bias` | 0.00e+00 | 0.00e+00 | 1.10e-05 | 2.24e-07 | 2.27e-02 |  |
| `transformer.6.attn.out_proj.weight` | 3.08e-05 | 4.00e-04 | -1.89e-13 | 1.78e-07 | 3.24e-01 |  |
| `transformer.6.attn.out_proj.bias` | 0.00e+00 | 0.00e+00 | -1.46e-11 | 2.02e-06 | 3.94e-02 |  |
| `transformer.6.ln_2.weight` | 1.00e+00 | 0.00e+00 | -7.60e-06 | 1.94e-07 | 1.22e-02 |  |
| `transformer.6.ln_2.bias` | 0.00e+00 | 0.00e+00 | 1.95e-05 | 2.17e-07 | 1.29e-02 |  |
| `transformer.6.mlp.c_fc.weight` | 2.52e-05 | 4.00e-04 | 2.25e-14 | 1.66e-07 | 6.25e-01 |  |
| `transformer.6.mlp.c_fc.bias` | 0.00e+00 | 0.00e+00 | -5.94e-06 | 1.93e-07 | 2.43e-02 |  |
| `transformer.6.mlp.c_proj.weight` | 1.51e-06 | 4.00e-04 | -1.21e-12 | 1.59e-07 | 6.13e-01 |  |
| `transformer.6.mlp.c_proj.bias` | 0.00e+00 | 0.00e+00 | -1.46e-11 | 1.90e-06 | 3.82e-02 |  |
| `transformer.7.ln_1.weight` | 1.00e+00 | 0.00e+00 | 1.19e-05 | 5.64e-08 | 6.58e-03 |  |
| `transformer.7.ln_1.bias` | 0.00e+00 | 0.00e+00 | 1.29e-05 | 2.96e-07 | 1.51e-02 |  |
| `transformer.7.attn.in_proj_weight` | 4.99e-05 | 6.50e-04 | 1.18e-14 | 3.73e-08 | 2.57e-01 |  |
| `transformer.7.attn.in_proj_bias` | 0.00e+00 | 0.00e+00 | 1.18e-05 | 1.82e-07 | 2.05e-02 |  |
| `transformer.7.attn.out_proj.weight` | 4.05e-06 | 4.00e-04 | 3.79e-13 | 1.56e-07 | 3.03e-01 |  |
| `transformer.7.attn.out_proj.bias` | 0.00e+00 | 0.00e+00 | -7.28e-12 | 1.74e-06 | 3.65e-02 |  |
| `transformer.7.ln_2.weight` | 1.00e+00 | 0.00e+00 | 9.01e-06 | 1.90e-07 | 1.21e-02 |  |
| `transformer.7.ln_2.bias` | 0.00e+00 | 0.00e+00 | 2.39e-05 | 2.31e-07 | 1.33e-02 |  |
| `transformer.7.mlp.c_fc.weight` | 1.60e-05 | 4.00e-04 | -1.84e-13 | 1.56e-07 | 6.07e-01 |  |
| `transformer.7.mlp.c_fc.bias` | 0.00e+00 | 0.00e+00 | 1.56e-06 | 1.76e-07 | 2.33e-02 |  |
| `transformer.7.mlp.c_proj.weight` | -1.89e-05 | 4.00e-04 | -1.62e-12 | 1.49e-07 | 5.92e-01 |  |
| `transformer.7.mlp.c_proj.bias` | 0.00e+00 | 0.00e+00 | -4.85e-12 | 1.60e-06 | 3.50e-02 |  |
| `transformer.8.ln_1.weight` | 1.00e+00 | 0.00e+00 | -9.93e-06 | 4.89e-08 | 6.13e-03 |  |
| `transformer.8.ln_1.bias` | 0.00e+00 | 0.00e+00 | -8.37e-06 | 2.20e-07 | 1.30e-02 |  |
| `transformer.8.attn.in_proj_weight` | -6.34e-06 | 6.51e-04 | 3.83e-14 | 3.27e-08 | 2.40e-01 |  |
| `transformer.8.attn.in_proj_bias` | 0.00e+00 | 0.00e+00 | -4.25e-06 | 1.47e-07 | 1.84e-02 |  |
| `transformer.8.attn.out_proj.weight` | -1.01e-05 | 4.01e-04 | 1.59e-13 | 1.65e-07 | 3.12e-01 |  |
| `transformer.8.attn.out_proj.bias` | 0.00e+00 | 0.00e+00 | -2.43e-11 | 1.55e-06 | 3.45e-02 |  |
| `transformer.8.ln_2.weight` | 1.00e+00 | 0.00e+00 | 1.34e-05 | 1.86e-07 | 1.19e-02 |  |
| `transformer.8.ln_2.bias` | 0.00e+00 | 0.00e+00 | 5.37e-06 | 1.97e-07 | 1.23e-02 |  |
| `transformer.8.mlp.c_fc.weight` | 3.98e-06 | 4.00e-04 | 1.56e-13 | 1.43e-07 | 5.81e-01 |  |
| `transformer.8.mlp.c_fc.bias` | 0.00e+00 | 0.00e+00 | -2.63e-06 | 1.59e-07 | 2.21e-02 |  |
| `transformer.8.mlp.c_proj.weight` | -6.95e-06 | 4.00e-04 | -2.02e-12 | 1.39e-07 | 5.72e-01 |  |
| `transformer.8.mlp.c_proj.bias` | 0.00e+00 | 0.00e+00 | -4.85e-12 | 1.49e-06 | 3.38e-02 |  |
| `transformer.9.ln_1.weight` | 1.00e+00 | 0.00e+00 | -5.44e-06 | 5.25e-08 | 6.35e-03 |  |
| `transformer.9.ln_1.bias` | 0.00e+00 | 0.00e+00 | 2.15e-05 | 2.37e-07 | 1.35e-02 |  |
| `transformer.9.attn.in_proj_weight` | -5.75e-06 | 6.51e-04 | 9.24e-14 | 3.41e-08 | 2.46e-01 |  |
| `transformer.9.attn.in_proj_bias` | 0.00e+00 | 0.00e+00 | -1.60e-05 | 1.58e-07 | 1.91e-02 |  |
| `transformer.9.attn.out_proj.weight` | -4.62e-05 | 4.06e-04 | -7.58e-14 | 1.53e-07 | 3.00e-01 |  |
| `transformer.9.attn.out_proj.bias` | 0.00e+00 | 0.00e+00 | -1.46e-11 | 1.40e-06 | 3.28e-02 |  |
| `transformer.9.ln_2.weight` | 1.00e+00 | 0.00e+00 | -2.37e-05 | 1.55e-07 | 1.09e-02 |  |
| `transformer.9.ln_2.bias` | 0.00e+00 | 0.00e+00 | 1.90e-05 | 1.75e-07 | 1.16e-02 |  |
| `transformer.9.mlp.c_fc.weight` | -4.29e-08 | 4.00e-04 | -5.76e-14 | 1.37e-07 | 5.68e-01 |  |
| `transformer.9.mlp.c_fc.bias` | 0.00e+00 | 0.00e+00 | -2.04e-06 | 1.44e-07 | 2.10e-02 |  |
| `transformer.9.mlp.c_proj.weight` | -9.03e-06 | 4.00e-04 | -1.01e-12 | 1.31e-07 | 5.57e-01 |  |
| `transformer.9.mlp.c_proj.bias` | 0.00e+00 | 0.00e+00 | -1.70e-11 | 1.32e-06 | 3.18e-02 |  |
| `transformer.10.ln_1.weight` | 1.00e+00 | 0.00e+00 | 1.50e-05 | 5.07e-08 | 6.25e-03 |  |
| `transformer.10.ln_1.bias` | 0.00e+00 | 0.00e+00 | -1.02e-05 | 1.95e-07 | 1.22e-02 |  |
| `transformer.10.attn.in_proj_weight` | -2.39e-05 | 6.52e-04 | -2.59e-14 | 3.14e-08 | 2.36e-01 |  |
| `transformer.10.attn.in_proj_bias` | 0.00e+00 | 0.00e+00 | -1.15e-05 | 1.30e-07 | 1.73e-02 |  |
| `transformer.10.attn.out_proj.weight` | -7.38e-06 | 4.01e-04 | 0.00e+00 | 1.47e-07 | 2.95e-01 |  |
| `transformer.10.attn.out_proj.bias` | 0.00e+00 | 0.00e+00 | -1.94e-11 | 1.25e-06 | 3.09e-02 |  |
| `transformer.10.ln_2.weight` | 1.00e+00 | 0.00e+00 | -1.24e-05 | 1.50e-07 | 1.07e-02 |  |
| `transformer.10.ln_2.bias` | 0.00e+00 | 0.00e+00 | 2.75e-05 | 1.57e-07 | 1.10e-02 |  |
| `transformer.10.mlp.c_fc.weight` | -2.43e-05 | 4.00e-04 | -4.38e-14 | 1.26e-07 | 5.45e-01 |  |
| `transformer.10.mlp.c_fc.bias` | 0.00e+00 | 0.00e+00 | -1.71e-06 | 1.24e-07 | 1.95e-02 |  |
| `transformer.10.mlp.c_proj.weight` | 2.25e-06 | 4.00e-04 | -8.08e-13 | 1.19e-07 | 5.30e-01 |  |
| `transformer.10.mlp.c_proj.bias` | 0.00e+00 | 0.00e+00 | -7.28e-12 | 1.17e-06 | 3.00e-02 |  |
| `transformer.11.ln_1.weight` | 1.00e+00 | 0.00e+00 | 6.36e-06 | 4.28e-08 | 5.73e-03 |  |
| `transformer.11.ln_1.bias` | 0.00e+00 | 0.00e+00 | -1.26e-06 | 1.62e-07 | 1.12e-02 |  |
| `transformer.11.attn.in_proj_weight` | 2.10e-06 | 6.51e-04 | 3.71e-14 | 2.87e-08 | 2.25e-01 |  |
| `transformer.11.attn.in_proj_bias` | 0.00e+00 | 0.00e+00 | 6.75e-06 | 1.11e-07 | 1.60e-02 |  |
| `transformer.11.attn.out_proj.weight` | -2.11e-05 | 3.99e-04 | 2.53e-14 | 1.26e-07 | 2.72e-01 |  |
| `transformer.11.attn.out_proj.bias` | 0.00e+00 | 0.00e+00 | -1.21e-11 | 1.13e-06 | 2.95e-02 |  |
| `transformer.11.ln_2.weight` | 1.00e+00 | 0.00e+00 | 4.29e-06 | 1.51e-07 | 1.08e-02 |  |
| `transformer.11.ln_2.bias` | 0.00e+00 | 0.00e+00 | 1.14e-05 | 1.52e-07 | 1.08e-02 |  |
| `transformer.11.mlp.c_fc.weight` | 7.63e-06 | 4.00e-04 | -3.00e-14 | 1.19e-07 | 5.30e-01 |  |
| `transformer.11.mlp.c_fc.bias` | 0.00e+00 | 0.00e+00 | 4.02e-06 | 1.26e-07 | 1.97e-02 |  |
| `transformer.11.mlp.c_proj.weight` | 2.08e-05 | 4.00e-04 | -7.07e-13 | 1.13e-07 | 5.16e-01 |  |
| `transformer.11.mlp.c_proj.bias` | 0.00e+00 | 0.00e+00 | -4.85e-12 | 1.09e-06 | 2.89e-02 |  |
| `ln_post.weight` | 1.00e+00 | 0.00e+00 | 3.14e-04 | 2.96e-06 | 4.85e-02 |  |
| `ln_post.bias` | 0.00e+00 | 0.00e+00 | -3.56e-06 | 3.09e-06 | 4.87e-02 |  |
| `head.weight` | -7.50e-05 | 4.02e-04 | 5.33e-12 | 1.45e-04 | 2.38e+00 |  |
| `head.bias` | 0.00e+00 | 0.00e+00 | -4.38e-10 | 1.51e-04 | 8.69e-02 |  |
