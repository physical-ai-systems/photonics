# SimpleEncoderNextLayer Architecture Analysis

## Overview

The `SimpleEncoderNextLayer` is a Transformer-based model designed to predict the thickness of layers in a photonic structure based on an input spectrum. It treats the problem as a sequence modeling task, where the spectrum is tokenized and serves as the context to autoregressively generate layer thickness values.

## Inputs & Outputs

| Tensor | Shape | Type | Description |
| :--- | :--- | :--- | :--- |
| **Input: `spectrum`** | `(Batch_Size, Spectrum_Len)` | Float | The input optical spectrum intensity values. |
| **Input: `layer_thickness`** | `(Batch_Size, Sequence_Len)` | Float/Long | (Optional) Ground truth layer thicknesses. Used for training (teacher forcing). Values are indices or normalized floats [0, 1]. |
| **Output: `logits`** | `(Batch_Size, Seq_Len_Out, Vocab_Size)` | Float | Logits over the thickness vocabulary for next-token prediction. `Seq_Len_Out` corresponds to the number of thickness layers + 1. |

## Architecture Diagram

```mermaid
graph TD
    subgraph Inputs
        S["Spectrum<br/>(B, Spectrum_Len)"]
        T["Layer Thickness<br/>(B, Seq_Len)"]
    end

    subgraph "Embedding Stage"
        direction TB
        S_R["Reshape<br/>(B, 1, Spectrum_Len)"]
        S_Conv["Conv1d Embedding<br/>k=16, s=16"]
        S_Perm["Permute<br/>(B, S_Tokens, Width)"]
        S_Pos["Add Pos Embed<br/>(S_Tokens, Width)"]
        
        T_Idx["Quantize/Index<br/>(if float)"]
        T_Emb["Embedding Layer<br/>Vocab -> Width"]
        T_Pos["Add Pos Embed<br/>(Seq_Len, Width)"]
        
        Concat[Concatenate]
    end

    subgraph "Transformer Encoder"
        LN_Pre["LayerNorm (Pre)"]
        Perm_In["Permute -> (Seq, B, Width)"]
        
        Blk1["ResidualAttentionBlock 1"]
        Blk_Dots["..."]
        BlkN["ResidualAttentionBlock N"]
        
        Perm_Out["Permute -> (B, Seq, Width)"]
        LN_Post["LayerNorm (Post)"]
    end

    subgraph "Prediction Head"
        Slice["Slice Output<br/>Start from last spectrum token"]
        Linear["Linear Head<br/>Width -> Vocab_Size"]
    end

    subgraph Outputs
        Logits["Logits<br/>(B, Out_Seq, Vocab_Size)"]
    end

    %% Data Flow
    S --> S_R --> S_Conv --> S_Perm --> S_Pos
    
    T -.->|If Provided| T_Idx --> T_Emb --> T_Pos
    
    S_Pos --> Concat
    T_Pos --> Concat
    
    Concat --> LN_Pre --> Perm_In --> Blk1 --> Blk_Dots --> BlkN --> Perm_Out --> LN_Post
    
    LN_Post --> Slice --> Linear --> Logits
```

