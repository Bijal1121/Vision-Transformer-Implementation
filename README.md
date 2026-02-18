# Vision Transformer (ViT) — From Scratch & Transfer Learning

This repository demonstrates two implementations of **Vision Transformers (ViT)** using PyTorch:

1. **ViT implemented from scratch** on the MNIST dataset  
2. **Pretrained ViT fine-tuned** on CIFAR-10  

The goal of this project is to understand both:
- The internal architecture of Vision Transformers  
- Practical transfer learning and fine-tuning workflows  

---

## Repository Structure
├── ViT_Implementation.ipynb
├── ViT_using_pretrained_model.ipynb
└── README.md
---

# 1️⃣ Vision Transformer from Scratch (MNIST)

**File:** `ViT_Implementation.ipynb`

## Overview

This notebook implements the Vision Transformer architecture from first principles using PyTorch.  
No pretrained transformer layers are used.

The model is trained on the **MNIST dataset** for handwritten digit classification.

---

## Architecture Components Implemented

### Patch Embedding
- Image divided into fixed-size patches  
- Each patch flattened and projected to embedding dimension  
- Converts image into a sequence of tokens  

### Learnable Class Token
- `[CLS]` token prepended to the patch sequence  
- Used for final classification  

### Positional Embeddings
- Learnable positional encodings added  
- Preserves spatial information  

### Multi-Head Self-Attention
- Query, Key, Value projections  
- Scaled dot-product attention  
- Multiple attention heads  
- Output projection layer  

### Transformer Encoder Block
Each encoder layer contains:
- Layer Normalization  
- Multi-Head Self-Attention  
- Residual connections  
- Feed-forward network (MLP)  
- Dropout (if applied)  

### Classification Head
- Final `[CLS]` token passed to linear layer  
- Outputs class probabilities (0–9)  

---

## Training Details

- Dataset: MNIST  
- Optimizer: Adam  
- Loss Function: CrossEntropyLoss  
- Evaluation Metric: Accuracy  

This implementation demonstrates a full understanding of transformer internals and end-to-end training.

---

# 2️⃣ Vision Transformer Using Pretrained Model (CIFAR-10)

**File:** `ViT_using_pretrained_model.ipynb`

## Overview

This notebook demonstrates transfer learning using a pretrained Vision Transformer model.

Base model:
- `vit_b_16` from `torchvision`
- Pretrained on ImageNet-1K  

The model is fine-tuned on **CIFAR-10**.

---

## Implementation Steps

### Load Pretrained Model

```python
from torchvision.models import vit_b_16, ViT_B_16_Weights
model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

