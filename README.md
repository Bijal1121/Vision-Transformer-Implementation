# Vision Transformer (ViT) — From Scratch & Transfer Learning

This repository contains two implementations of **Vision Transformers (ViT)** using PyTorch:

1. A complete **ViT architecture implemented from scratch**
2. A **pretrained ViT fine-tuned** using transfer learning

The project demonstrates both architectural understanding and a practical deep learning workflow.

---

## Repository Structure

```
.
├── ViT_Implementation.ipynb
├── ViT_using_pretrained_model.ipynb
└── README.md
```

---

# 1️⃣ Vision Transformer From Scratch (MNIST)

**Notebook:** `ViT_Implementation.ipynb`

## Objective

Manually implement the Vision Transformer architecture without relying on pretrained transformer modules.

---

## Architecture Overview

The following components are implemented from first principles:

### Patch Embedding
- Image split into fixed-size patches  
- Patches flattened and projected into embedding space  
- Image converted into a token sequence  

### Learnable `[CLS]` Token
- Prepended to the token sequence  
- Used for final classification  

### Positional Embeddings
- Learnable positional encodings  
- Preserve spatial ordering  

### Multi-Head Self-Attention
- Query, Key, Value projections  
- Scaled dot-product attention  
- Multiple attention heads  
- Output projection  

### Transformer Encoder Block
Each encoder block includes:
- Layer Normalization  
- Multi-Head Self-Attention  
- Residual connections  
- Feed-forward MLP  
- Dropout (if applied)  

### Classification Head
- Final `[CLS]` token passed to linear layer  
- Outputs digit probabilities (0–9)

---

## Model Configuration (From Scratch)

- Dataset: MNIST  
- Optimizer: Adam  
- Loss: CrossEntropyLoss  
- Evaluation Metric: Accuracy  
- Fully trained from scratch  

---

# 2️⃣ Pretrained ViT Fine-Tuning (CIFAR-10)

**Notebook:** `ViT_using_pretrained_model.ipynb`

## Objective

Apply transfer learning using a pretrained Vision Transformer and adapt it to CIFAR-10.

---

## Base Model

- `vit_b_16` from `torchvision`
- Pretrained on ImageNet-1K

```python
from torchvision.models import vit_b_16, ViT_B_16_Weights
model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
```

---

## Replace Classification Head

```python
model.heads.head = torch.nn.Linear(768, 10)
```

---

## Fine-Tuning Configuration

- Optimizer: Adam  
- Learning Rate: 5e-5  
- LR Scheduler: ExponentialLR  
- Evaluation Metric: Accuracy  
- Entire model fine-tuned  

---

# Datasets

- **MNIST**  
  Loaded using `torchvision.datasets.MNIST`  
  Automatically downloaded via PyTorch  

- **CIFAR-10**  
  Loaded using `torchvision.datasets.CIFAR10`  
  Automatically downloaded via PyTorch  

No manual dataset download is required.

---

# Results

| Model | Dataset | Approach | Accuracy |
|-------|----------|----------|----------|
| ViT (From Scratch) | MNIST | Full custom implementation | 93% |
| Pretrained ViT | CIFAR-10 | Transfer learning + fine-tuning | 95% |

---

# Requirements

## Recommended Environment

- Python 3.8+  
- PyTorch  
- torchvision  
- torchmetrics  
- NumPy  
- Matplotlib  

## Install Dependencies

```bash
pip install torch torchvision torchmetrics numpy matplotlib
```

---

# How to Run

1. Clone the repository  
2. Install dependencies  
3. Open the notebooks:
   - `ViT_Implementation.ipynb`
   - `ViT_using_pretrained_model.ipynb`
4. Run all cells sequentially  

GPU is recommended for faster training, but not required.

---
  
