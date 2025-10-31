# LoRA Quick Reference Guide for Diffusion Models

## Core Concept

**Paper:** Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models"  
**arXiv:** 2106.09685

### The Main Idea (Page 2, Section 1)

Instead of fine-tuning all parameters of a large pre-trained model, LoRA freezes the original weights and trains small low-rank matrices that are added to the model.

---

## Key Mathematical Formula

### Low-Rank Decomposition (Page 4, Section 4.1, Equation 3)

```
h = W₀x + ΔWx = W₀x + BAx
```

Where:
- **W₀ ∈ ℝᵈˣᵏ** = Pre-trained weight matrix (FROZEN, no gradients)
- **B ∈ ℝᵈˣʳ** = Trainable low-rank matrix  
- **A ∈ ℝʳˣᵏ** = Trainable low-rank matrix
- **r ≪ min(d, k)** = Rank (typically 1-64)
- **ΔW = BA** = The update to the weights

### Initialization (Page 4, Section 4.1)

```
A ~ N(0, σ²)   [Random Gaussian]
B = 0           [Zero matrix]
```

This ensures **ΔW = BA = 0** at the start of training.

### Scaling (Page 4, Section 4.1)

```
Output = W₀x + (α/r) × BAx
```

Where α is a constant (often set equal to r).

---

## Paper Statistics

### Parameter Reduction (Page 1, Abstract)

- **GPT-3 175B:** Reduces trainable parameters by **10,000×**
- **Memory:** Reduces GPU memory requirement by **3×**
- **Checkpoint size:** ~**35MB** (with LoRA) vs ~**350GB** (full model)

**Citation (Page 5, Section 4.2):**
> "With r = 4 and only the query and value projection matrices being adapted, the checkpoint size is reduced by roughly 10,000× (from 350GB to 35MB)"

### Optimal Rank Values (Page 10, Section 7.2, Table 6)

| Rank (r) | WikiSQL Accuracy | MultiNLI Accuracy |
|----------|------------------|-------------------|
| r = 1    | 68.8%           | 90.7%            |
| r = 2    | 69.6%           | 90.9%            |
| r = 4    | 70.5%           | 91.1%            |
| r = 8    | 70.4%           | 90.7%            |
| r = 64   | 70.0%           | 90.7%            |

**Key Finding (Page 10):**
> "LoRA already performs competitively with a very small r (more so for {Wq, Wv} than just Wq)"

### Which Layers to Adapt (Page 9, Section 7.1, Table 5)

Testing on GPT-3 175B with 18M trainable parameters:

| Weight Matrices  | WikiSQL | MultiNLI |
|-----------------|---------|----------|
| Wq only         | 70.4%   | 91.0%    |
| Wk only         | 70.0%   | 90.8%    |
| Wv only         | 73.0%   | 91.0%    |
| Wo only         | 73.2%   | 91.3%    |
| **Wq + Wv**     | **73.7%** | **91.3%** |
| Wq + Wk + Wv + Wo | 73.7% | 91.7%    |

**Conclusion (Page 9):**
> "Adapting both Wq and Wv yields the best result"

---

## Application to Diffusion Models

### Target Modules in Stable Diffusion

LoRA is applied to the **attention layers** in the UNet:

```python
target_modules = [
    "to_q",      # Query projection
    "to_k",      # Key projection  
    "to_v",      # Value projection
    "to_out.0"   # Output projection
]
```

**Source:** HuggingFace Diffusers documentation

### Typical Hyperparameters for Stable Diffusion

Based on diffusers library and community practice:

```python
lora_config = LoraConfig(
    r=4,                            # Rank: 4-8 for most tasks
    lora_alpha=4,                   # Usually equal to r
    init_lora_weights="gaussian",   # A~N(0,σ²), B=0
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    lora_dropout=0.0,               # Optional regularization
)
```

### Training Settings

```python
# Optimizer
optimizer = AdamW(
    lora_parameters,    # ONLY LoRA params
    lr=1e-4,           # Can use higher LR than full fine-tuning
    weight_decay=0.01
)

# Typical training
batch_size = 2-4            # Depends on GPU memory
num_epochs = 5-100          # Depends on dataset size
gradient_accumulation = 4   # If low GPU memory
```

**Citation (Page 18, Table 9):**
Learning rate of 2e-4 used for GPT-3 LoRA training.

---

## Memory and Efficiency

### Training Memory (Page 5, Section 5.1)

For GPT-2 Medium (354M parameters):

| Method | Trainable Params | Inference Latency |
|--------|------------------|-------------------|
| Full Fine-tune | 354M | Baseline |
| Adapter | 0.37M - 11M | +2% to +30% |
| LoRA | 0.35M | **No increase** |

**Key Advantage (Page 4, Section 4.1):**
> "Our simple linear design allows us to merge the trainable matrices with the frozen weights when deployed, introducing no inference latency"

### VRAM Reduction (Page 5, Section 4.2)

On GPT-3 175B:
- **Full fine-tuning:** 1.2TB VRAM needed
- **With LoRA:** 350GB VRAM needed
- **Reduction:** ~3× less memory

---

## Quick Code Snippets

### 1. Freeze Base Model

```python
# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False
```

**Citation (Page 4):** "W₀ is frozen and does not receive gradient updates"

### 2. Add LoRA Adapters

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=4,
    lora_alpha=4,
    target_modules=["to_q", "to_v"],
    init_lora_weights="gaussian"
)

model.add_adapter(lora_config)
```

### 3. Get Trainable Parameters

```python
lora_params = filter(lambda p: p.requires_grad, model.parameters())
```

**Citation (Page 4):** "A and B contain trainable parameters"

### 4. Save LoRA Weights

```python
from peft.utils import get_peft_model_state_dict

lora_state = get_peft_model_state_dict(model)
torch.save(lora_state, "lora_weights.pth")
```

### 5. Load LoRA Weights

```python
from peft.utils import set_peft_model_state_dict

lora_weights = torch.load("lora_weights.pth")
set_peft_model_state_dict(model, lora_weights)
```

### 6. Merge for Deployment

```python
# Merge LoRA weights into base model
for module in model.modules():
    if hasattr(module, 'merge_adapter'):
        module.merge_adapter()
```

**Citation (Page 4):** "We can explicitly compute and store W = W₀ + BA"

---

## Common Hyperparameter Values

### Rank (r)

| Value | Use Case |
|-------|----------|
| r = 1-2 | Very simple adaptations, minimal compute |
| **r = 4-8** | **Most common, good balance** |
| r = 16-32 | Complex adaptations, more parameters |
| r = 64+ | Approaching full fine-tuning, diminishing returns |

**Guidance (Page 10, Section 7.2):**
Start with r=4. Increase if underfitting, decrease if overfitting.

### Alpha (α)

Usually set **α = r** (most common practice).

The scaling factor α/r helps reduce need to retune hyperparameters when varying r.

### Learning Rate

- **Full fine-tuning:** 1e-5 to 5e-5
- **LoRA:** 1e-4 to 2e-4 (can use higher LR)

**Citation (Page 18, Table 9):** 2e-4 used for GPT-3

---

## When to Use LoRA

### ✅ Good Use Cases

1. **Limited compute resources** (consumer GPUs)
2. **Multiple task adaptations** (easy to swap)
3. **Frequent updates** (fast training)
4. **Storage constraints** (small file sizes)
5. **Style transfer** (new artistic styles)
6. **Character/object concepts** (DreamBooth-style)

### ❌ Not Ideal For

1. **Complete model retraining** (fundamentally different domain)
2. **When full model capacity needed** (very complex tasks)
3. **When inference speed not important** (full fine-tuning okay)

---

## Troubleshooting

### Low Performance

- **Try higher rank:** Increase r from 4 to 8 or 16
- **Add more modules:** Train text encoder too
- **More data:** LoRA works better with more examples
- **Higher learning rate:** Try 2e-4 instead of 1e-4

### Out of Memory

- **Lower rank:** Try r=2 or r=1
- **Reduce batch size:** Use gradient accumulation
- **Mixed precision:** Use torch.float16
- **Gradient checkpointing:** Enable in training

### Overfitting

- **Lower rank:** Use r=2 or r=4
- **Add dropout:** Set lora_dropout=0.1
- **More regularization:** Increase weight_decay
- **More data:** Get more training samples

---

## Complete Citations

### Primary Paper

**Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021).** LoRA: Low-Rank Adaptation of Large Language Models. *arXiv preprint arXiv:2106.09685*.

**Key Sections:**
- **Introduction (Page 1-2):** Motivation and overview
- **Method (Page 4, Section 4):** Mathematical formulation  
- **Experiments (Page 5-7, Section 5):** Empirical results
- **Analysis (Page 9-11, Section 7):** Understanding low-rank updates

### Implementation Resources

1. **Original LoRA:** https://github.com/microsoft/LoRA
2. **PEFT Library:** https://github.com/huggingface/peft
3. **Diffusers:** https://github.com/huggingface/diffusers
4. **Paper PDF:** https://arxiv.org/pdf/2106.09685.pdf

---

## One-Page Visual Summary

```
┌─────────────────────────────────────────────────────────┐
│                  LoRA Fine-Tuning                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Pre-trained Model: W₀ ∈ ℝᵈˣᵏ                          │
│       ↓                                                 │
│  [FROZEN - No gradients]                               │
│       +                                                 │
│  LoRA Adapters: ΔW = BA                                │
│       ↓                                                 │
│  [TRAINABLE - Only A,B updated]                        │
│       ↓                                                 │
│  Forward Pass: h = W₀x + BAx                           │
│       ↓                                                 │
│  Output (with adapted weights)                         │
│                                                         │
├─────────────────────────────────────────────────────────┤
│  Key Stats (Page 1, Hu et al. 2021):                  │
│  • 10,000× fewer trainable parameters                  │
│  • 3× less GPU memory                                  │
│  • No inference latency increase                       │
│  • 35MB vs 350GB checkpoint size                       │
└─────────────────────────────────────────────────────────┘
```

---

**Quick Start Command:**

```bash
python lora_complete_example.py
```

This will train a LoRA model, save weights, and generate a test image!
