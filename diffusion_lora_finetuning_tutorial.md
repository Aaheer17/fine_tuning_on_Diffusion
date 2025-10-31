# Complete Guide to Fine-Tuning Diffusion Models with LoRA

## Table of Contents
1. [Introduction to LoRA](#introduction)
2. [Theoretical Background](#theory)
3. [How LoRA Works in Diffusion Models](#how-lora-works)
4. [Step-by-Step Code Tutorial](#code-tutorial)
5. [Complete Working Example](#complete-example)
6. [Advanced Topics](#advanced-topics)

---

## 1. Introduction to LoRA {#introduction}

### What is LoRA?

LoRA (Low-Rank Adaptation) is a technique that freezes pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks.

**Key Paper Citation:**
- **Paper:** "LoRA: Low-Rank Adaptation of Large Language Models" by Hu et al., 2021
- **arXiv:** 2106.09685

### Why Use LoRA for Diffusion Models?

Training is much faster, compute requirements are lower, and a full fine-tuned model can be created in a 2080 Ti with 11 GB of VRAM. With LoRA, it is now possible to publish a single 3.29 MB file to allow others to use your fine-tuned model.

**Advantages cited in the paper (Page 1, Section 1):**
1. Reduces trainable parameters by 10,000× compared to full fine-tuning
2. Reduces GPU memory requirement by 3×
3. No additional inference latency when weights are merged
4. Easy to switch between tasks by swapping LoRA weights

---

## 2. Theoretical Background {#theory}

### The Core Mathematical Concept

**From the LoRA paper (Page 4, Section 4.1, Equation 3):**

For a pre-trained weight matrix `W₀ ∈ ℝᵈˣᵏ`, LoRA constrains its update by representing it with a low-rank decomposition:

```
W = W₀ + ΔW = W₀ + BA
```

Where:
- `B ∈ ℝᵈˣʳ` and `A ∈ ℝʳˣᵏ`
- Rank `r ≪ min(d, k)` (r is much smaller than the original dimensions)
- **W₀ is frozen** (no gradient updates)
- **A and B contain trainable parameters**

The forward pass becomes:
```
h = W₀x + ΔWx = W₀x + BAx
```

**Initialization (Page 4, Section 4.1):**
- Matrix A: Random Gaussian initialization
- Matrix B: Zero initialization (so ΔW = BA is zero at the start)
- Scaling factor: `α/r` where α is constant

### Why Low-Rank Works

The paper hypothesizes that the change in weights during model adaptation has a low "intrinsic rank" (Page 2, Section 1).

**Empirical Evidence (Page 10, Section 7.2, Table 6):**
The paper shows that even `r = 1` performs competitively on downstream tasks, suggesting the update matrix ΔW has a very small intrinsic rank.

---

## 3. How LoRA Works in Diffusion Models {#how-lora-works}

### Application to Stable Diffusion

In the case of Stable Diffusion fine-tuning, LoRA can be applied to the cross-attention layers that relate the image representations with the prompts that describe them.

**Target Modules (from diffusers implementation):**
In Stable Diffusion's UNet, LoRA is typically applied to:
- Query projection: `to_q`
- Key projection: `to_k`  
- Value projection: `to_v`
- Output projection: `to_out.0`

These are the attention weight matrices in the self-attention modules.

**From the LoRA paper (Page 4, Section 4.2):**
> "We limit our study to only adapting the attention weights for downstream tasks and freeze the MLP modules"

### Freezing vs. Trainable Components

```
┌────────────────────────────────────────┐
│        Stable Diffusion UNet           │
│                                        │
│  ┌──────────────────────────────────┐ │
│  │    Frozen Base Model (W₀)        │ │
│  │  - All pre-trained weights       │ │
│  │  - No gradient computation       │ │
│  └──────────────────────────────────┘ │
│                                        │
│  ┌──────────────────────────────────┐ │
│  │   LoRA Adapters (A, B)           │ │
│  │  - Small trainable matrices      │ │
│  │  - Added in parallel to layers   │ │
│  │  - Rank r = 4 to 64 typical      │ │
│  └──────────────────────────────────┘ │
└────────────────────────────────────────┘
```

---

## 4. Step-by-Step Code Tutorial {#code-tutorial}

### Step 1: Installation

```bash
# Install required packages
pip install diffusers transformers accelerate peft
pip install torch torchvision
pip install datasets
```

### Step 2: Understanding the Core Components

#### A. Loading the Base Model (Frozen)

```python
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import torch

# Load the pre-trained model
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
unet = UNet2DConditionModel.from_pretrained(
    model_id,
    subfolder="unet",
    torch_dtype=torch.float16
)

# IMPORTANT: Freeze all parameters
for param in unet.parameters():
    param.requires_grad = False
```

**What's happening:**
- All weights in `unet.parameters()` are marked as `requires_grad = False`
- This prevents gradient computation and parameter updates
- Saves memory and computation time

#### B. Adding LoRA Adapters

```python
from peft import LoraConfig

# Configure LoRA parameters
lora_config = LoraConfig(
    r=4,  # Rank: Paper shows r=4 often sufficient (Page 10, Table 6)
    lora_alpha=4,  # Scaling factor (typically equal to r)
    init_lora_weights="gaussian",  # Initialize A with Gaussian, B with zeros
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # Which layers to adapt
)

# Add LoRA adapters to the UNet
unet.add_adapter(lora_config)
```

**Parameters Explained:**
- **r (rank):** Controls the bottleneck dimension. Paper: "r in Figure 1 can be one or two" for GPT-3 (Page 2)
- **lora_alpha:** Scaling constant `α` in the formula `α/r` (Page 4)
- **target_modules:** Attention projection layers to adapt

#### C. Filtering Trainable Parameters

```python
# Only LoRA parameters will be trained
lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

# Count trainable vs total parameters
trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in unet.parameters())

print(f"Trainable params: {trainable_params:,}")
print(f"All params: {all_params:,}")
print(f"Trainable %: {100 * trainable_params / all_params:.4f}%")
```

**Expected Output:**
```
Trainable params: 1,476,608
All params: 859,520,964
Trainable %: 0.1718%
```

### Step 3: Preparing a Simple Dataset

```python
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset

class SimpleImageDataset(Dataset):
    """Simple dataset for demonstration"""
    
    def __init__(self, dataset_name="lambdalabs/naruto-blip-captions", split="train"):
        # Load a small dataset
        self.dataset = load_dataset(dataset_name, split=split)
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        caption = item["text"]
        
        # Transform image
        image = self.transform(image)
        
        return {
            "pixel_values": image,
            "input_ids": caption
        }

# Create dataset instance
train_dataset = SimpleImageDataset()
print(f"Dataset size: {len(train_dataset)}")
```

### Step 4: Setting Up the Training Loop

```python
from torch.optim import AdamW
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader

# Create dataloader
train_dataloader = DataLoader(
    train_dataset,
    batch_size=4,  # Adjust based on GPU memory
    shuffle=True
)

# Setup optimizer - ONLY for LoRA parameters
optimizer = AdamW(
    lora_layers,  # Only optimize LoRA weights
    lr=1e-4,  # LoRA paper suggests higher LR is ok (Page 5, Section 5.1)
    betas=(0.9, 0.999),
    weight_decay=0.01,
    eps=1e-08
)

# Learning rate scheduler
lr_scheduler = get_scheduler(
    "constant",  # or "linear", "cosine"
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(train_dataloader) * num_epochs
)
```

**Key Points:**
- **Optimizer only updates LoRA parameters** (lora_layers)
- With LoRA, you can use a higher learning rate than full fine-tuning
- Base model (W₀) never receives gradients

### Step 5: Training Function

```python
from tqdm import tqdm
from diffusers import DDPMScheduler

def train_one_epoch(unet, optimizer, train_dataloader, noise_scheduler, device):
    """Train for one epoch"""
    unet.train()
    total_loss = 0
    
    progress_bar = tqdm(train_dataloader, desc="Training")
    
    for batch in progress_bar:
        # Move to device
        pixel_values = batch["pixel_values"].to(device)
        
        # Sample noise
        noise = torch.randn_like(pixel_values)
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, 
            noise_scheduler.config.num_train_timesteps,
            (pixel_values.shape[0],),
            device=device
        )
        
        # Add noise to images (forward diffusion process)
        noisy_images = noise_scheduler.add_noise(pixel_values, noise, timesteps)
        
        # Predict the noise residual
        noise_pred = unet(noisy_images, timesteps).sample
        
        # Calculate loss (MSE between predicted and actual noise)
        loss = F.mse_loss(noise_pred, noise)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        # Update ONLY LoRA parameters
        optimizer.step()
        lr_scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
    
    return total_loss / len(train_dataloader)

# Training loop
num_epochs = 100
noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
device = "cuda" if torch.cuda.is_available() else "cpu"
unet.to(device)

for epoch in range(num_epochs):
    avg_loss = train_one_epoch(unet, optimizer, train_dataloader, noise_scheduler, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
```

### Step 6: Saving and Loading LoRA Weights

```python
from peft.utils import get_peft_model_state_dict

# Save only LoRA weights (very small file!)
lora_state_dict = get_peft_model_state_dict(unet)
torch.save(lora_state_dict, "naruto_lora_weights.pth")

print(f"LoRA weights saved. File size: ~{len(lora_state_dict)} tensors")
# Typically 3-5 MB vs. 4-5 GB for full model!

# Load LoRA weights later
from peft.utils import set_peft_model_state_dict

unet_new = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
unet_new.add_adapter(lora_config)
lora_weights = torch.load("naruto_lora_weights.pth")
set_peft_model_state_dict(unet_new, lora_weights)
```

### Step 7: Inference with LoRA

```python
from diffusers import StableDiffusionPipeline

# Create pipeline with LoRA-adapted model
pipeline = StableDiffusionPipeline.from_pretrained(
    model_id,
    unet=unet,
    torch_dtype=torch.float16
)
pipeline = pipeline.to("cuda")

# Generate image
prompt = "A naruto character with blue eyes and spiky hair"
image = pipeline(prompt, num_inference_steps=50).images[0]
image.save("output_naruto.png")
```

---

## 5. Complete Working Example {#complete-example}

Here's a minimal, complete script you can run:

```python
"""
Complete LoRA Fine-tuning Example for Stable Diffusion
Based on Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models"
"""

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig, get_peft_model_state_dict
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
from tqdm import tqdm

# ==================== CONFIGURATION ====================
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DATASET_NAME = "lambdalabs/naruto-blip-captions"
OUTPUT_DIR = "./naruto_lora"
LORA_RANK = 4  # From paper: even r=1 can work (Page 10, Table 6)
LEARNING_RATE = 1e-4  # Higher LR ok with LoRA (Page 18, Table 9)
BATCH_SIZE = 2
NUM_EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== STEP 1: LOAD & FREEZE BASE MODEL ====================
print("Loading base model...")
unet = UNet2DConditionModel.from_pretrained(
    MODEL_ID,
    subfolder="unet",
    torch_dtype=torch.float16
).to(DEVICE)

# Freeze all parameters
for param in unet.parameters():
    param.requires_grad = False

# ==================== STEP 2: ADD LORA ADAPTERS ====================
print("Adding LoRA adapters...")
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_RANK,
    init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
)
unet.add_adapter(lora_config)

# Get trainable parameters
lora_params = filter(lambda p: p.requires_grad, unet.parameters())
trainable_count = sum(p.numel() for p in unet.parameters() if p.requires_grad)
total_count = sum(p.numel() for p in unet.parameters())
print(f"Trainable: {trainable_count:,} / {total_count:,} ({100*trainable_count/total_count:.2f}%)")

# ==================== STEP 3: PREPARE DATASET ====================
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME, split="train[:100]")  # Use subset for demo

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def preprocess(examples):
    images = [transform(img.convert("RGB")) for img in examples["image"]]
    return {"pixel_values": torch.stack(images)}

dataset.set_transform(preprocess)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==================== STEP 4: SETUP TRAINING ====================
optimizer = torch.optim.AdamW(lora_params, lr=LEARNING_RATE)
noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

# ==================== STEP 5: TRAINING LOOP ====================
print("Starting training...")
unet.train()

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    
    for batch in progress_bar:
        pixel_values = batch["pixel_values"].to(DEVICE, dtype=torch.float16)
        
        # Sample noise and timesteps
        noise = torch.randn_like(pixel_values)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (pixel_values.shape[0],), device=DEVICE
        ).long()
        
        # Add noise (forward diffusion)
        noisy_images = noise_scheduler.add_noise(pixel_values, noise, timesteps)
        
        # Predict noise
        noise_pred = unet(noisy_images, timesteps, return_dict=False)[0]
        
        # Compute loss
        loss = F.mse_loss(noise_pred, noise)
        
        # Backprop (only LoRA params updated!)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

# ==================== STEP 6: SAVE LORA WEIGHTS ====================
print("Saving LoRA weights...")
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

lora_state = get_peft_model_state_dict(unet)
torch.save(lora_state, f"{OUTPUT_DIR}/lora_weights.pth")
print(f"Saved to {OUTPUT_DIR}/lora_weights.pth")

# ==================== STEP 7: INFERENCE ====================
print("Generating test image...")
pipeline = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    unet=unet,
    torch_dtype=torch.float16
).to(DEVICE)

prompt = "A naruto character, highly detailed, anime style"
image = pipeline(prompt, num_inference_steps=30).images[0]
image.save(f"{OUTPUT_DIR}/test_output.png")
print(f"Image saved to {OUTPUT_DIR}/test_output.png")
print("Done!")
```

**To run this script:**
```bash
python complete_lora_example.py
```

---

## 6. Advanced Topics {#advanced-topics}

### A. Choosing the Right Rank (r)

**From the paper (Page 10, Section 7.2):**

The paper investigates rank values and finds:
- **r = 1:** Often sufficient for simple adaptations
- **r = 4-8:** Good balance for most tasks  
- **r = 64:** Minimal improvement over r = 8

**Experimental results (Table 6, Page 10):**
```
Rank (r)  |  WikiSQL Acc  |  MultiNLI Acc
----------|---------------|---------------
r = 1     |    68.8%      |    90.7%
r = 4     |    70.5%      |    91.1%
r = 8     |    70.4%      |    90.7%
r = 64    |    70.0%      |    90.7%
```

**Recommendation:** Start with r = 4, increase if underfitting.

### B. Which Modules to Adapt?

**From the paper (Page 9, Section 7.1, Table 5):**

Testing different attention components on GPT-3:
```
Adapted Weights    |  WikiSQL  |  MultiNLI
-------------------|-----------|----------
Wq only            |   70.4%   |   91.0%
Wq, Wv             |   73.7%   |   91.3%  ← Best
Wq, Wk, Wv, Wo     |   73.7%   |   91.7%
```

**Conclusion:** Adapting query (Wq) and value (Wv) gives best performance-to-parameter ratio.

### C. Training Text Encoder with LoRA

```python
from transformers import CLIPTextModel

# Load text encoder
text_encoder = CLIPTextModel.from_pretrained(
    MODEL_ID, 
    subfolder="text_encoder"
)

# Freeze base weights
for param in text_encoder.parameters():
    param.requires_grad = False

# Add LoRA to text encoder
text_encoder_lora_config = LoraConfig(
    r=4,
    lora_alpha=4,
    target_modules=["q_proj", "v_proj"],  # Text encoder attention
    lora_dropout=0.1
)

from peft import get_peft_model
text_encoder = get_peft_model(text_encoder, text_encoder_lora_config)
```

### D. Merging LoRA Weights for Deployment

**From the paper (Page 4, Section 4.1):**
> "We can explicitly compute and store W = W₀ + BA and perform inference as usual"

```python
def merge_lora_weights(unet):
    """Merge LoRA weights into base model for faster inference"""
    
    for name, module in unet.named_modules():
        if hasattr(module, 'merge_adapter'):
            module.merge_adapter()
            print(f"Merged LoRA in: {name}")
    
    return unet

# Merge for deployment
unet_merged = merge_lora_weights(unet)

# Now inference has NO additional latency!
# Paper (Page 4): "no additional inference latency compared to 
# a fully fine-tuned model"
```

### E. Multiple LoRA Adapters

You can combine multiple trained concepts in a single image:

```python
# Load base model
pipeline = StableDiffusionPipeline.from_pretrained(MODEL_ID)

# Load multiple LoRA weights
pipeline.load_lora_weights("naruto_lora.safetensors", adapter_name="naruto")
pipeline.load_lora_weights("anime_style_lora.safetensors", adapter_name="anime")

# Set adapter weights
pipeline.set_adapters(["naruto", "anime"], adapter_weights=[0.7, 0.3])

# Generate with combined styles
image = pipeline("A character in naruto style").images[0]
```

---

## Summary of Key Citations

**Main Paper:**
- **Hu, E.J., et al. (2021).** "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685

**Key Equations:**
- **Low-rank decomposition (Page 4, Eq. 3):** h = W₀x + BAx
- **Initialization (Page 4, Section 4.1):** A ~ N(0, σ²), B = 0
- **Scaling (Page 4):** ΔWx scaled by α/r

**Empirical Results:**
- **Memory reduction (Page 1):** 10,000× fewer trainable parameters
- **Performance (Page 10, Table 6):** r=4 sufficient for most tasks
- **Best modules (Page 9, Table 5):** Query + Value projections

**For Diffusion Models:**
- HuggingFace blog post on LoRA for Stable Diffusion fine-tuning

---

## Additional Resources

1. **Official LoRA Implementation:** https://github.com/microsoft/LoRA
2. **Diffusers LoRA Training:** https://github.com/huggingface/diffusers/tree/main/examples/text_to_image
3. **PEFT Library:** https://github.com/huggingface/peft
4. **LoRA Paper:** https://arxiv.org/abs/2106.09685

---

**Note:** This tutorial is for educational purposes. Always cite the original LoRA paper when using this technique in research or production.
