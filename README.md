# LoRA Fine-Tuning for Diffusion Models - Complete Learning Package

Welcome! This package contains everything you need to learn and implement LoRA (Low-Rank Adaptation) fine-tuning for diffusion models.

## 📚 What You Have

### 1. **diffusion_lora_finetuning_tutorial.md** (21 KB)
**The Complete Tutorial** - Start here!

This comprehensive guide includes:
- ✅ Introduction to LoRA and why it's important
- ✅ Theoretical background with mathematical formulas
- ✅ Paper citations with specific page and line numbers
- ✅ Step-by-step code explanations
- ✅ How to freeze layers and add adapters
- ✅ Complete working examples
- ✅ Advanced topics (rank selection, module choices, merging weights)

**Who it's for:** Anyone wanting to understand LoRA deeply before implementing it.

**Time needed:** 45-60 minutes to read thoroughly

---

### 2. **lora_complete_example.py** (14 KB)
**Ready-to-Run Python Script**

A fully commented, executable Python script that demonstrates:
- ✅ Loading and freezing a base Stable Diffusion model
- ✅ Adding LoRA adapters to attention layers
- ✅ Training only the low-rank matrices
- ✅ Saving lightweight LoRA weights (~3MB)
- ✅ Loading weights and generating images
- ✅ Every major step has paper citations

**Who it's for:** Hands-on learners who want to see working code immediately.

**Time needed:** 5-10 minutes to run (varies by hardware)

---

### 3. **lora_quick_reference.md** (11 KB)
**Cheat Sheet & Quick Reference**

A condensed reference guide with:
- ✅ Key formulas and equations
- ✅ Paper statistics (10,000× reduction, 3× memory savings)
- ✅ Optimal hyperparameter values
- ✅ Common troubleshooting tips
- ✅ Quick code snippets
- ✅ All with specific paper citations

**Who it's for:** Quick lookup when you need specific information.

**Time needed:** 10 minutes to skim, keep handy while coding

---

## 🎯 Quick Start Guide

### For Absolute Beginners

**Path 1: Theory First**
1. Read `diffusion_lora_finetuning_tutorial.md` sections 1-3
2. Run `lora_complete_example.py` 
3. Read sections 4-6 of the tutorial while referring to the code
4. Keep `lora_quick_reference.md` open for quick lookups

**Path 2: Code First** (for experienced practitioners)
1. Skim `lora_quick_reference.md` (10 min)
2. Run `lora_complete_example.py` immediately
3. Read `diffusion_lora_finetuning_tutorial.md` for deeper understanding
4. Experiment with different parameters

---

## 💻 Running the Example

### Prerequisites

```bash
# Install required packages
pip install diffusers transformers accelerate peft
pip install torch torchvision
pip install datasets tqdm
```

### Run the Script

```bash
# Basic execution
python lora_complete_example.py

# On GPU (recommended)
CUDA_VISIBLE_DEVICES=0 python lora_complete_example.py

# With custom settings (edit the script)
# Change LORA_RANK, LEARNING_RATE, NUM_EPOCHS at the top
```

### Expected Outputs

After running, you'll have:
- `my_lora_model/lora_weights.pth` - Your trained LoRA weights (~3-5 MB)
- `my_lora_model/test_generation.png` - A generated test image

---

## 📖 Key Concepts (Quick Summary)

### What is LoRA?

Instead of updating ALL parameters in a huge model (billions of params), LoRA:
1. **Freezes** the original weights (W₀)
2. **Adds** small trainable matrices (A and B) 
3. **Combines** them: output = W₀x + BAx

**Result:** Train 0.1% of parameters, get 99% of full fine-tuning performance!

### The Math (Simplified)

```
Original:    h = Wx
With LoRA:   h = W₀x + BAx
             ↑      ↑
         Frozen  Trainable
```

Where:
- W₀: Pre-trained weights (e.g., 859M parameters) - FROZEN
- B, A: Low-rank matrices (e.g., 1.5M parameters) - TRAINABLE
- r (rank): How "big" the low-rank matrices are (typically 4-16)

### Why It Works

**Paper Citation:** Hu et al. (2021), Page 2
> "We hypothesize that the change in weights during model adaptation also has a low 'intrinsic rank'"

Translation: You don't need to change all parameters equally. Most of the adaptation can happen in a much smaller subspace.

---

## 🔬 Paper Citations Used

**Primary Reference:**
- **Hu, E.J., et al. (2021).** "LoRA: Low-Rank Adaptation of Large Language Models"
- **arXiv:** 2106.09685
- **URL:** https://arxiv.org/pdf/2106.09685.pdf

**Key Sections Cited:**
- **Page 1 (Abstract):** Parameter reduction statistics
- **Page 4, Section 4.1, Equation 3:** Core mathematical formulation
- **Page 4, Section 4.2:** Application to Transformers
- **Page 5, Section 4.2:** VRAM and checkpoint size reductions
- **Page 9, Section 7.1, Table 5:** Which weight matrices to adapt
- **Page 10, Section 7.2, Table 6:** Optimal rank values
- **Page 18, Table 9:** Training hyperparameters

**For Diffusion Models:**
- HuggingFace blog: "Using LoRA for Efficient Stable Diffusion Fine-Tuning"
- URL: https://huggingface.co/blog/lora

---

## 🎨 Example Use Cases

### 1. Style Transfer
Train LoRA to generate images in a specific artistic style:
- Anime style
- Oil painting style
- Sketch style

**Dataset needed:** 10-100 images of target style

### 2. Character/Object Concepts
Teach the model new concepts:
- Your pet
- A specific character design  
- Custom objects

**Dataset needed:** 5-20 images of the subject

### 3. Domain Adaptation
Adapt to specific domains:
- Medical images
- Satellite imagery
- Architecture

**Dataset needed:** 100-1000 images

---

## ⚙️ Hyperparameter Guide

### Start With These Values

```python
LORA_RANK = 4            # Good balance (try 2, 4, 8, 16)
LORA_ALPHA = 4           # Usually equal to rank
LEARNING_RATE = 1e-4     # Can go higher with LoRA
BATCH_SIZE = 2-4         # Depends on GPU memory
NUM_EPOCHS = 5-100       # Depends on dataset size
```

### If You See Issues

**Underfitting (loss not decreasing enough):**
- Increase `LORA_RANK` (4 → 8 → 16)
- Increase `LEARNING_RATE` (1e-4 → 2e-4)
- Train longer (`NUM_EPOCHS`)

**Overfitting (training loss low, but outputs bad):**
- Decrease `LORA_RANK` (8 → 4 → 2)
- Add more data
- Add dropout: `lora_dropout=0.1`

**Out of Memory:**
- Decrease `BATCH_SIZE` (4 → 2 → 1)
- Decrease `LORA_RANK` (8 → 4)
- Use gradient accumulation
- Enable mixed precision training

---

## 🚀 Next Steps After This Tutorial

### 1. Try With Real Datasets

```python
# Naruto characters
dataset = load_dataset("lambdalabs/naruto-blip-captions")

# Pokemon
dataset = load_dataset("lambdalabs/pokemon-blip-captions")

# Your own images
# Put images in a folder and write a custom dataset class
```

### 2. Train Text Encoder Too

Add LoRA to the text encoder for better prompt understanding:

```python
from transformers import CLIPTextModel

text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder")
text_encoder.add_adapter(text_encoder_lora_config)
```

### 3. Combine Multiple LoRAs

Load multiple trained LoRAs and blend them:

```python
pipeline.load_lora_weights("style_lora.safetensors", adapter_name="style")
pipeline.load_lora_weights("character_lora.safetensors", adapter_name="character")
pipeline.set_adapters(["style", "character"], weights=[0.7, 0.3])
```

### 4. Explore Advanced Techniques

- **DreamBooth + LoRA:** Better for specific subjects
- **LoRA+:** Improved version with separate learning rates for A and B
- **DoRA:** Combines LoRA with other techniques
- **Pivotal Tuning:** Combines Textual Inversion with LoRA

---

## 📊 Expected Results

After training LoRA on ~50-100 images for 5-10 epochs:

**Parameter Count:**
- Total model parameters: ~860M
- Trainable (LoRA): ~1.5M (0.17%)
- Reduction: ~550×

**File Sizes:**
- Full model checkpoint: ~4 GB
- LoRA weights only: ~3-5 MB
- Reduction: ~1000×

**Training Time:**
- On RTX 3090: ~2-3 hours for 100 epochs
- On RTX 2080 Ti: ~4-5 hours for 100 epochs
- On consumer GPUs: 6-10 hours for 100 epochs

**Memory Usage:**
- Full fine-tuning: 20-24 GB VRAM
- With LoRA: 8-12 GB VRAM
- Savings: ~50%

---

## 🆘 Troubleshooting

### Script Won't Run

**Import Errors:**
```bash
pip install --upgrade diffusers transformers peft
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CUDA Out of Memory:**
```python
# In script, reduce:
BATCH_SIZE = 1
LORA_RANK = 2
```

**Model Download Issues:**
```bash
# Login to HuggingFace
huggingface-cli login

# Or set token
export HF_TOKEN="your_token_here"
```

### Training Issues

**Loss not decreasing:**
- Check learning rate (try 2e-4)
- Verify data loading works
- Ensure LoRA adapters are actually trainable

**NaN loss:**
- Lower learning rate (try 5e-5)
- Check for corrupted images in dataset
- Enable gradient clipping

**Poor quality outputs:**
- Train longer (more epochs)
- Use more training data
- Increase rank (r=4 → r=8)

---

## 💡 Tips for Success

1. **Start small:** Use a subset of data first (50 images)
2. **Monitor loss:** It should decrease steadily
3. **Test frequently:** Generate images every few epochs
4. **Compare:** Save outputs to see progress
5. **Be patient:** Good results need 50-200 training steps minimum
6. **Experiment:** Try different ranks and learning rates
7. **Use citations:** Reference the paper in your work!

---

## 📚 Additional Resources

### Official Implementations
- Original LoRA: https://github.com/microsoft/LoRA
- PEFT library: https://github.com/huggingface/peft
- Diffusers: https://github.com/huggingface/diffusers

### Community Resources
- Civitai: https://civitai.com (1000s of trained LoRAs)
- r/StableDiffusion: https://reddit.com/r/StableDiffusion
- LoRA training discussions: HuggingFace forums

### Papers to Read Next
- DreamBooth: "DreamBooth: Fine Tuning Text-to-Image Diffusion Models"
- Textual Inversion: "An Image is Worth One Word"
- LoRA+: "LoRA+: Efficient Low Rank Adaptation of Large Models"

---

## 🎓 Learning Objectives Checklist

After completing this tutorial, you should be able to:

- [ ] Explain what LoRA is and why it's useful
- [ ] Describe the mathematical formulation (W = W₀ + BA)
- [ ] Identify which parameters are frozen vs. trainable
- [ ] Configure LoRA hyperparameters (rank, alpha, target modules)
- [ ] Write code to add LoRA adapters to a model
- [ ] Train a LoRA-adapted model on custom data
- [ ] Save and load LoRA weights
- [ ] Generate images using a LoRA-adapted model
- [ ] Troubleshoot common issues
- [ ] Cite the original paper correctly

---

## ✅ Verification Steps

Run these to verify your understanding:

```bash
# 1. Can you run the example?
python lora_complete_example.py

# 2. Can you modify the rank?
# Edit script: LORA_RANK = 8
python lora_complete_example.py

# 3. Can you change the target modules?
# Edit script: TARGET_MODULES = ["to_q", "to_v", "to_out.0"]
python lora_complete_example.py
```

---

## 📝 Assignment Ideas

To solidify your learning:

1. **Compare ranks:** Train with r=2, 4, 8 and compare outputs
2. **Ablation study:** Try different target modules
3. **Real dataset:** Train on Naruto or Pokemon dataset
4. **Multi-LoRA:** Train two LoRAs and combine them
5. **Documentation:** Write a blog post explaining LoRA

---

## 🙏 Acknowledgments

This tutorial is based on:
- **Hu et al. (2021)** - Original LoRA paper
- **HuggingFace** - Diffusers and PEFT libraries  
- **Community** - Countless tutorials and experiments

---

## 📧 Questions?

If you're stuck:
1. Re-read relevant sections
2. Check the Quick Reference guide
3. Search for error messages
4. Ask on HuggingFace forums
5. Check GitHub issues

---

**Happy Learning! 🚀**

Remember: LoRA is about making AI more accessible. You're training models that once required data center resources on your own GPU. That's pretty amazing!

*Last updated: October 2025*
