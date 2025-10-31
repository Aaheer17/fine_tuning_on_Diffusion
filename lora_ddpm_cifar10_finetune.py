# lora_ddpm_cifar10_finetune.py
# LoRA fine-tuning of a pretrained DDPM UNet on a CIFAR-10 class subset.

import argparse, os, math, random, numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from diffusers import DiffusionPipeline, DDPMScheduler, UNet2DModel
from peft import LoraConfig

# --------------------------
# Utils
# --------------------------
def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def save_grid(tensor, path, nrow=8):
    """
    tensor: [N,C,H,W] in [-1,1]; saves an RGB grid image.
    """
    x = (tensor.add(1).mul(0.5)).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
    N, H, W, C = x.shape
    rows = (N + nrow - 1) // nrow
    canvas = np.ones((rows * H, nrow * W, 3), dtype=np.float32)
    idx = 0
    for r in range(rows):
        for c in range(nrow):
            if idx < N:
                img = x[idx]
                if img.shape[-1] == 1:
                    img = np.repeat(img, 3, axis=-1)
                canvas[r * H:(r + 1) * H, c * W:(c + 1) * W, :] = img
                idx += 1
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(nrow, rows))
    plt.axis("off"); plt.imshow(canvas, vmin=0, vmax=1); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()

@torch.no_grad()
def sample_ddpm(unet: UNet2DModel, scheduler: DDPMScheduler, n=32, shape=(3, 32, 32)):
    """
    Basic DDPM sampling loop with Diffusers scheduler.
    """
    unet.eval()
    x = torch.randn(n, *shape, device=DEVICE)
    scheduler.set_timesteps(scheduler.config.num_train_timesteps)  # e.g., 1000
    for t in scheduler.timesteps:
        noise_pred = unet(x, t).sample
        step_out = scheduler.step(noise_pred, t, x)
        x = step_out.prev_sample
    return x.clamp(-1, 1)

# --------------------------
# Data
# --------------------------
def make_cifar10_loader(data_root, classes, batch_size=256, num_workers=2, seed=0):
    """
    Returns DataLoader over CIFAR-10 training set filtered to 'classes' (list of ints 0..9).
    Images normalized to [-1, 1] to match DDPM training convention.
    """
    tx = transforms.Compose([
        transforms.ToTensor(),  # CIFAR10 is 32x32 already
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=tx)
    targets = np.array(ds.targets)
    idx = np.where(np.isin(targets, classes))[0]
    # keep order deterministic
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    ds_subset = Subset(ds, idx)
    loader = DataLoader(ds_subset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True, drop_last=True)
    return loader

# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--model_id", type=str, default="google/ddpm-cifar10-32",
                    help="Pretrained DDPM to adapt")
    ap.add_argument("--classes", type=int, nargs="+", default=[5,6,7,8,9],
                    help="CIFAR-10 class IDs to fine-tune on (default: 5..9)")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--train_steps", type=int, default=20_000,
                    help="Number of LoRA fine-tuning steps")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=float, default=8.0)
    ap.add_argument("--save_dir", type=str, default="./ckpts_lora")
    ap.add_argument("--sample_n", type=int, default=32)
    ap.add_argument("--log_interval", type=int, default=200)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # 1) Load pretrained DDPM pipeline (CIFAR10 32x32)
    pipe = DiffusionPipeline.from_pretrained(args.model_id)
    unet: UNet2DModel = pipe.unet.to(DEVICE)
    scheduler: DDPMScheduler = pipe.scheduler

    # 2) Save a pre-finetune sample grid (baseline)
    pre_samples = sample_ddpm(unet, scheduler, n=args.sample_n, shape=(3, 32, 32))
    save_grid(pre_samples, os.path.join(args.save_dir, "samples_pretrained.png"))
    print("Saved:", os.path.join(args.save_dir, "samples_pretrained.png"))

    # 3) Freeze base weights
    for p in unet.parameters():
        p.requires_grad_(False)

    # 4) Attach LoRA adapters to attention projections
    #    (to_q, to_k, to_v, to_out.0 are the common attn proj names in Diffusers UNet)
    lcfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet.add_adapter(lcfg)

    # 5) Optimizer over ONLY LoRA params
    trainable = [p for p in unet.parameters() if p.requires_grad]
    n_all = sum(p.numel() for p in unet.parameters())
    n_tr = sum(p.numel() for p in trainable)
    print(f"Trainable params (LoRA only): {n_tr:,} / {n_all:,} "
          f"({100.0 * n_tr / n_all:.2f}%)")
    optim = torch.optim.AdamW(trainable, lr=args.lr)

    # 6) Data loader over a class subset (domain shift)
    loader = make_cifar10_loader(
        data_root=args.data_root,
        classes=args.classes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    )

    # 7) Training loop (standard noise-prediction MSE)
    unet.train()
    scheduler.set_timesteps(scheduler.config.num_train_timesteps)  # ensure config is loaded
    step = 0
    while step < args.train_steps:
        for x0, _ in loader:
            x0 = x0.to(DEVICE)  # [-1, 1], [B,3,32,32]
            b = x0.size(0)
            t = torch.randint(0, scheduler.config.num_train_timesteps,
                              (b,), device=DEVICE, dtype=torch.long)
            noise = torch.randn_like(x0)
            x_t = scheduler.add_noise(x0, noise, t)

            pred = unet(x_t, t).sample
            loss = F.mse_loss(pred, noise)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optim.step()

            step += 1
            if step % args.log_interval == 0:
                print(f"[step {step:6d}] loss={loss.item():.4f}")

            if step >= args.train_steps:
                break

    # 8) Save LoRA adapters only (tiny)
    unet.save_pretrained(args.save_dir, save_adapter=True)
    print("Saved LoRA adapters to:", args.save_dir)

    # 9) Post-finetune samples
    post_samples = sample_ddpm(unet, scheduler, n=args.sample_n, shape=(3, 32, 32))
    save_grid(post_samples, os.path.join(args.save_dir, "samples_finetuned_lora.png"))
    print("Saved:", os.path.join(args.save_dir, "samples_finetuned_lora.png"))

if __name__ == "__main__":
    main()
