# ddpm_full_finetune_real.py
# Full fine-tuning DDPM on real torchvision datasets (MNIST/FashionMNIST/CIFAR10)
# Usage examples:
#   python ddpm_full_finetune_real.py --dataset mnist --pretrain-classes 0 1 2 3 4 --finetune-classes 5 6 7 8 9
#   python ddpm_full_finetune_real.py --dataset cifar10 --pretrain-classes 0 1 2 3 4 --finetune-classes 5 6 7 8 9

import os, math, argparse, random, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# -------------------------------
# Utilities
# -------------------------------
def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def save_grid(tensor, path, nrow=8):
    """
    tensor: [N,C,H,W] in [-1,1]
    Saves a grid as RGB. If C==1, repeats channels to 3.
    """
    x = (tensor.add(1).mul(0.5)).clamp(0,1).cpu().permute(0,2,3,1).numpy()  # [N,H,W,C]
    N, H, W, C = x.shape

    rows = (N + nrow - 1) // nrow
    canvas = np.ones((rows*H, nrow*W, 3), dtype=np.float32)  # always 3-channel

    idx = 0
    for r in range(rows):
        for c in range(nrow):
            if idx < N:
                img = x[idx]
                if img.shape[-1] == 1:
                    img = np.repeat(img, 3, axis=-1)  # [H,W,3]
                canvas[r*H:(r+1)*H, c*W:(c+1)*W, :] = img
                idx += 1

    import matplotlib.pyplot as plt
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(nrow, rows))
    plt.axis("off")
    plt.imshow(canvas, vmin=0, vmax=1)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# -------------------------------
# Simple UNet with time conditioning
# -------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch), nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch), nn.SiLU(),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.lin1 = nn.Linear(dim, dim*4)
        self.lin2 = nn.Linear(dim*4, dim)
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device).float() / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        emb = self.lin2(F.silu(self.lin1(emb)))
        return emb

class UNetSmall(nn.Module):
    def __init__(self, in_ch=1, base=64, time_dim=128):
        super().__init__()
        self.time_mlp = TimeEmbedding(time_dim)
        self.inc   = DoubleConv(in_ch, base)
        self.down1 = Down(base, base*2)
        self.down2 = Down(base*2, base*4)
        self.bot   = DoubleConv(base*4, base*4)
        self.up2   = Up(base*4 + base*2, base*2)
        self.up1   = Up(base*2 + base, base)
        self.outc  = nn.Conv2d(base, in_ch, 1)
        # FiLM per stage
        self.s1, self.b1 = nn.Linear(time_dim, base),     nn.Linear(time_dim, base)
        self.s2, self.b2 = nn.Linear(time_dim, base*2),   nn.Linear(time_dim, base*2)
        self.s3, self.b3 = nn.Linear(time_dim, base*4),   nn.Linear(time_dim, base*4)

    def film(self, x, s, b):
        return x * (1 + s[..., None, None]) + b[..., None, None]

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x1 = self.inc(x)         # [B, b, 32, 32]
        x2 = self.down1(x1)      # [B, 2b, 16, 16]
        x3 = self.down2(x2)      # [B, 4b, 8, 8]
        xb = self.bot(x3)        # [B, 4b, 8, 8]
        # time FiLM
        x1 = self.film(x1, self.s1(t_emb), self.b1(t_emb))
        x2 = self.film(x2, self.s2(t_emb), self.b2(t_emb))
        xb = self.film(xb, self.s3(t_emb), self.b3(t_emb))
        u2 = self.up2(xb, x2)    # [B, 2b, 16, 16]
        u1 = self.up1(u2, x1)    # [B, b, 32, 32]
        return self.outc(u1)     # predict noise

# -------------------------------
# DDPM core
# -------------------------------
class DDPM:
    def __init__(self, timesteps=200, beta_start=1e-4, beta_end=2e-2, device=DEVICE):
        self.T = timesteps
        self.device = device
        beta = torch.linspace(beta_start, beta_end, self.T, device=device)
        alpha = 1.0 - beta
        self.register(alpha, beta)

    def register(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.alphabar = torch.cumprod(alpha, dim=0)                       # [T]
        self.sqrt_ab = torch.sqrt(self.alphabar)                           # [T]
        self.sqrt_1mab = torch.sqrt(torch.clamp(1.0 - self.alphabar, 1e-20))
        self.one_over_sqrt_a = torch.sqrt(1.0 / self.alpha)                # [T]
        # posterior variance for t >= 1 (length T-1)
        self.posterior_var = self.beta[1:] * (1.0 - self.alphabar[:-1]) / torch.clamp(1.0 - self.alphabar[1:], 1e-20)

    def q_sample(self, x0, t, noise=None):
        if noise is None: noise = torch.randn_like(x0)
        sqrt_ab   = self.sqrt_ab[t].view(-1,1,1,1)
        sqrt_1mab = self.sqrt_1mab[t].view(-1,1,1,1)
        return sqrt_ab * x0 + sqrt_1mab * noise

    @torch.no_grad()
    def p_sample(self, model, x_t, t):
        eps = model(x_t, t)
        a_t   = self.alpha[t].view(-1,1,1,1)
        ab_t  = self.alphabar[t].view(-1,1,1,1)
        oosa  = self.one_over_sqrt_a[t].view(-1,1,1,1)
        x0_hat = (x_t - torch.sqrt(torch.clamp(1 - ab_t, 1e-20)) * eps) / torch.sqrt(torch.clamp(ab_t, 1e-20))
        mean   = oosa * (x_t - (1 - a_t) / torch.sqrt(torch.clamp(1 - ab_t, 1e-20)) * eps)
        if (t == 0).all():
            return x0_hat
        var = self.posterior_var[(t-1).clamp(min=0)].view(-1,1,1,1)
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(torch.clamp(var, 1e-20)) * noise

    @torch.no_grad()
    def sample(self, model, n=32, shape=(1,32,32)):
        model.eval()
        x = torch.randn(n, *shape, device=self.device)
        for ti in reversed(range(self.T)):
            t = torch.full((n,), ti, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t)
        return x.clamp(-1,1)

        
def freeze_early_blocks(model, train_decoder_only=True):
    """
    Freeze encoder/early blocks. Keep late blocks trainable.
    If train_decoder_only=True: train bot + up2 + up1 + outc + late FiLM (s3/b3).
    """
    # 1) freeze everything
    for p in model.parameters():
        p.requires_grad_(False)

    # 2) ALWAYS keep time MLP trainable a little bit? (optional)
    # Comment out if you want it frozen.
    for p in model.time_mlp.parameters():
        p.requires_grad_(True)

    # 3) unfreeze late path
    train_names = [
        "bot",            # bottleneck
        "up2", "up1",     # decoder
        "outc",           # final conv
        "s3", "b3",       # late FiLM applied to bottleneck features
    ]
    if not train_decoder_only:
        # optionally also unfreeze mid encoder stage to give more capacity
        train_names += ["down2", "s2", "b2"]

    for name, module in model.named_modules():
        if name in train_names:
            for p in module.parameters():
                p.requires_grad_(True)

    # 4) sanity print
    n_all = sum(p.numel() for p in model.parameters())
    n_tr  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[freeze] trainable params: {n_tr:,} / {n_all:,} "
          f"({100.0*n_tr/n_all:.2f}%)")


# -------------------------------
# Data: loaders with class filtering
# -------------------------------
def get_dataloaders(dataset="mnist", pretrain_classes=(0,1,2,3,4), finetune_classes=(5,6,7,8,9),
                    batch_size=128, data_root="./data"):
    ds_name = dataset.lower()
    if ds_name not in ["mnist", "fashion", "cifar10"]:
        raise ValueError("dataset must be one of: mnist, fashion, cifar10")

    if ds_name in ["mnist", "fashion"]:
        in_ch = 1
        # normalize to [-1,1]
        tx = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5]),
        ])
        if ds_name == "mnist":
            train_full = datasets.MNIST(root=data_root, train=True, download=True, transform=tx)
        else:
            train_full = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=tx)
    else:
        in_ch = 3
        tx = transforms.Compose([
            transforms.ToTensor(),                      # CIFAR10 is already 32x32
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
        ])
        train_full = datasets.CIFAR10(root=data_root, train=True, download=True, transform=tx)

    # indices for pretrain vs finetune classes
    targets = np.array(train_full.targets if hasattr(train_full, "targets") else train_full.train_labels)
    pre_idx = np.where(np.isin(targets, pretrain_classes))[0]
    fin_idx = np.where(np.isin(targets, finetune_classes))[0]

    train_pre = Subset(train_full, pre_idx)
    train_fin = Subset(train_full, fin_idx)

    loader_pre = DataLoader(train_pre, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    loader_fin = DataLoader(train_fin, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    return loader_pre, loader_fin, in_ch

# -------------------------------
# Training loops
# -------------------------------
def train_epoch(model, ddpm, loader, opt):
    model.train()
    total = 0.0
    for x0, _ in loader:
        x0 = x0.to(DEVICE)                          # [B,C,32,32] in [-1,1]
        b = x0.size(0)
        t = torch.randint(0, ddpm.T, (b,), device=DEVICE, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = ddpm.q_sample(x0, t, noise)
        pred = model(x_t, t)
        loss = F.mse_loss(pred, noise)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * b
    return total / len(loader.dataset)

# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist","fashion","cifar10"])
    parser.add_argument("--pretrain-classes", type=int, nargs="+", default=[0,1,2,3,4])
    parser.add_argument("--finetune-classes", type=int, nargs="+", default=[5,6,7,8,9])
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_decoder_only", action="store_true",
                    help="Freeze encoder; train decoder/bot/outc only")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs("samples", exist_ok=True)

    loader_pre, loader_fin, in_ch = get_dataloaders(
        dataset=args.dataset,
        pretrain_classes=tuple(args.pretrain_classes),
        finetune_classes=tuple(args.finetune_classes),
        batch_size=args.batch_size
    )

    # Model + DDPM
    base_width = 64 if in_ch == 3 else 64
    time_dim = 128
    model = UNetSmall(in_ch=in_ch, base=base_width, time_dim=time_dim).to(DEVICE)
    ddpm  = DDPM(timesteps=200, beta_start=1e-4, beta_end=2e-2, device=DEVICE)

    # Stage 1: pretrain on split A
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    for ep in range(args.pretrain_epochs):
        loss = train_epoch(model, ddpm, loader_pre, opt)
        print(f"[pretrain] epoch {ep+1:02d}  loss={loss:.4f}")

    with torch.no_grad():
        samples = ddpm.sample(model, n=32, shape=(in_ch,32,32))
    save_grid(samples, f"samples/pretrained_{args.dataset}.png")
    print("saved:", f"samples/pretrained_{args.dataset}.png")
    if args.train_decoder_only:
        freeze_early_blocks(model, train_decoder_only=True)
    
        # optimizer over trainable subset only
        opt_ft = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
        for ep in range(args.finetune_epochs):
            loss = train_epoch(model, ddpm, loader_fin, opt_ft)
            print(f"[finetune (partial-freeze)] epoch {ep+1:02d}  loss={loss:.4f}")
    else:    
        # Stage 2: FULL fine-tune on split B (all params trainable)
        opt_ft = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for ep in range(args.finetune_epochs):
            loss = train_epoch(model, ddpm, loader_fin, opt_ft)
            print(f"[finetune] epoch {ep+1:02d}  loss={loss:.4f}")

    with torch.no_grad():
        samples_ft = ddpm.sample(model, n=32, shape=(in_ch,32,32))
    save_grid(samples_ft, f"samples/finetuned_{args.dataset}.png")
    print("saved:", f"samples/finetuned_{args.dataset}.png")

if __name__ == "__main__":
    main()
