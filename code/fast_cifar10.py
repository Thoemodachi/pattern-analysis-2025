# DO NOT RUN ON LOCAL COMPUTER; Google Colab and Rangpur
# fast_cifar10.py
import math, time, os, random, sys, json
from dataclasses import dataclass, asdict
from argparse import ArgumentParser
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --------------------------
# Utilities
# --------------------------
def set_fast_cuda():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")  # TF32 matmul on Ampere+
        except Exception:
            pass

class Tee:
    """Mirror prints to stdout and a file."""
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._file = open(path, "a", buffering=1)
        self._stdout = sys.stdout
    def write(self, x):
        self._stdout.write(x)
        self._file.write(x)
    def flush(self):
        self._stdout.flush()
        self._file.flush()

class EMA:
    """Track EMA over floating-point parameters only; apply/restore for eval."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters()
                       if p.dtype.is_floating_point}
    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
    @torch.no_grad()
    def apply_to(self, model):
        backup = {n: p.detach().clone() for n, p in model.named_parameters() if n in self.shadow}
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])
        return backup
    @torch.no_grad()
    def restore(self, model, backup):
        for n, p in model.named_parameters():
            if n in backup:
                p.data.copy_(backup[n])

# --------------------------
# MixUp / CutMix
# --------------------------
from torchvision.transforms import RandAugment  # used by transforms below

def mixup_cutmix(x, y, num_classes=10, alpha=0.2, cutmix_prob=0.5):
    """Returns mixed inputs and soft targets."""
    one_hot = F.one_hot(y, num_classes=num_classes).float()
    if alpha <= 0:
        return x, one_hot
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    if random.random() < cutmix_prob:
        # CutMix
        B, C, H, W = x.size()
        rx, ry = random.randrange(W), random.randrange(H)
        r_w = int(W * math.sqrt(1 - lam))
        r_h = int(H * math.sqrt(1 - lam))
        x1 = max(rx - r_w // 2, 0); y1 = max(ry - r_h // 2, 0)
        x2 = min(rx + r_w // 2, W); y2 = min(ry + r_h // 2, H)
        perm = torch.randperm(B, device=x.device)
        x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
        targets = lam * one_hot + (1 - lam) * one_hot[perm]
        return x, targets
    else:
        # MixUp
        perm = torch.randperm(x.size(0), device=x.device)
        x_mixed = lam * x + (1 - lam) * x[perm]
        targets = lam * one_hot + (1 - lam) * one_hot[perm]
        return x_mixed, targets

# --------------------------
# ResNet-18 from scratch (CIFAR stem)
# --------------------------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.down  = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.down = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.down(x)
        return F.relu(out)

class ResNet18(nn.Module):
    def __init__(self, num_classes=10, width=64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, 3, 1, 1, bias=False)  # CIFAR stem
        self.bn1   = nn.BatchNorm2d(width)
        self.layer1 = self._make_layer(width,   width,   2, stride=1)
        self.layer2 = self._make_layer(width,   2*width, 2, stride=2)
        self.layer3 = self._make_layer(2*width, 4*width, 2, stride=2)
        self.layer4 = self._make_layer(4*width, 8*width, 2, stride=2)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(8*width, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    def _make_layer(self, in_planes, planes, blocks, stride):
        layers = [BasicBlock(in_planes, planes, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

# --------------------------
# Args & CLI
# --------------------------
@dataclass
class Args:
    data: str = "./data"
    batch: int = 1024
    epochs: int = 32
    lr: float = 0.4
    wd: float = 5e-4
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    label_smoothing: float = 0.05
    workers: int = 8
    ema: float = 0.999
    seed: int = 42
    out_dir: str = "./runs"

def parse_args() -> Args:
    p = ArgumentParser()
    p.add_argument("--data", type=str, default=Args.data)
    p.add_argument("--batch", type=int, default=Args.batch)
    p.add_argument("--epochs", type=int, default=Args.epochs)
    p.add_argument("--lr", type=float, default=Args.lr)
    p.add_argument("--wd", type=float, default=Args.wd)
    p.add_argument("--mixup_alpha", type=float, default=Args.mixup_alpha)
    p.add_argument("--cutmix_alpha", type=float, default=Args.cutmix_alpha)
    p.add_argument("--label_smoothing", type=float, default=Args.label_smoothing)
    p.add_argument("--workers", type=int, default=Args.workers)
    p.add_argument("--ema", type=float, default=Args.ema)
    p.add_argument("--seed", type=int, default=Args.seed)
    p.add_argument("--out_dir", type=str, default=Args.out_dir)
    ns = p.parse_args()
    return Args(**vars(ns))

# --------------------------
# Training
# --------------------------
def main(a: Args):
    torch.manual_seed(a.seed); random.seed(a.seed)

    # --- Run directory (SLURM-aware) ---
    ts   = time.strftime("%Y%m%d-%H%M%S")
    jn   = os.environ.get("SLURM_JOB_NAME", "local")
    jid  = os.environ.get("SLURM_JOB_ID", "nojob")
    run_name = f"{jn}-{jid}-{ts}"
    run_dir  = os.path.join(a.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Mirror stdout to file
    sys.stdout = Tee(os.path.join(run_dir, "train.log"))
    print(f"[INFO] Run dir: {run_dir}")
    print(f"[INFO] Args: {json.dumps(asdict(a))}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda = (device == "cuda")
    set_fast_cuda()

    # Data
    mean = (0.4914, 0.4822, 0.4465); std = (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.RandAugment(num_ops=2, magnitude=10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_set = datasets.CIFAR10(a.data, train=True, download=True, transform=train_tf)
    test_set  = datasets.CIFAR10(a.data, train=False, download=True, transform=test_tf)

    # DataLoader device-aware settings
    suggested_workers = 2 if not use_cuda else a.workers
    workers = min(a.workers, suggested_workers) if not use_cuda else a.workers

    train_loader = DataLoader(
        train_set,
        batch_size=a.batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=use_cuda,
        persistent_workers=use_cuda and workers > 0,
    )
    test_loader  = DataLoader(
        test_set,
        batch_size=2048,
        shuffle=False,
        num_workers=workers,
        pin_memory=use_cuda,
        persistent_workers=use_cuda and workers > 0,
    )

    # Model/opt
    model = ResNet18()
    model = model.to(device, memory_format=torch.channels_last) if use_cuda else model.to(device)
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception:
            pass

    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=a.wd, nesterov=True)

    # One-cycle schedule
    steps_per_epoch = math.ceil(len(train_loader.dataset) / a.batch)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, epochs=a.epochs, steps_per_epoch=steps_per_epoch,
        pct_start=0.15, div_factor=25, final_div_factor=1e4
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    ema = EMA(model, decay=a.ema)

    best = 0.0
    for epoch in range(a.epochs):
        model.train()
        t0 = time.time()
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True,
                           memory_format=torch.channels_last if use_cuda else torch.contiguous_format)
            labels = labels.to(device, non_blocking=True)

            # MixUp/CutMix
            imgs, targets = mixup_cutmix(imgs, labels, num_classes=10, alpha=a.mixup_alpha, cutmix_prob=0.5)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
                logits = model(imgs)
                # Cross-entropy with soft targets
                loss = -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            scheduler.step()
            ema.update(model)

        tr_time = time.time() - t0

        # --- Eval with EMA weights ---
        model.eval()
        backup = ema.apply_to(model)
        correct = tot = 0
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
            for imgs, labels in test_loader:
                imgs = imgs.to(device, non_blocking=True,
                               memory_format=torch.channels_last if use_cuda else torch.contiguous_format)
                labels = labels.to(device, non_blocking=True)
                logits = model(imgs)
                pred = logits.argmax(1)
                tot += labels.size(0)
                correct += (pred == labels).sum().item()
        ema.restore(model, backup)

        acc = 100.0 * correct / tot
        best = max(best, acc)
        summary = {
            "epoch": epoch + 1,
            "epochs": a.epochs,
            "train_time_sec": round(tr_time, 3),
            "test_acc": round(acc, 3),
            "best_acc": round(best, 3),
        }
        print(f"Epoch {epoch+1:02d}/{a.epochs} | {tr_time:.1f}s | test acc {acc:.2f}% | best {best:.2f}%")
        with open(os.path.join(run_dir, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(summary) + "\n")

    # Save for inference/demo
    ckpt_path = os.path.join(run_dir, "cifar10_resnet18_amp.pt")
    torch.save({"model": model.state_dict()}, ckpt_path)
    print(f"Saved model to {ckpt_path}")

if __name__ == "__main__":
    main(parse_args())
