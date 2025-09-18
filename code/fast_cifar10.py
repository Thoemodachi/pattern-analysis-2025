# DO NOT RUN ON LOCAL COMPUTER; Google Colab and Ranpur
# fast_cifar10.py  — baseline that learns + proper val split + runtime tracking

import math, time, os, random
from dataclasses import dataclass
from argparse import ArgumentParser
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# --------------------------
# Utilities
# --------------------------
def set_fast_cuda():
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")  # TF32 on Ampere+
    except Exception:
        pass

class EMA:
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
# ResNet-18 (CIFAR stem)
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
        self.conv1 = nn.Conv2d(3, width, 3, 1, 1, bias=False)
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
# Args
# --------------------------
@dataclass
class Args:
    data: str = "./data"
    batch: int = 128
    epochs: int = 40
    lr: float = 0.1
    wd: float = 5e-4
    workers: int = 1
    ema: float = 0.999
    seed: int = 42
    val_size: int = 5000   # from CIFAR-10 train set (50k)
    log_interval: int = 100  # print every N steps

def parse_args():
    ap = ArgumentParser()
    for f in Args.__dataclass_fields__.values():
        ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    args, _ = ap.parse_known_args()
    return args

# --------------------------
# Training
# --------------------------
def main():
    a = parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_fast_cuda()

    # Data transforms: SIMPLE baseline that learns
    mean = (0.4914, 0.4822, 0.4465); std = (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Datasets
    full_train = datasets.CIFAR10(a.data, train=True,  download=True, transform=train_tf)
    test_set   = datasets.CIFAR10(a.data, train=False, download=True, transform=test_tf)

    # Train / Val split (reproducible)
    val_size = a.val_size
    train_size = len(full_train) - val_size
    gen = torch.Generator().manual_seed(a.seed)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=gen)

    # Loaders
    train_loader = DataLoader(train_set, batch_size=a.batch, shuffle=True,
                              num_workers=a.workers, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_set,   batch_size=2048, shuffle=False,
                              num_workers=a.workers, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_set,  batch_size=2048, shuffle=False,
                              num_workers=a.workers, pin_memory=True, persistent_workers=True)

    # Model / Opt (baseline: NO compile, NO AMP)
    model = ResNet18().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=a.wd, nesterov=True)
    steps_per_epoch = len(train_loader)  # per-iteration scheduler

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, epochs=a.epochs, steps_per_epoch=steps_per_epoch,
        pct_start=0.15, div_factor=25, final_div_factor=1e4
    )
    ema = EMA(model, decay=a.ema)

    # --------------------------
    # RUNTIME TRACKING
    # --------------------------
    print("Training")
    overall_start = time.time()

    best_val = 0.0
    for epoch in range(a.epochs):
        model.train()
        epoch_start = time.time()

        total_step = len(train_loader)
        running_loss = 0.0

        for i, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            scheduler.step()
            ema.update(model)

            running_loss += loss.item()

            if (i + 1) % a.log_interval == 0 or (i + 1) == total_step:
                avg_loss = running_loss / (i + 1)
                print(f"Epoch [{epoch+1}/{a.epochs}] "
                      f"Step [{i+1}/{total_step}] "
                      f"LR {scheduler.get_last_lr()[0]:.5f} "
                      f"Loss {loss.item():.4f} (avg {avg_loss:.4f})")

        train_epoch_time = time.time() - epoch_start

        # ---- Validation with EMA weights ----
        val_start = time.time()
        model.eval()
        backup = ema.apply_to(model)

        correct = tot = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(imgs)
                pred = logits.argmax(1)
                tot += labels.size(0)
                correct += (pred == labels).sum().item()

        ema.restore(model, backup)
        val_time = time.time() - val_start

        val_acc = 100.0 * correct / tot
        best_val = max(best_val, val_acc)
        print(f"Epoch {epoch+1:02d}/{a.epochs} | "
              f"train {train_epoch_time:.1f}s | "
              f"val {val_time:.1f}s | "
              f"val acc {val_acc:.2f}% | best {best_val:.2f}%")

    # ---- Final Test (EMA weights) ----
    test_start = time.time()
    model.eval()
    backup = ema.apply_to(model)
    correct = tot = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(imgs)
            pred = logits.argmax(1)
            tot += labels.size(0)
            correct += (pred == labels).sum().item()
    test_time = time.time() - test_start
    test_acc = 100.0 * correct / tot
    print(f"[FINAL] test acc {test_acc:.2f}% | test time {test_time:.1f}s")

    ema.restore(model, backup)

    # Overall runtime
    overall_elapsed = time.time() - overall_start
    print(f"Training took: {overall_elapsed:.1f} seconds "
          f"({overall_elapsed/60:.2f} minutes)")

    # Save
    torch.save({"model": model.state_dict()}, "trained_model.pt")
    print("Saved model to trained_model.pt")

if __name__ == "__main__":
    main()
