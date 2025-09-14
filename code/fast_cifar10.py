# fast_cifar10.py
import math, time, os, random
from dataclasses import dataclass
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --------------------------
# Utilities
# --------------------------
def set_fast_cuda():
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")  # enables TF32 matmul on Ampere+
    except Exception:
        pass

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1-self.decay)
    def load(self, model):
        model.load_state_dict(self.shadow, strict=False)

# --------------------------
# RandAugment (lightweight)
# --------------------------
from torchvision.transforms import RandAugment  # allowed as a transform utility

def mixup_cutmix(x, y, num_classes=10, alpha=0.2, cutmix_prob=0.5):
    """Returns mixed inputs, mixed targets; y one-hot."""
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
        self.layer1 = self._make_layer(width,   width, 2, stride=1)
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
# Training
# --------------------------
@dataclass
class Args:
    data: str = "./data"
    batch: int = 1024        # A100 can handle this with AMP; tune if OOM
    epochs: int = 32
    lr: float = 0.4          # peak 1cycle LR for large batch
    wd: float = 5e-4
    cutmix_alpha: float = 1.0
    mixup_alpha: float = 0.2
    label_smoothing: float = 0.05
    workers: int = 8
    ema: float = 0.999
    seed: int = 42

def main(a=Args()):
    torch.manual_seed(a.seed); random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    train_loader = DataLoader(train_set, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                              pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_set, batch_size=2048, shuffle=False, num_workers=a.workers,
                              pin_memory=True, persistent_workers=True)

    # Model/opt
    model = ResNet18().to(device, memory_format=torch.channels_last)
    model = torch.compile(model) if hasattr(torch, "compile") else model  # PyTorch 2.x
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=a.wd, nesterov=True)

    # One-cycle schedule
    steps_per_epoch = math.ceil(len(train_loader.dataset)/a.batch)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, epochs=a.epochs, steps_per_epoch=steps_per_epoch,
        pct_start=0.15, div_factor=25, final_div_factor=1e4
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))
    criterion = nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)
    ema = EMA(model, decay=a.ema)

    # Train
    best = 0.0
    for epoch in range(a.epochs):
        model.train()
        t0 = time.time()
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            # MixUp/CutMix
            imgs, targets = mixup_cutmix(imgs, labels, num_classes=10, alpha=a.mixup_alpha, cutmix_prob=0.5)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device=="cuda")):
                logits = model(imgs)
                loss = -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()  # CE with soft targets
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            scheduler.step()
            ema.update(model)
        tr_time = time.time() - t0

        # Eval (EMA weights)
        model.eval()
        ema.load(model)
        correct = tot = 0
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device=="cuda")):
            for imgs, labels in test_loader:
                imgs = imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
                labels = labels.to(device, non_blocking=True)
                logits = model(imgs); pred = logits.argmax(1)
                tot += labels.size(0); correct += (pred==labels).sum().item()
        acc = 100.0*correct/tot
        best = max(best, acc)
        print(f"Epoch {epoch+1:02d}/{a.epochs} | {tr_time:.1f}s | test acc {acc:.2f}% | best {best:.2f}%")

    # Save for inference demo
    torch.save({"model": model.state_dict()}, "cifar10_resnet18_amp.pt")
    print("Saved model to cifar10_resnet18_amp.pt")

if __name__ == "__main__":
    main()
