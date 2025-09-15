import time, random, argparse
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
        torch.set_float32_matmul_precision("high")  # TF32 on Ampere+
    except Exception:
        pass

# --------------------------
# ResNet-18 (CIFAR stem) — identical to your training script
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
                nn.BatchNorm2d(planes),
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
    ckpt: str = "cifar10_resnet18_baseline.pt"
    data: str = "./data"
    batch: int = 2048
    workers: int = 8
    amp: bool = True     # mixed-precision inference on GPU
    seed: int = 42

def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=Args.ckpt)
    p.add_argument("--data", type=str, default=Args.data)
    p.add_argument("--batch", type=int, default=Args.batch)
    p.add_argument("--workers", type=int, default=Args.workers)
    p.add_argument("--amp", type=lambda s: s.lower() in {"1","true","yes","y"}, default=Args.amp)
    p.add_argument("--seed", type=int, default=Args.seed)
    ns, _ = p.parse_known_args()
    return Args(**vars(ns))

# --------------------------
# Timed evaluation
# --------------------------
def main():
    a = parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_fast_cuda()

    # Data
    mean = (0.4914, 0.4822, 0.4465); std = (0.2470, 0.2435, 0.2616)
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_set = datasets.CIFAR10(a.data, train=False, download=True, transform=tfm)
    test_loader = DataLoader(
        test_set, batch_size=a.batch, shuffle=False,
        num_workers=a.workers, pin_memory=(device=="cuda"), persistent_workers=(device=="cuda" and a.workers>0)
    )

    # Model
    model = ResNet18().to(device)
    if device == "cuda":
        model = model.to(memory_format=torch.channels_last)

    ckpt = torch.load(a.ckpt, map_location=device)
    state = ckpt.get("model", ckpt)  # allow raw state_dict or {"model": state_dict}
    model.load_state_dict(state, strict=True)
    model.eval()

    # Warm-up (build cuDNN kernels; excluded from timing)
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(device=="cuda" and a.amp)):
        for i, (x, _) in enumerate(test_loader):
            x = x.to(device, non_blocking=True)
            if device == "cuda":
                x = x.to(memory_format=torch.channels_last)
            _ = model(x)
            if i >= 1:  # a couple of warmup batches
                break

    # Timed loop
    n_correct = 0
    n_total = 0
    torch.cuda.synchronize() if device=="cuda" else None
    t0 = time.time()

    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(device=="cuda" and a.amp)):
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if device == "cuda":
                x = x.to(memory_format=torch.channels_last)
            logits = model(x)
            preds = logits.argmax(1)
            n_total += y.size(0)
            n_correct += (preds == y).sum().item()

    torch.cuda.synchronize() if device=="cuda" else None
    elapsed = time.time() - t0

    top1 = 100.0 * n_correct / n_total
    ips = n_total / elapsed if elapsed > 0 else float("inf")

    print(f"[TEST] device={device} amp={a.amp} "
          f"| batches={len(test_loader)} "
          f"| total_imgs={n_total} "
          f"| time={elapsed:.3f}s "
          f"| throughput={ips:.1f} img/s "
          f"| top1={top1:.2f}%")

if __name__ == "__main__":
    main()
