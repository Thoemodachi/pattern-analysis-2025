import os, time, argparse, json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

# Import the model definition from your training file
from fast_cifar10 import ResNet18

CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

def build_loader(data_dir, batch=2048, workers=4, use_cuda=True):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=tf)
    loader = DataLoader(
        test_set,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=use_cuda,
        persistent_workers=use_cuda and workers > 0,
    )
    return loader, test_set

def load_model(ckpt_path, device):
    model = ResNet18().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)  # support both dict or raw state_dict
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def eval_testset(model, loader, device, use_cuda=True, save_preds=None):
    correct = tot = 0
    preds_all, labels_all = [], []
    t0 = time.time()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(imgs)
            pred = logits.argmax(1)
            correct += (pred == labels).sum().item()
            tot += labels.size(0)
            preds_all.append(pred.cpu())
            labels_all.append(labels.cpu())
    elapsed = time.time() - t0
    acc = 100.0 * correct / tot
    preds_all = torch.cat(preds_all)
    labels_all = torch.cat(labels_all)

    out = {
        "test_accuracy": round(acc, 3),
        "num_examples": int(tot),
        "elapsed_seconds": round(elapsed, 3),
    }

    if save_preds:
        # Save per-example predictions as JSONL
        with open(save_preds, "w") as f:
            for yhat, y in zip(preds_all.tolist(), labels_all.tolist()):
                f.write(json.dumps({"pred": int(yhat), "label": int(y)}) + "\n")

    return out, preds_all, labels_all

def confusion_matrix(preds, labels, num_classes=10):
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for p, t in zip(preds, labels):
        cm[t, p] += 1
    return cm

def infer_single(model, image_path, device, use_cuda=True):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    tf = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
        logits = model(x)
        prob = F.softmax(logits, dim=1)[0].cpu()
        pred_idx = int(prob.argmax().item())
    return pred_idx, prob.tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="./data", help="CIFAR-10 root")
    ap.add_argument("--ckpt", type=str, default="cifar10_resnet18_baseline.pt", help="checkpoint path")
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out_dir", type=str, default="./runs")
    ap.add_argument("--save_preds", type=str, default="", help="optional JSONL path for predictions")
    ap.add_argument("--single_image", type=str, default="", help="classify one image instead of test set")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda = (device == "cuda")

    # Load model
    model = load_model(args.ckpt, device)

    if args.single_image:
        pred_idx, probs = infer_single(model, args.single_image, device, use_cuda)
        print(json.dumps({
            "mode": "single_image",
            "image": args.single_image,
            "pred_index": pred_idx,
            "pred_class": CIFAR10_CLASSES[pred_idx],
            "probs": probs
        }, indent=2))
        return

    # Test-set evaluation
    loader, _ = build_loader(args.data, batch=args.batch, workers=args.workers, use_cuda=use_cuda)
    out, preds, labels = eval_testset(model, loader, device, use_cuda, save_preds=args.save_preds or None)
    print(json.dumps({"mode":"testset", **out}, indent=2))

    # Confusion matrix (printed as a compact list of lists)
    cm = confusion_matrix(preds, labels, num_classes=10).tolist()
    print(json.dumps({"confusion_matrix": cm, "classes": CIFAR10_CLASSES}, indent=2))

if __name__ == "__main__":
    main()
