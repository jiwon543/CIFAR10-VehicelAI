"""
Load and Test ResNet50 Transfer Model on CIFAR-10

Project structure:
CIFAR10_project/
├─ main/
│  └─ test_ResNet50Transfer.py
├─ saved_models/
│  └─ ResNet50_Transfer_best.pth
├─ result_image/
└─ data/
"""
import warnings
try:
    from numpy.exceptions import VisibleDeprecationWarning
except ImportError:
    from numpy import VisibleDeprecationWarning
warnings.filterwarnings(
    "ignore",
    category=VisibleDeprecationWarning,
    message=r"dtype\(\): align should be passed.*"
)

from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ─────────────────────────────────────
# Path Config
# ─────────────────────────────────────
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent

DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "saved_models"
RESULT_DIR = PROJECT_DIR / "result_image"

MODEL_NAME = "ResNet50_Transfer"
MODEL_PATH = MODEL_DIR / "ResNet50_Transfer_best.pth"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────
# Config
# ─────────────────────────────────────
BATCH_SIZE = 64
NUM_WORKERS = 4
IMG_SIZE = 224

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ─────────────────────────────────────
# Model
# ─────────────────────────────────────
def make_model(num_classes=10):
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)

    # ImageNet 1000 class → CIFAR-10 10 class
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


# ─────────────────────────────────────
# Test Loader
# ─────────────────────────────────────
def get_test_loader():
    test_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    testset = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=False,
        transform=test_tf
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if NUM_WORKERS > 0 else False
    )

    return testloader


# ─────────────────────────────────────
# Evaluate
# ─────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    use_amp = device.type == "cuda"
    device_type = "cuda" if device.type == "cuda" else "cpu"

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            out = model(x)
            loss = criterion(out, y)

        pred = out.argmax(dim=1)

        loss_sum += loss.item() * x.size(0)
        correct += (pred == y).sum().item()
        total += y.size(0)

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    avg_loss = loss_sum / total
    avg_acc = 100.0 * correct / total

    return avg_loss, avg_acc, np.array(all_preds), np.array(all_labels)


# ─────────────────────────────────────
# Load State Dict
# ─────────────────────────────────────
def load_model_weights(model, model_path, device):
    if not model_path.exists():
        raise FileNotFoundError(
            f"\nModel file not found:\n{model_path}\n\n"
            f"현재 찾는 위치가 맞는지 확인.\n"
            f"예상 위치: CIFAR10_project/saved_models/ResNet50_Transfer_best.pth"
        )

    print(f"\nLoading model from:")
    print(model_path)

    try:
        checkpoint = torch.load(str(model_path), map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(str(model_path), map_location=device)

    # 그냥 state_dict만 저장한 경우
    if isinstance(checkpoint, dict) and "state_dict" not in checkpoint:
        state_dict = checkpoint

    # checkpoint 형태로 저장한 경우
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        raise ValueError("Unsupported checkpoint format.")

    model.load_state_dict(state_dict)
    return model


# ─────────────────────────────────────
# Confusion Matrix
# ─────────────────────────────────────
def plot_confusion_matrix(preds, labels, test_acc):
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CIFAR10_CLASSES
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        xticks_rotation=45
    )

    ax.set_title(f"{MODEL_NAME} - Test Confusion Matrix Acc: {test_acc:.2f}%")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")

    plt.tight_layout()

    save_path = RESULT_DIR / f"test_{MODEL_NAME}_confusion_matrix.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[SAVE] Confusion matrix saved to: {save_path}")

    plt.show()


# ─────────────────────────────────────
# Sample Prediction Visualization
# ─────────────────────────────────────
def denormalize(img):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    img = img.cpu() * std + mean
    img = torch.clamp(img, 0, 1)

    return img


@torch.no_grad()
def show_sample_predictions(model, loader, device, num_samples=10):
    model.eval()

    images, labels = next(iter(loader))
    images = images.to(device, non_blocking=True)

    out = model(images)
    preds = out.argmax(dim=1).cpu()

    images = images.cpu()

    plt.figure(figsize=(15, 4))

    for i in range(num_samples):
        img = denormalize(images[i])
        img = img.permute(1, 2, 0).numpy()

        gt_name = CIFAR10_CLASSES[labels[i].item()]
        pred_name = CIFAR10_CLASSES[preds[i].item()]

        plt.subplot(2, 5, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"GT: {gt_name}\nPred: {pred_name}", fontsize=9)

    plt.tight_layout()

    save_path = RESULT_DIR / f"test_{MODEL_NAME}_samples.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[SAVE] Sample predictions saved to: {save_path}")

    plt.show()


# ─────────────────────────────────────
# Main
# ─────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("ResNet50 Transfer Test")
    print("=" * 60)
    print(f"Device     : {device}")

    if device.type == "cuda":
        print(f"GPU        : {torch.cuda.get_device_name(0)}")

    print(f"Project Dir: {PROJECT_DIR}")
    print(f"Data Dir   : {DATA_DIR}")
    print(f"Model Path : {MODEL_PATH}")
    print(f"Result Dir : {RESULT_DIR}")

    testloader = get_test_loader()

    model = make_model(num_classes=10).to(device)
    model = load_model_weights(model, MODEL_PATH, device)

    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc, preds, labels = evaluate(
        model=model,
        loader=testloader,
        criterion=criterion,
        device=device
    )

    print("\n" + "=" * 60)
    print("Final Test Result")
    print("=" * 60)
    print(f"Test Loss    : {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("=" * 60)

    plot_confusion_matrix(preds, labels, test_acc)
    show_sample_predictions(model, testloader, device, num_samples=10)


# Windows 필수
if __name__ == "__main__":
    main()