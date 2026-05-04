"""
main_test.py

6개 모델 중 하나를 UI로 선택하여 CIFAR-10 test set 평가

지원 모델:
1. BasicNN_MLP
2. CustomCNN
3. ResNet18
4. ResNet50
5. ResNet50_Transfer
6. ViT_B16_Transfer

특징:
- Tkinter UI로 모델 선택
- CIFAR-10 test set 10,000장 전체 평가
- Confusion Matrix 표시
- Sample Prediction 표시
- 결과 이미지는 저장하지 않음
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
import random
import tkinter as tk
from tkinter import ttk, messagebox

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50, vit_b_16

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

DATA_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────
# Common Config
# ─────────────────────────────────────
NUM_WORKERS = 4

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ─────────────────────────────────────
# Model Definitions
# ─────────────────────────────────────
class BasicNN_MLP(nn.Module):
    def __init__(self, hidden1=1024, hidden2=512, num_classes=10):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),                    # net.0

            nn.Linear(32 * 32 * 3, hidden1), # net.1
            nn.BatchNorm1d(hidden1),         # net.2
            nn.ReLU(inplace=True),           # net.3
            nn.Dropout(0.3),                 # net.4

            nn.Linear(hidden1, hidden2),      # net.5
            nn.BatchNorm1d(hidden2),         # net.6
            nn.ReLU(inplace=True),           # net.7
            nn.Dropout(0.3),                 # net.8

            nn.Linear(hidden2, num_classes), # net.9
        )

    def forward(self, x):
        return self.net(x)


class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),

                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),

                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(3, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def make_resnet18(num_classes=10):
    model = resnet18(weights=None)

    model.conv1 = nn.Conv2d(
        3, 64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def make_resnet50(num_classes=10):
    model = resnet50(weights=None)

    model.conv1 = nn.Conv2d(
        3, 64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def make_resnet50_transfer(num_classes=10):
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def make_vit_b16_transfer(num_classes=10):
    model = vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model


# ─────────────────────────────────────
# Model Config
# ─────────────────────────────────────
MODEL_CONFIGS = {
    "1. BasicNN_MLP": {
        "model_name": "BasicNN_MLP",
        "path": MODEL_DIR / "BasicNN_MLP_best.pth",
        "type": "mlp_auto",
        "input_size": 32,
        "mean": CIFAR10_MEAN,
        "std": CIFAR10_STD,
        "batch_size": 256,
        "label_smoothing": 0.0,
    },
    "2. CustomCNN": {
        "model_name": "CustomCNN",
        "path": MODEL_DIR / "CustomCNN_best.pth",
        "maker": lambda: CustomCNN(num_classes=10),
        "input_size": 32,
        "mean": CIFAR10_MEAN,
        "std": CIFAR10_STD,
        "batch_size": 256,
        "label_smoothing": 0.0,
    },
    "3. ResNet18": {
        "model_name": "ResNet18",
        "path": MODEL_DIR / "ResNet18_best.pth",
        "maker": lambda: make_resnet18(num_classes=10),
        "input_size": 32,
        "mean": CIFAR10_MEAN,
        "std": CIFAR10_STD,
        "batch_size": 128,
        "label_smoothing": 0.0,
    },
    "4. ResNet50": {
        "model_name": "ResNet50",
        "path": MODEL_DIR / "ResNet50_best.pth",
        "maker": lambda: make_resnet50(num_classes=10),
        "input_size": 32,
        "mean": CIFAR10_MEAN,
        "std": CIFAR10_STD,
        "batch_size": 128,
        "label_smoothing": 0.0,
    },
    "5. ResNet50 Transfer": {
        "model_name": "ResNet50_Transfer",
        "path": MODEL_DIR / "ResNet50_Transfer_best.pth",
        "maker": lambda: make_resnet50_transfer(num_classes=10),
        "input_size": 224,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "batch_size": 64,
        "label_smoothing": 0.0,
    },
    "6. ViT-B/16 Transfer": {
        "model_name": "ViT_B16_Transfer",
        "path": MODEL_DIR / "ViT_B16_Transfer_best.pth",
        "maker": lambda: make_vit_b16_transfer(num_classes=10),
        "input_size": 224,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "batch_size": 64,
        "label_smoothing": 0.1,
    },
}


# ─────────────────────────────────────
# UI
# ─────────────────────────────────────
def choose_model_ui():
    selected = {
        "model_key": None,
        "random_sample": False
    }

    root = tk.Tk()
    root.title("CIFAR-10 Model Test")
    root.geometry("420x220")
    root.resizable(False, False)

    title = ttk.Label(
        root,
        text="테스트할 모델을 선택하세요",
        font=("Arial", 14, "bold")
    )
    title.pack(pady=15)

    model_var = tk.StringVar(value=list(MODEL_CONFIGS.keys())[0])

    combo = ttk.Combobox(
        root,
        textvariable=model_var,
        values=list(MODEL_CONFIGS.keys()),
        state="readonly",
        width=35
    )
    combo.pack(pady=10)

    random_var = tk.BooleanVar(value=False)
    random_check = ttk.Checkbutton(
        root,
        text="Sample prediction 이미지를 랜덤으로 선택",
        variable=random_var
    )
    random_check.pack(pady=10)

    def run_clicked():
        selected["model_key"] = model_var.get()
        selected["random_sample"] = random_var.get()
        root.destroy()

    def cancel_clicked():
        selected["model_key"] = None
        root.destroy()

    button_frame = ttk.Frame(root)
    button_frame.pack(pady=15)

    run_btn = ttk.Button(button_frame, text="Run Test", command=run_clicked)
    run_btn.grid(row=0, column=0, padx=10)

    cancel_btn = ttk.Button(button_frame, text="Cancel", command=cancel_clicked)
    cancel_btn.grid(row=0, column=1, padx=10)

    root.mainloop()

    return selected["model_key"], selected["random_sample"]


# ─────────────────────────────────────
# Checkpoint Load
# ─────────────────────────────────────
def load_checkpoint(model_path, device):
    if not model_path.exists():
        raise FileNotFoundError(
            f"\nModel file not found:\n{model_path}\n\n"
            f"saved_models 폴더 안에 해당 .pth 파일이 있는지 확인하세요."
        )

    try:
        checkpoint = torch.load(str(model_path), map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(str(model_path), map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format.")

    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v

    return new_state_dict


def make_mlp_from_state_dict(state_dict):
    hidden1 = state_dict["net.1.weight"].shape[0]
    hidden2 = state_dict["net.5.weight"].shape[0]
    num_classes = state_dict["net.9.weight"].shape[0]

    print("\n[INFO] Inferred MLP architecture")
    print(f"Input   : 3072")
    print(f"Hidden1 : {hidden1}")
    print(f"Hidden2 : {hidden2}")
    print(f"Output  : {num_classes}")

    return BasicNN_MLP(
        hidden1=hidden1,
        hidden2=hidden2,
        num_classes=num_classes
    )


def build_and_load_model(config, device):
    state_dict = load_checkpoint(config["path"], device)

    if config.get("type") == "mlp_auto":
        model = make_mlp_from_state_dict(state_dict)
    else:
        model = config["maker"]()

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)

    return model


# ─────────────────────────────────────
# DataLoader
# ─────────────────────────────────────
def get_test_dataset_and_loader(config):
    input_size = config["input_size"]
    mean = config["mean"]
    std = config["std"]

    tf_list = []

    if input_size != 32:
        tf_list.append(transforms.Resize((input_size, input_size), antialias=True))

    tf_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_tf = transforms.Compose(tf_list)

    cifar_dir = DATA_DIR / "cifar-10-batches-py"
    if not cifar_dir.exists():
        raise FileNotFoundError(
            f"\nCIFAR-10 데이터가 없습니다.\n"
            f"예상 위치: {cifar_dir}\n\n"
            f"기존 cifar-10-batches-py 폴더를 {DATA_DIR} 안에 복사하세요."
        )

    testset = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=False,
        transform=test_tf
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if NUM_WORKERS > 0 else False
    )

    return testset, testloader


# ─────────────────────────────────────
# Evaluation
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
# Visualization
# ─────────────────────────────────────
def denormalize(img, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)

    img = img.cpu() * std + mean
    img = torch.clamp(img, 0, 1)

    return img


def plot_confusion_matrix(preds, labels, model_name, test_acc):
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

    ax.set_title(f"{model_name} - Test Confusion Matrix Acc: {test_acc:.2f}%")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def show_sample_predictions(
    model,
    testset,
    device,
    model_name,
    mean,
    std,
    random_sample=False,
    num_samples=10
):
    model.eval()

    if random_sample:
        indices = random.sample(range(len(testset)), num_samples)
    else:
        indices = list(range(num_samples))

    images = []
    labels = []

    for idx in indices:
        img, label = testset[idx]
        images.append(img)
        labels.append(label)

    images = torch.stack(images, dim=0).to(device)
    labels = torch.tensor(labels)

    use_amp = device.type == "cuda"
    device_type = "cuda" if device.type == "cuda" else "cpu"

    with torch.amp.autocast(device_type=device_type, enabled=use_amp):
        out = model(images)

    preds = out.argmax(dim=1).cpu()
    images = images.cpu()

    plt.figure(figsize=(15, 4))

    for i in range(num_samples):
        img = denormalize(images[i], mean, std)
        img = img.permute(1, 2, 0).numpy()

        gt_name = CIFAR10_CLASSES[labels[i].item()]
        pred_name = CIFAR10_CLASSES[preds[i].item()]

        plt.subplot(2, 5, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"GT: {gt_name}\nPred: {pred_name}", fontsize=9)

    sample_type = "Random Samples" if random_sample else "Fixed First Samples"
    plt.suptitle(f"{model_name} - {sample_type}", fontsize=14)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────
# Main
# ─────────────────────────────────────
def main():
    model_key, random_sample = choose_model_ui()

    if model_key is None:
        print("Test canceled.")
        return

    config = MODEL_CONFIGS[model_key]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("CIFAR-10 Model Test")
    print("=" * 70)
    print(f"Selected   : {model_key}")
    print(f"Device     : {device}")

    if device.type == "cuda":
        print(f"GPU        : {torch.cuda.get_device_name(0)}")

    print(f"Project Dir: {PROJECT_DIR}")
    print(f"Data Dir   : {DATA_DIR}")
    print(f"Model Path : {config['path']}")
    print("=" * 70)

    testset, testloader = get_test_dataset_and_loader(config)

    model = build_and_load_model(config, device)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=config["label_smoothing"]
    )

    test_loss, test_acc, preds, labels = evaluate(
        model=model,
        loader=testloader,
        criterion=criterion,
        device=device
    )

    print("\n" + "=" * 70)
    print("Final Test Result")
    print("=" * 70)
    print(f"Model        : {config['model_name']}")
    print(f"Test Loss    : {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("=" * 70)

    plot_confusion_matrix(
        preds=preds,
        labels=labels,
        model_name=config["model_name"],
        test_acc=test_acc
    )

    show_sample_predictions(
        model=model,
        testset=testset,
        device=device,
        model_name=config["model_name"],
        mean=config["mean"],
        std=config["std"],
        random_sample=random_sample,
        num_samples=10
    )


if __name__ == "__main__":
    main()