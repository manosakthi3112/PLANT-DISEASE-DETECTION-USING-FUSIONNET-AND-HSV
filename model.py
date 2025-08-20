# train_hybrid.py
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from PIL import Image
import numpy as np
import cv2
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from torch.cuda.amp import autocast, GradScaler

# ----------------- User config -----------------
DATASET_ROOT = r"C:\Users\manos\Music\plant\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)"
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VALID_DIR = os.path.join(DATASET_ROOT, "valid")

BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# Choose number of workers (on Windows, using >0 is OK if __main__ used)
NUM_WORKERS = min(4, max(0, (os.cpu_count() or 1) // 2))

# HSV mask ranges (you can tune these)
GREEN_LOWER = np.array([20, 40, 40], dtype=np.uint8)
GREEN_UPPER = np.array([90, 255, 255], dtype=np.uint8)

# OPTIONAL: brown/dark diseased spot range (uncomment to use additional mask)
# BROWN_LOWER = np.array([5, 50, 20], dtype=np.uint8)
# BROWN_UPPER = np.array([30, 255, 200], dtype=np.uint8)
# ------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------- Custom HSV Dataset ---------------
class HSVFilteredDataset(torch.utils.data.Dataset):
    """
    Wraps an ImageFolder dataset and applies an HSV mask to emphasize leaf regions.
    Returns a PIL image after masking so torchvision transforms work.
    """
    def __init__(self, imagefolder_dataset, lower_hsv, upper_hsv, transform=None):
        self.base = imagefolder_dataset
        self.lower_hsv = np.array(lower_hsv, dtype=np.uint8)
        self.upper_hsv = np.array(upper_hsv, dtype=np.uint8)
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        path, label = self.base.samples[idx]
        # read with cv2 to apply HSV; convert to RGB for PIL
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            # If image failed to load, return a zero tensor (keeps pipeline running)
            blank = Image.new("RGB", (256, 256), (0, 0, 0))
            if self.transform:
                return self.transform(blank), label
            return blank, label

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)

        # OPTIONAL: combine with additional mask for brown spots
        # hsv_b = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        # mask_b = cv2.inRange(hsv_b, BROWN_LOWER, BROWN_UPPER)
        # mask = cv2.bitwise_or(mask, mask_b)

        result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(result_rgb)

        if self.transform:
            pil_img = self.transform(pil_img)
        return pil_img, label

# ---------------- Hybrid Model ----------------
class DenseEfficient(nn.Module):
    """
    DenseNet201 + EfficientNetV2-M hybrid:
    - extract features from both backbones (without their classifiers)
    - global average pool both, concat, and predict
    """
    def __init__(self, num_classes, use_pretrained=True):
        super().__init__()
        # DenseNet201 backbone
        densenet = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1 if use_pretrained else None)
        # For densenet, .features contains conv blocks (feature map)
        self.densenet_features = densenet.features  # output channels: 1920

        # EfficientNetV2-M backbone
        effnet = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1 if use_pretrained else None)
        # EfficientNet typically exposes .features as well
        self.effnet_features = effnet.features  # output channels: 1280 (before classifier)

        # Adaptive pooling to 1x1 and classifier
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(1920 + 1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # DenseNet path
        f1 = self.densenet_features(x)            # [B, 1920, H, W]
        f1 = self.pool(f1)
        f1 = torch.flatten(f1, 1)                # [B, 1920]

        # EfficientNet path
        f2 = self.effnet_features(x)              # [B, 1280, H, W]
        f2 = self.pool(f2)
        f2 = torch.flatten(f2, 1)                # [B, 1280]

        f = torch.cat([f1, f2], dim=1)           # [B, 3200]
        out = self.classifier(f)
        return out

# ----------------- Utility: safe folder checks -----------------
def count_classes(train_dir):
    return len([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

# ----------------- Main: training pipeline -----------------
if __name__ == "__main__":
    if not os.path.exists(TRAIN_DIR):
        raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")

    use_validation = os.path.exists(VALID_DIR) and len(os.listdir(VALID_DIR)) > 0 and \
                     any(os.path.isdir(os.path.join(VALID_DIR, d)) for d in os.listdir(VALID_DIR))

    num_classes = count_classes(TRAIN_DIR)
    if num_classes == 0:
        raise ValueError("No class subfolders found in training directory.")

    print(f"Found {num_classes} classes. Validation set present: {use_validation}")

    # Transforms: keep same normalization as ImageNet backbones expect
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Base ImageFolder datasets
    train_base = datasets.ImageFolder(TRAIN_DIR)
    train_dataset = HSVFilteredDataset(train_base, GREEN_LOWER, GREEN_UPPER, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)

    if use_validation:
        valid_base = datasets.ImageFolder(VALID_DIR)
        valid_dataset = HSVFilteredDataset(valid_base, GREEN_LOWER, GREEN_UPPER, transform=transform)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=True)
    else:
        valid_loader = None

    # Model, criterion, optimizer, scaler
    model = DenseEfficient(num_classes=num_classes, use_pretrained=True).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # Training bookkeeping
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    precision_list, recall_list, f1_list = [], [], []

    best_val_loss = float("inf")
    print("Starting training for", NUM_EPOCHS, "epochs")
    t0 = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(enabled=(device.type == "cuda")):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs.detach(), dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = 100.0 * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        precision_list.append(precision); recall_list.append(recall); f1_list.append(f1)

        epoch_model_path = os.path.join(SAVE_DIR, f"model_epoch{epoch+1}_loss{epoch_loss:.4f}.pth")
        torch.save(model.state_dict(), epoch_model_path)

        print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% "
              f"| P: {precision:.4f} R: {recall:.4f} F1: {f1:.4f}")

        # Validation
        if valid_loader is not None:
            model.eval()
            v_running_loss = 0.0
            v_correct = 0
            v_total = 0
            v_preds = []
            v_labels = []

            with torch.no_grad():
                for imgs, labels in valid_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    with autocast(enabled=(device.type == "cuda")):
                        outputs = model(imgs)
                        loss = criterion(outputs, labels)
                    v_running_loss += loss.item() * imgs.size(0)
                    preds = torch.argmax(outputs, dim=1)
                    v_correct += (preds == labels).sum().item()
                    v_total += labels.size(0)
                    v_preds.extend(preds.cpu().numpy())
                    v_labels.extend(labels.cpu().numpy())

            val_loss = v_running_loss / len(valid_dataset)
            val_acc = 100.0 * v_correct / v_total
            val_losses.append(val_loss); val_accs.append(val_acc)

            print(f" --> Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(SAVE_DIR, f"best_model_epoch{epoch+1}_valloss{val_loss:.4f}.pth")
                torch.save(model.state_dict(), best_path)
                print(f"Saved best model to {best_path}")

        else:
            val_losses.append(float("nan")); val_accs.append(float("nan"))

    t1 = time.time()
    print(f"Training finished in {(t1 - t0) / 60.0:.2f} minutes")

    # ----------------- Plots -----------------
    epochs_range = range(1, NUM_EPOCHS + 1)
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, train_accs, label="Train Acc")
    if use_validation:
        plt.plot(epochs_range, val_accs, label="Val Acc")
    plt.title("Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.legend(); plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, train_losses, label="Train Loss")
    if use_validation:
        plt.plot(epochs_range, val_losses, label="Val Loss")
    plt.title("Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, precision_list, label="Precision")
    plt.plot(epochs_range, recall_list, label="Recall")
    plt.plot(epochs_range, f1_list, label="F1")
    plt.title("Metrics"); plt.xlabel("Epoch"); plt.ylabel("Score"); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.show()

    # ----------------- Final confusion matrix on validation (if available) -----------------
    if use_validation:
        cm = confusion_matrix(v_labels, v_preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=valid_base.classes)
        disp.plot(xticks_rotation=90)
        plt.title("Confusion Matrix (Validation)")
        plt.show()

    print("All done. Models saved in:", SAVE_DIR)
