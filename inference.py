# inference.py
import os
import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ----------------- User config -----------------
MODEL_PATH = r"C:\Users\manos\Music\plant\saved_models\best_model_epoch7_valloss0.0281.pth"   # <-- change to your saved model path
DATASET_ROOT = r"C:\Users\manos\Music\plant\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"
IMAGE_PATH = r"C:\Users\manos\Music\plant\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid\Apple___Cedar_apple_rust\0cd24b0c-0a9d-483f-8734-5c08988e029f___FREC_C.Rust 3762_newPixel25.JPG" # <-- change to your test image
folder_name = os.path.basename(os.path.dirname(IMAGE_PATH))
# HSV ranges (same as training)
GREEN_LOWER = np.array([20, 40, 40], dtype=np.uint8)
GREEN_UPPER = np.array([90, 255, 255], dtype=np.uint8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Hybrid Model ----------------
class DenseEfficient(nn.Module):
    def __init__(self, num_classes, use_pretrained=False):
        super().__init__()
        densenet = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1 if use_pretrained else None)
        self.densenet_features = densenet.features  

        effnet = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1 if use_pretrained else None)
        self.effnet_features = effnet.features  

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(1920 + 1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        f1 = self.pool(self.densenet_features(x))
        f1 = torch.flatten(f1, 1)
        f2 = self.pool(self.effnet_features(x))
        f2 = torch.flatten(f2, 1)
        f = torch.cat([f1, f2], dim=1)
        return self.classifier(f)

# ---------------- Inference Utils ----------------
def apply_hsv_mask(image_path, lower_hsv, upper_hsv):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    original_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return original_rgb, result_rgb

def load_model(model_path, num_classes):
    model = DenseEfficient(num_classes=num_classes, use_pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model

# ---------------- Main Inference ----------------
if __name__ == "__main__":
    # class labels from training set
    base_dataset = datasets.ImageFolder(DATASET_ROOT)
    idx_to_class = {v: k for k, v in base_dataset.class_to_idx.items()}
    num_classes = len(idx_to_class)

    # Load model
    model = load_model(MODEL_PATH, num_classes)

    # Apply HSV preprocessing
    orig_img, hsv_img = apply_hsv_mask(IMAGE_PATH, GREEN_LOWER, GREEN_UPPER)

    # Transform for model
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    pil_img = Image.fromarray(hsv_img)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        pred_class = torch.argmax(outputs, dim=1).item()

    predicted_label = idx_to_class[pred_class]
    print(f"Predicted Class: {predicted_label}")

    # Show results
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(orig_img)
    plt.title(f"Original Image {folder_name}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(hsv_img)
    plt.title(f"Predicted: {predicted_label}")
    plt.axis("off")

    plt.show()
