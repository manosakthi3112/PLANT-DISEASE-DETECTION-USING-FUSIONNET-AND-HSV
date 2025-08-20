
# 🌱 Plant Disease Detection using HSV + Hybrid CNN (DenseNet201 + EfficientNetV2-M)

This repository contains a **deep learning pipeline** for **plant disease classification**, combining **HSV color filtering** with a **fusion hybrid model (DenseNet201 + EfficientNetV2-M)**.  
It is trained on the **New Plant Diseases Dataset (Augmented)** and achieves **state-of-the-art performance**.

---

## 🚀 Features
- ✅ **HSV Filtering Preprocessing** – highlights green leaf regions, suppressing background noise.  
- ✅ **Hybrid CNN Architecture** – combines DenseNet201 + EfficientNetV2-M feature maps for robust classification.  
- ✅ **Mixed Precision Training** (AMP) for faster GPU training.  
- ✅ **Train & Validation Plots** – accuracy, loss, precision, recall, F1.  
- ✅ **Confusion Matrix Visualization** for validation set.  
- ✅ **Model Checkpointing** – saves best and per-epoch models automatically.  

---

## 📂 Dataset
We use the **[New Plant Diseases Dataset (Augmented)]**.  

**Folder structure (required):**


New Plant Diseases Dataset(Augmented)/
│── train/
│    ├── Apple\_\_\_Cedar\_apple\_rust/
│    ├── Apple\_\_\_healthy/
│    ├── ...
│
│── valid/
├── Apple\_\_\_Cedar\_apple\_rust/
├── Apple\_\_\_healthy/
├── ...


📌 [Dataset on Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)



## ⚙️ Installation

bash
# Clone repo
git clone https://github.com/your-username/plant-disease-hybrid.git
cd plant-disease-hybrid

# Create venv (recommended)
python -m venv venv
source venv/bin/activate    # On Linux/Mac
venv\Scripts\activate       # On Windows

# Install dependencies
pip install -r requirements.txt


## 🏋️ Training

bash
python model.py


Default config:

* **Batch size**: 8
* **Epochs**: 10
* **Learning rate**: 1e-4
* **Optimizer**: AdamW
* **Weight decay**: 1e-2

Trained models are saved in `saved_models/`.



## 📊 Results

* Achieved **99.14% validation accuracy** in just **10 epochs**.
* Example metrics plots:

| Metric    | Result  |
| --------- | ------- |
| Accuracy  | \~99.1% |
| Precision | \~99.0% |
| Recall    | \~99.0% |
| F1 Score  | \~99.0% |



## 🧠 Model Architecture

          ┌───────────────┐        ┌───────────────┐
          │   DenseNet201 │        │ EfficientNetV2-M │
          └───────┬───────┘        └───────────────┘
                  │                         │
          Global AvgPool             Global AvgPool
                  │                         │
                  └──────────┬──────────────┘
                             │
                        Concatenation
                             │
                        Fully Connected
                             │
                         Softmax Output


---

## 📌 Example Training Logs


[Epoch 1/10] Train Loss: 0.5432 | Train Acc: 85.62% | P: 0.8534 R: 0.8521 F1: 0.8527
 --> Val Loss: 0.3211 | Val Acc: 91.23%
Saved best model to saved_models/best_model_epoch1_valloss0.3211.pth

---

## 📈 Visualizations

* Training/Validation Accuracy & Loss
* Precision / Recall / F1 Curve
* Confusion Matrix on validation set

---

## 📑 Requirements

* Python 3.8+
* PyTorch ≥ 2.0
* Torchvision
* Scikit-learn
* Matplotlib
* OpenCV
* PIL

Install all with:

pip install torch torchvision scikit-learn matplotlib opencv-python pillow

---



