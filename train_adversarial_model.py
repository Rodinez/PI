"""
Treinamento com FER-2013 + RAF-DB
"""

import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from model import MiniXception  

# ============ CONFIGURAÇÕES ============

BATCH_SIZE = 32
NUM_EPOCHS = 10000
NUM_CLASSES = 7
INPUT_SHAPE = (1, 48, 48)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FER_PATH = 'Datasets/FER-2013/train'
RAF_PATH = 'Datasets/RAF-DB/DATASET/train'
BASE_PATH = './trained_models/emotion_models/'
VAL_SPLIT = 0.2
PATIENCE = 50
DATASET_NAME = "fer_raf"

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
label_to_index = {label: i for i, label in enumerate(EMOTION_LABELS)}
rafdb_id_to_label = {
    1: "surprise",
    2: "fear",
    3: "disgust",
    4: "happy",
    5: "sad",
    6: "angry",
    7: "neutral"
}

# ============ PRÉ-PROCESSAMENTO ============

def preprocess_input(img):
    img = img.astype('float32') / 255.0
    img = (img - 0.5) * 2.0
    return img

def load_fer2013(base_path):
    images, labels = [], []
    for emotion in EMOTION_LABELS:
        folder = os.path.join(base_path, emotion)
        if not os.path.isdir(folder):
            continue
        for file in os.listdir(folder):
            if not file.lower().endswith(('.jpg', '.png')):
                continue
            path = os.path.join(folder, file)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (48, 48))
            img = preprocess_input(img)
            images.append(img)
            labels.append(label_to_index[emotion])
    return np.array(images), np.array(labels)

def load_rafdb(image_root_dir):
    images, labels = [], []
    for folder_name in map(str, range(1, 8)):
        folder_path = os.path.join(image_root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        label_str = rafdb_id_to_label[int(folder_name)]
        label_idx = label_to_index[label_str]
        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            path = os.path.join(folder_path, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (48, 48))
            img = preprocess_input(img)
            images.append(img)
            labels.append(label_idx)
    return np.array(images), np.array(labels)


# ============ DATASET PYTORCH ============

class EmotionDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        img_uint8 = ((img + 1) * 127.5).astype(np.uint8)
        img_pil = Image.fromarray(img_uint8)

        if self.transform:
            img_pil = self.transform(img_pil)

        img_tensor = transforms.ToTensor()(img_pil)

        return img_tensor, label

    def __len__(self):
        return len(self.images)


# ============ AUGMENTAÇÃO ============

train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomHorizontalFlip()
])

# ============ MAIN ============

def main():
    print(f"Treinando com datasets: FER-2013 + RAF-DB")

    x_fer, y_fer = load_fer2013(FER_PATH)
    x_raf, y_raf = load_rafdb(RAF_PATH)

    x_total = np.concatenate([x_fer, x_raf], axis=0)
    y_total = np.concatenate([y_fer, y_raf], axis=0)

    x_train, x_val, y_train, y_val = train_test_split(
        x_total, y_total, test_size=VAL_SPLIT, stratify=y_total, random_state=42
    )

    class_sample_counts = np.bincount(y_train)
    class_weights = 1.0 / class_sample_counts
    samples_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)

    train_dataset = EmotionDataset(x_train, y_train, transform=train_transform)
    val_dataset = EmotionDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = MiniXception(num_classes=NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load("trained_models/mini_xception_adv_best.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint)
    print("Pesos pré-treinados carregados de mini_xception_adv_best.pth")

    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "final_conv" in name:
            param.requires_grad = True

    blocks_to_unfreeze = ["block4", "res4", "block3", "res3", "block2", "res2", "block1", "res1", "conv2", "conv1"]
    current_unfreeze_idx = 0
    UNFREEZE_EVERY = 5 

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = np.inf
    epochs_no_improve = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss, train_correct, total_train = 0.0, 0, 0

        if epoch % UNFREEZE_EVERY == 0 and current_unfreeze_idx < len(blocks_to_unfreeze):
            block_name = blocks_to_unfreeze[current_unfreeze_idx]
            print(f"-> -> Descongelando: {block_name}")
            for name, param in model.named_parameters():
                if block_name in name:
                    param.requires_grad = True
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-4, weight_decay=5e-4
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
            current_unfreeze_idx += 1

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch} - Training", leave=False):
            inputs, labels = inputs.to(DEVICE).long(), labels.to(DEVICE).long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_loss += loss.item() * inputs.size(0)
            total_train += inputs.size(0)

        train_acc = train_correct / total_train
        train_loss /= total_train

        # Validação
        model.eval()
        val_loss, val_correct, total_val = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch} - Validation", leave=False):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_loss += loss.item() * inputs.size(0)
                total_val += inputs.size(0)

        val_acc = val_correct / total_val
        val_loss /= total_val

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        scheduler.step(epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            model_path = f"{BASE_PATH}{DATASET_NAME}_transfer_mini_XCEPTION.{epoch:02d}-{val_acc:.2f}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"### Modelo salvo no epoch {epoch} com val_acc {val_acc:.4f} ###")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}!")
                break


if __name__ == "__main__":
    os.makedirs(BASE_PATH, exist_ok=True)
    main()
