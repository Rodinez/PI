import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
import cv2
from sklearn.model_selection import train_test_split
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
from model import MiniXception 

BATCH_SIZE = 64
NUM_EPOCHS = 100
NUM_CLASSES = 7
INPUT_SHAPE = (1, 64, 64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_PATH = './trained_models/'
FER_PATH = 'Datasets/FER-2013/train'
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
label_to_index = {label: i for i, label in enumerate(EMOTION_LABELS)}

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
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (64, 64))
            img = preprocess_input(img)
            images.append(img)
            labels.append(label_to_index[emotion])

    images = np.array(images)
    labels = np.array(labels)
    return train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)

class FERDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        img = self.images[idx]
        img = np.expand_dims(img, axis=0)
        if self.transform:
            img = self.transform(torch.from_numpy(img).float())
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.images)

def main():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),  
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), 
        transforms.Normalize(mean=[0.0], std=[1.0])  
    ])

    val_transform = transforms.Normalize(mean=[0.0], std=[1.0])
    
    x_train, x_val, y_train, y_val = load_fer2013(FER_PATH)

    train_dataset = FERDataset(x_train, y_train, transform=train_transform)
    val_dataset = FERDataset(x_val, y_val, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = MiniXception(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    loss_fn = nn.CrossEntropyLoss()

    classifier = PyTorchClassifier(
        model=model,
        loss=loss_fn,
        optimizer=optimizer,
        input_shape=INPUT_SHAPE,
        nb_classes=NUM_CLASSES,
        clip_values=(-1.0, 1.0),
        device_type="gpu" if torch.cuda.is_available() else "cpu"
    )

    attack = ProjectedGradientDescent(
        estimator=classifier,
        eps=0.003,
    )

    best_val_loss = float('inf')
    os.makedirs(BASE_PATH, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        total_correct = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = loss_fn(outputs, y_batch.long())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == y_batch).sum().item()

        train_acc = total_correct / len(train_dataset)
        train_loss = total_loss / len(train_loader)

        # Avaliação
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                outputs = model(x_batch)
                loss = loss_fn(outputs, y_batch.long())
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == y_batch).sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / len(val_dataset)

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(BASE_PATH, "mini_xception_adv_best.pth"))
            print(f"### Modelo salvo no epoch {epoch+1} com val_loss {val_loss:.4f} ###")

    print("\nAvaliando no conjunto limpo:")
    preds_clean = classifier.predict(x_val[:, np.newaxis, :, :])
    clean_acc = np.mean(np.argmax(preds_clean, axis=1) == y_val)
    print(f"Acurácia limpa: {clean_acc*100:.2f}%")

    torch.save(model.state_dict(), os.path.join(BASE_PATH, "mini_xception_adv_final.pth"))
    print(f"Modelo salvo em {BASE_PATH}")

if __name__ == "__main__":
    main()
