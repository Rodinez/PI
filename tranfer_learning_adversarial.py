import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from model import MiniXception
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "trained_models/mini_xception_adv_final.pth"
SAVE_PATH = "trained_models/adversarial/"
os.makedirs(SAVE_PATH, exist_ok=True)

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
label_to_index = {label: i for i, label in enumerate(EMOTION_LABELS)}

EPSILONS = [1.0, 5.0, 10.0] 
EPSILONS = [e / 255.0 for e in EPSILONS] 

def preprocess_input(img):
    img = img.astype('float32') / 255.0
    img = (img - 0.5) * 2.0
    return img

class MultiFolderFERDataset(Dataset):
    def __init__(self, root_dirs, label_to_index, transform=None):
        self.samples = []
        self.label_to_index = label_to_index
        self.transform = transform

        for root_dir in root_dirs:
            for label in os.listdir(root_dir):
                label_path = os.path.join(root_dir, label)
                if not os.path.isdir(label_path):
                    continue
                class_index = self.label_to_index.get(label.lower())
                if class_index is None:
                    continue
                for fname in os.listdir(label_path):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(label_path, fname)
                        self.samples.append((img_path, class_index))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (48, 48))
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0) 
        if self.transform:
            image = torch.tensor(image, dtype=torch.float32)
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32)
        return image, label

train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=[0.0], std=[1.0]),  
])

val_transform = transforms.Compose([
    transforms.Normalize(mean=[0.0], std=[1.0]),
])

def main():
    print("==== Transfer Learning Adversarial Robust Training ====")
    print(f"Device: {DEVICE}")

    TRAIN_DIRS = ["Datasets/FER-2013/train"]  
    full_dataset = MultiFolderFERDataset(TRAIN_DIRS, label_to_index, transform=train_transform)

    model = MiniXception(num_classes=7).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    print("Modelo carregado:", MODEL_PATH)

    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    classifier = PyTorchClassifier(
        model=model,
        loss=loss_fn,
        optimizer=optimizer,
        input_shape=(1, 48, 48),
        nb_classes=7,
        clip_values=(-1.0, 1.0),
        device_type='gpu' if torch.cuda.is_available() else 'cpu',
    )

    attacks = []
    for eps in EPSILONS:
        attacks.append(FastGradientMethod(estimator=classifier, eps=eps))
        attacks.append(ProjectedGradientDescent(estimator=classifier, eps=eps))

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"[Epoch {epoch}]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            x_np = images.cpu().numpy()
            y_np = labels.cpu().numpy()

            x_batch_all = [images]
            y_batch_all = [labels]

            num_clean = len(x_np)
            num_attacks = len(attacks)
            num_per_attack = num_clean // num_attacks

            for attack in attacks:
                if num_per_attack == 0:
                    continue
                idx = np.random.choice(num_clean, num_per_attack, replace=False)
                x_sub = x_np[idx]
                y_sub = y_np[idx]

                x_adv = attack.generate(x_sub)
                x_adv_tensor = torch.tensor(x_adv, dtype=torch.float32).to(DEVICE)
                y_adv_tensor = torch.tensor(y_sub, dtype=torch.long).to(DEVICE)

                x_batch_all.append(x_adv_tensor)
                y_batch_all.append(y_adv_tensor)

            x_all = torch.cat(x_batch_all, dim=0)
            y_all = torch.cat(y_batch_all, dim=0)

            optimizer.zero_grad()
            outputs = model(x_all)
            loss = loss_fn(outputs, y_all)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            correct += (preds == y_all).sum().item()
            total_loss += loss.item() * x_all.size(0)
            total += x_all.size(0)

        acc = correct / total
        avg_loss = total_loss / total
        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")

        model_name = f"mini_XCEPTION_adv_e{epoch:02d}_acc{acc:.2f}.pth"
        torch.save(model.state_dict(), os.path.join(SAVE_PATH, model_name))
        print(f"Modelo salvo: {model_name}")

if __name__ == "__main__":
    main()