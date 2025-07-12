import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from model import MiniXception  
from datetime import datetime

# ================================
# CONFIG
# ================================
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "trained_models/mini_xception_adv_final.pth"
SAVE_PATH = "trained_models/emotion_models/"
os.makedirs(SAVE_PATH, exist_ok=True)

ADVERSARIAL_DIRS = [
    "attacked_datasets/cleverhans_FGSM_1.0/FER",
    "attacked_datasets/cleverhans_FGSM_5.0/FER",
    "attacked_datasets/cleverhans_FGSM_10.0/FER",
    "attacked_datasets/cleverhans_PGD_1.0/FER",
    "attacked_datasets/cleverhans_PGD_5.0/FER",
    "attacked_datasets/cleverhans_PGD_10.0/FER",
    "attacked_datasets/foolbox_Deepfool_1.0/FER",
    "attacked_datasets/foolbox_Deepfool_5.0/FER",
    "attacked_datasets/foolbox_Deepfool_10.0/FER",
]

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
label_to_index = {label: i for i, label in enumerate(EMOTION_LABELS)}

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
        image = torch.from_numpy(image).float()

        if self.transform:
            image = self.transform(image)

        return image, label


train_transform = transforms.Compose([
    transforms.RandomRotation(10),                  
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)), 
    transforms.RandomHorizontalFlip(),             
    transforms.ToTensor(),                        
    transforms.Normalize(mean=[0.5], std=[0.5])  
])

def main():
    print("==== Transfer Learning Adversarial ====")
    print(f"Device: {DEVICE}")
    print(f"Dataset(s): {ADVERSARIAL_DIRS}")
    print("---------------------------------------")

    dataset = MultiFolderFERDataset(ADVERSARIAL_DIRS, label_to_index, transform=train_transform)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = MiniXception(num_classes=7).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    print("Modelo carregado:", MODEL_PATH)

    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"[Epoch {epoch}]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_loss += loss.item() * images.size(0)
            total += images.size(0)

        acc = correct / total
        avg_loss = total_loss / total
        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")

        # Salvar modelo a cada época
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"trained_models/adversarial/mini_XCEPTION_adversarial_e{epoch:02d}_acc{acc:.2f}_{timestamp}.pth"
        torch.save(model.state_dict(), os.path.join(SAVE_PATH, model_name))
        print(f"Modelo salvo: {model_name}")

if __name__ == "__main__":
    main()
