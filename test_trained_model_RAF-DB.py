import os
import cv2
import torch
import numpy as np
import pandas as pd
from model import MiniXception

IMAGE_ROOT_DIR = "Datasets/RAF-DB/DATASET/test"
LABEL_CSV = "Datasets/RAF-DB/test_labels.csv"
MODEL_PATH = "trained_models/mini_XCEPTION_adv_e19_acc0.52.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emotion_labels_model = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
rafdb_id_to_label = {
    1: "surprise",
    2: "fear",
    3: "disgust",
    4: "happy",
    5: "sad",
    6: "angry",
    7: "neutral"
}

def preprocess_input(x, v2=True):
    x = x.astype('float32') / 255.0
    if v2:
        x = (x - 0.5) * 2.0
    return x

def preprocess(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(image, (48, 48))  
    img = preprocess_input(resized)
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    return img.to(DEVICE)

model = MiniXception(num_classes=7).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def predict(input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        sorted_indices = np.argsort(probs)[::-1]
        top2 = [emotion_labels_model[i] for i in sorted_indices[:2]]
        return top2[0], top2

def main():
    labels_df = pd.read_csv(LABEL_CSV)
    labels_dict = dict(zip(labels_df["image"], labels_df["label"]))

    correct = 0
    top2_correct = 0
    total = 0

    image_paths = []
    for subfolder in map(str, range(1, 8)):
        folder_path = os.path.join(IMAGE_ROOT_DIR, subfolder)
        if not os.path.isdir(folder_path):
            continue
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_paths.append((os.path.join(folder_path, filename), filename))

    print(f"Processando {len(image_paths)} imagens...")

    for image_path, filename in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            continue

        label_id = labels_dict.get(filename)
        true_label = rafdb_id_to_label.get(label_id)
        if true_label is None:
            continue

        input_tensor = preprocess(image)
        pred, top2 = predict(input_tensor)

        if pred.lower() == true_label.lower():
            correct += 1
        elif true_label.lower() in [e.lower() for e in top2[1:]]:
            top2_correct += 1

        total += 1
        if total % 100 == 0:
            print(f"Processadas: {total}/{len(image_paths)} imagens")

    if total == 0:
        print("Nenhuma imagem foi processada com sucesso.")
        return

    print("\n" + "=" * 50)
    print("Resultados finais:")
    print("=" * 50)
    print(f"Total de imagens processadas:         {total}")
    print(f"Acurácia mini_XCEPTION (top-1):       {correct / total * 100:.2f}%")
    print(f"Top-2 correta no mini_XCEPTION:       {top2_correct / total * 100:.2f}%")
    print(f"Acurácia Top(1-2) mini_XCEPTION:      {(correct + top2_correct) / total * 100:.2f}%")

if __name__ == "__main__":
    main()
