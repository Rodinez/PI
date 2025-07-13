import os
import cv2
import torch
import numpy as np
from model import MiniXception

IMAGE_ROOT_DIR = "attacked_datasets/self_model/foolbox_PGD_10.0/FER"
MODEL_PATH = "trained_models/mini_xception_final.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emotion_labels_model = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
label_to_index = {label: i for i, label in enumerate(emotion_labels_model)}

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

def count_total_images(root_dir):
    count = 0
    for label_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, label_name)
        if not os.path.isdir(folder_path):
            continue
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                count += 1
    return count

def main():
    correct = 0
    top2_correct = 0
    total = 0

    total_expected = count_total_images(IMAGE_ROOT_DIR)
    print(f"Total de imagens encontradas: {total_expected}")

    for true_label in os.listdir(IMAGE_ROOT_DIR):
        folder_path = os.path.join(IMAGE_ROOT_DIR, true_label)
        if not os.path.isdir(folder_path):
            continue

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue

            input_tensor = preprocess(image)
            pred, top2 = predict(input_tensor)

            if pred.lower() == true_label.lower():
                correct += 1
            elif true_label.lower() in [e.lower() for e in top2[1:]]:
                top2_correct += 1

            total += 1
            if total % 100 == 0:
                print(f"Processadas: {total}/{total_expected} imagens")

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
