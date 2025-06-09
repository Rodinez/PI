import os
import cv2
from matplotlib.image import pil_to_array
import numpy as np
import pandas as pd
import tensorflow as tf
from fer import FER
from deepface import DeepFace

IMAGE_DIR = "Datasets/RAF-DB/DATASET/test/3/"
LABEL_CSV = "Datasets/RAF-DB/test_labels.csv"
MINI_XCEPTION_PATH = "_mini_XCEPTION.102-0.66.hdf5"

emotion_labels = ["","surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]

fer_detector = FER(mtcnn=True)
mini_xception_model = tf.keras.models.load_model(MINI_XCEPTION_PATH, compile=False)

def predict_with_fer(image):
    result = fer_detector.detect_emotions(image)
    if result:
        return max(result[0]["emotions"], key=result[0]["emotions"].get)
    return None

def predict_with_deepface(image):
    result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
    return result[0]["dominant_emotion"]

def preprocess_for_mini_xception(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    img_pixels = resized.astype('float32')  
    img_pixels = np.expand_dims(img_pixels, axis=0)  
    img_pixels = np.expand_dims(img_pixels, axis=-1)  
    img_pixels /= 255.0
    return img_pixels

def predict_with_mini_xception(image):
    input_img = preprocess_for_mini_xception(image)
    preds = mini_xception_model.predict(input_img, verbose=0)
    return emotion_labels[np.argmax(preds)+1]

def main():
    labels_df = pd.read_csv(LABEL_CSV)
    labels_dict = dict(zip(labels_df["image"], labels_df["label"]))

    correct_fer = 0
    correct_df = 0
    correct_xcp = 0
    total = 0

    for filename in os.listdir(IMAGE_DIR):
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        image_path = os.path.join(IMAGE_DIR, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Não foi possível carregar a imagem: {filename}")
            continue

        label_id = labels_dict.get(filename)
        if label_id is None or label_id >= len(emotion_labels):
            print(f"Sem label ou label inválida para: {filename} (base: {filename})")
            continue

        true_label = emotion_labels[label_id]
        print(f"Processando: {filename} com label: {true_label}")

        fer_pred = predict_with_fer(image)
        df_pred = predict_with_deepface(image)
        xcp_pred = predict_with_mini_xception(image)

        if fer_pred and fer_pred.lower() == true_label.lower():
            correct_fer += 1
        if df_pred and df_pred.lower() == true_label.lower():
            correct_df += 1
        if xcp_pred and xcp_pred.lower() == true_label.lower():
            correct_xcp += 1

        total += 1

    if total == 0:
        print("Nenhuma imagem foi processada. Verifique os nomes no CSV e no diretório.")
        return

    print("\nResultados finais:")
    print("Total imagens avaliadas:", total)
    print(f"FER accuracy:           {correct_fer / total * 100:.2f}%")
    print(f"DeepFace accuracy:      {correct_df / total * 100:.2f}%")
    print(f"mini_XCEPTION accuracy: {correct_xcp / total * 100:.2f}%")


if __name__ == "__main__":
    main()
