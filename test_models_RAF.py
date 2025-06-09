import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from fer import FER
from deepface import DeepFace

IMAGE_ROOT_DIR = "Datasets/RAF-DB/DATASET/test/"
LABEL_CSV = "Datasets/RAF-DB/test_labels.csv"
MINI_XCEPTION_PATH = "_mini_XCEPTION.102-0.66.hdf5"

emotion_labels_model = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
rafdb_id_to_label = {
    1: "surprise",
    2: "fear",
    3: "disgust",
    4: "happy",
    5: "sad",
    6: "angry",
    7: "neutral"
}

fer_detector = FER(mtcnn=True)
mini_xception_model = tf.keras.models.load_model(MINI_XCEPTION_PATH, compile=False)

def preprocess_for_deep_models(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    resized = cv2.resize(image, (224, 224))
    return resized

def preprocess_for_mini_xception(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(image, (64, 64))
    img_pixels = resized.astype('float32') / 255.0
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels = np.expand_dims(img_pixels, axis=-1)
    return img_pixels

def predict_with_fer(image):
    result = fer_detector.detect_emotions(image)
    if result:
        return max(result[0]["emotions"], key=result[0]["emotions"].get)
    return None

def predict_with_deepface(image):
    result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
    return result[0]["dominant_emotion"]

def predict_with_mini_xception(input_img):
    preds = mini_xception_model.predict(input_img, verbose=0)
    return emotion_labels_model[np.argmax(preds)]

def main():
    labels_df = pd.read_csv(LABEL_CSV)
    labels_dict = dict(zip(labels_df["image"], labels_df["label"]))

    correct_fer = 0
    correct_df = 0
    correct_xcp = 0
    total = 0

    for subfolder in map(str, range(1, 8)):
        folder_path = os.path.join(IMAGE_ROOT_DIR, subfolder)
        if not os.path.isdir(folder_path):
            continue

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            image_path = os.path.join(folder_path, filename)
            image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image_gray is None:
                continue

            label_id = labels_dict.get(filename)
            true_label = rafdb_id_to_label.get(label_id)

            if true_label is None:
                continue

            image_color = preprocess_for_deep_models(image_gray)
            xcp_input = preprocess_for_mini_xception(image_gray)

            fer_pred = predict_with_fer(image_color)
            df_pred = predict_with_deepface(image_color)
            xcp_pred = predict_with_mini_xception(xcp_input)

            if fer_pred and fer_pred.lower() == true_label.lower():
                correct_fer += 1
            if df_pred and df_pred.lower() == true_label.lower():
                correct_df += 1
            if xcp_pred and xcp_pred.lower() == true_label.lower():
                correct_xcp += 1

            total += 1

    print("\nResultados finais:")
    print("Total imagens avaliadas:", total)
    print(f"FER accuracy:           {correct_fer / total * 100:.2f}%")
    print(f"DeepFace accuracy:      {correct_df / total * 100:.2f}%")
    print(f"mini_XCEPTION accuracy: {correct_xcp / total * 100:.2f}%")

if __name__ == "__main__":
    main()
