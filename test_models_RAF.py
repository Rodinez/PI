import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from fer import FER
from deepface import DeepFace
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('deepface').setLevel(logging.ERROR)

IMAGE_ROOT_DIR = "Datasets/RAF-DB/DATASET/test/"
LABEL_CSV = "Datasets/RAF-DB/test_labels.csv"
MINI_XCEPTION_PATH = "_mini_XCEPTION.102-0.66.hdf5"

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

fer_detector = FER(mtcnn=True)
mini_xception_model = tf.keras.models.load_model(MINI_XCEPTION_PATH, compile=False)

def preprocess_for_deep_models(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    h, w = image.shape[:2]
    pad_h = int(0.2 * h)
    pad_w = int(0.2 * w)

    padded = cv2.copyMakeBorder(
        image,
        top=pad_h,
        bottom=pad_h,
        left=pad_w,
        right=pad_w,
        borderType=cv2.BORDER_REFLECT
    )

    scale = 224 / min(padded.shape[0], padded.shape[1])
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    resized = cv2.resize(padded, (224, 224), interpolation=interp)
    return padded

def preprocess_for_mini_xception(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)

    resized = cv2.resize(image, (64, 64))
    img_pixels = resized.astype('float32') / 255.0
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels = np.expand_dims(img_pixels, axis=-1)
    return img_pixels

def predict_with_fer(image):
    try:
        result = fer_detector.detect_emotions(image)
        if result:
            emotions = result[0]["emotions"]
            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
            top_emotions = [e[0] for e in sorted_emotions[:2]]
            return top_emotions[0], top_emotions
    except Exception:
        pass
    return None, []

def predict_with_deepface(image):
    try:
        result = DeepFace.analyze(
            image,
            actions=['emotion'],
            detector_backend='mtcnn',
            enforce_detection=False,
            silent=True
        )
        emotions = result[0]["emotion"]
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        top_emotions = [e[0].lower() for e in sorted_emotions[:2]]
        return top_emotions[0], top_emotions
    except Exception:
        return "neutral", []

def predict_with_mini_xception(input_img):
    try:
        preds = mini_xception_model.predict(input_img, verbose=0)
        sorted_indices = np.argsort(preds[0])[::-1]
        top2 = [emotion_labels_model[i] for i in sorted_indices[:2]]
        return top2[0], top2
    except Exception:
        return "neutral", []

def main():
    labels_df = pd.read_csv(LABEL_CSV)
    labels_dict = dict(zip(labels_df["image"], labels_df["label"]))

    correct_fer, correct_df, correct_xcp = 0, 0, 0
    second_fer, second_df, second_xcp = 0, 0, 0
    total, fer_failures = 0, 0
    face_detected_count = 0

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
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            continue

        label_id = labels_dict.get(filename)
        true_label = rafdb_id_to_label.get(label_id)
        if true_label is None:
            continue

        image_color = preprocess_for_deep_models(image)
        xcp_input = preprocess_for_mini_xception(image)

        fer_pred, fer_top2 = predict_with_fer(image_color)
        df_pred, df_top2 = predict_with_deepface(image_color)
        xcp_pred, xcp_top2 = predict_with_mini_xception(xcp_input)

        if fer_pred is None:
            fer_failures += 1
            fer_pred = ""

        if fer_pred.lower() == true_label.lower():
            correct_fer += 1
        elif true_label.lower() in [e.lower() for e in fer_top2[1:]]:
            second_fer += 1

        if df_pred.lower() == true_label.lower():
            correct_df += 1
        elif true_label.lower() in [e.lower() for e in df_top2[1:]]:
            second_df += 1

        if xcp_pred.lower() == true_label.lower():
            correct_xcp += 1
        elif true_label.lower() in [e.lower() for e in xcp_top2[1:]]:
            second_xcp += 1

        total += 1

        if total % 100 == 0:
            print(f"Processadas: {total}/{len(image_paths)} imagens")

    if total == 0:
        print("Nenhuma imagem foi processada com sucesso.")
        return

    print("\n" + "=" * 50)
    print("Resultados finais:")
    print("=" * 50)
    print(f"Total de imagens com rosto detectado: {face_detected_count}")
    print(f"Total de imagens processadas:         {total}")
    print(f"Taxa de falha do FER:                 {fer_failures / total * 100:.2f}%")
    print(f"Acurácia FER (top-1):                 {correct_fer / total * 100:.2f}%")
    print(f"Top-2 correta no FER:                 {second_fer / total * 100:.2f}%")
    print(f"Acurácia DeepFace (top-1):            {correct_df / total * 100:.2f}%")
    print(f"Top-2 correta no DeepFace:            {second_df / total * 100:.2f}%")
    print(f"Acurácia mini_XCEPTION (top-1):       {correct_xcp / total * 100:.2f}%")
    print(f"Top-2 correta no mini_XCEPTION:       {second_xcp / total * 100:.2f}%")

if __name__ == "__main__":
    main()
