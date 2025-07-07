import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

IMAGE_ROOT_DIR = "attacked_datasets/foolbox_PGD_10.0/RAF"
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

mini_xception_model = tf.keras.models.load_model(MINI_XCEPTION_PATH, compile=False)

def preprocess_input(x, v2=True):
    x = x.astype('float32') / 255.0
    if v2:
        x = (x - 0.5) * 2.0
    return x

def preprocess_for_mini_xception(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(image, (64, 64))
    img_pixels = preprocess_input(resized)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels = np.expand_dims(img_pixels, axis=-1)
    return img_pixels

def predict_with_mini_xception(input_img):
    preds = mini_xception_model.predict(input_img, verbose=0)
    sorted_indices = np.argsort(preds[0])[::-1]
    top2 = [emotion_labels_model[i] for i in sorted_indices[:2]]
    return top2[0], top2

def main():
    labels_df = pd.read_csv(LABEL_CSV)
    labels_dict = dict(zip(labels_df["image"], labels_df["label"]))

    correct_xcp = 0
    second_xcp = 0
    total = 0

    image_paths = [
        (os.path.join(IMAGE_ROOT_DIR, fname), fname)
        for fname in os.listdir(IMAGE_ROOT_DIR)
        if fname.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]

    print(f"Processando {len(image_paths)} imagens...")

    for image_path, filename in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Erro ao carregar imagem: {filename}")
            continue

        label_id = labels_dict.get(filename)
        true_label = rafdb_id_to_label.get(label_id)
        if true_label is None:
            print(f"Label não encontrada para: {filename}")
            continue

        xcp_input = preprocess_for_mini_xception(image)
        xcp_pred, xcp_top2 = predict_with_mini_xception(xcp_input)

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
    print(f"Total de imagens processadas:         {total}")
    print(f"Acurácia mini_XCEPTION (top-1):       {correct_xcp / total * 100:.2f}%")
    print(f"Top-2 correta no mini_XCEPTION:       {second_xcp / total * 100:.2f}%")
    print(f"Acurácia Top(1-2) mini_XCEPTION:      {(correct_xcp + second_xcp) / total * 100:.2f}%")

if __name__ == "__main__":
    main()
