import cv2
import numpy as np
import tensorflow as tf
from fer import FER
from deepface import DeepFace

# Caminhos
IMAGE_PATH = "Datasets/RAF-DB/DATASET/test/5/test_0001_aligned.jpg"
MINI_XCEPTION_PATH = "_mini_XCEPTION.102-0.66.hdf5"

# Labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Carregando modelos
fer_detector = FER(mtcnn=True)
mini_xception_model = tf.keras.models.load_model(MINI_XCEPTION_PATH, compile=False)

# Carrega imagem
image = cv2.imread(IMAGE_PATH)

print("\n========== FER ==========")
fer_result = fer_detector.detect_emotions(image)
if fer_result:
    fer_emotions = fer_result[0]['emotions']
    for emotion in emotion_labels:
        val = fer_emotions.get(emotion, 0.0)
        print(f"{emotion:10s}: {val:.4f}")
else:
    print("Nenhuma emoção detectada com FER.")

print("\n======= DeepFace =======")
df_result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)[0]
df_emotions = df_result['emotion']
for emotion in emotion_labels:
    val = df_emotions.get(emotion, 0.0)
    print(f"{emotion:10s}: {val:.4f}")

print("\n==== mini_XCEPTION ====")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray_image, (64, 64))  # Tamanho esperado do modelo
input_img = resized.astype('float32') / 255.0
input_img = np.expand_dims(input_img, axis=(0, -1))  # (1, 48, 48, 1)
preds = mini_xception_model.predict(input_img, verbose=0)[0]
for label, prob in zip(emotion_labels, preds):
    print(f"{label:10s}: {prob:.4f}")
