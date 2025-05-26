import cv2
from deepface import DeepFace
import argparse

def analyze_emotions(image_name, image_name_attacked):
    orig_img = cv2.imread(f"imgs/{image_name}")
    adv_img = cv2.imread(f"attacked_imgs/{image_name_attacked}")

    orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    adv_rgb = cv2.cvtColor(adv_img, cv2.COLOR_BGR2RGB)


    result_orig = DeepFace.analyze(orig_rgb, actions=['emotion'], enforce_detection=False)[0]
    print(f"Emoção dominante (Original): {result_orig['dominant_emotion']}")
    print(f"Emoções (Original): {result_orig['emotion']}")

    result_adv = DeepFace.analyze(adv_rgb, actions=['emotion'], enforce_detection=False)[0]
    print(f"Emoção dominante (Atacada): {result_adv['dominant_emotion']}")
    print(f"Emoções (Atacada): {result_adv['emotion']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Análise de emoções com DeepFace em imagem original e atacada")
    parser.add_argument("image_name", type=str, help="Nome do arquivo da imagem")
    parser.add_argument("image_name_attacked", type=str, help="Nome do arquivo da imagem atacada")

    args = parser.parse_args()
    analyze_emotions(args.image_name, args.image_name_attacked)
