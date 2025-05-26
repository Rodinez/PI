import tensorflow as tf
import foolbox as fb
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import os

def run_attack(image_name, epsilon, save_img):
    img = cv2.imread(f"../imgs/{image_name}")
    img_resized = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
    image_tensor = tf.convert_to_tensor(img_resized[None, ...], dtype=tf.float32)

    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    fmodel = fb.TensorFlowModel(model, bounds=(0, 1))

    preds = model(image_tensor).numpy()
    label = int(np.argmax(preds, axis=1)[0])
    labels = np.array([label], dtype=np.int64)

    attack = fb.attacks.FGSM()
    epsilons = [epsilon]

    advs, clipped, success = attack(fmodel, image_tensor, labels, epsilons=epsilons)
    adv_img = advs[0].numpy()[0] * 255.0
    adv_img = np.clip(adv_img, 0, 255).astype(np.uint8)

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(adv_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Adversarial (ε={epsilon})")
    plt.axis("off")
    plt.show()

    if save_img:
        name = os.path.splitext(image_name)[0]
        out_path = f'../attacked_imgs/{name}_attacked_foolbox.png'
        cv2.imwrite(out_path, adv_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image_name', type=str, help="Image filename")
    parser.add_argument('epsilon', type=float, help="Set epsilon for attack")
    parser.add_argument('--save', action='store_true', help="Save attacked image")
    args = parser.parse_args()
    run_attack(args.image_name, args.epsilon, args.save)
