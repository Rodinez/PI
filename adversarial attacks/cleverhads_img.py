import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
import argparse
import os

def run_attack(image_name, epsilon, save_img):
    img_bgr = cv2.imread(f"../imgs/{image_name}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224)).astype(np.float32)

    input_tensor = tf.convert_to_tensor(img_resized[None, ...], dtype=tf.float32)
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(input_tensor)

    preds = model(preprocessed)
    label = tf.argmax(preds, axis=1)

    adv_tensor = fast_gradient_method(model_fn=model,
                                      x=preprocessed,
                                      eps=epsilon,
                                      norm=np.inf,
                                      y=label,
                                      targeted=False)

    adv = adv_tensor[0].numpy()
    adv_img = (adv + 1.0) * 127.5
    adv_img_uint8 = np.clip(adv_img, 0, 255).astype(np.uint8)

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(img_rgb)
    plt.title("Original")
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow(adv_img_uint8)
    plt.title(f"Adversarial (ε={epsilon})")
    plt.axis("off")
    plt.show()

    if save_img:
        name = os.path.splitext(image_name)[0]
        out_path = f'../attacked_imgs/{name}_attacked_cleverhans.png'
        cv2.imwrite(out_path, cv2.cvtColor(adv_img_uint8, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image_name', type=str, help="Image filename")
    parser.add_argument('epsilon', type=float, help="Set epsilon for attack")
    parser.add_argument('--save', action='store_true', help="Save attacked image")
    args = parser.parse_args()
    run_attack(args.image_name, args.epsilon, args.save)
