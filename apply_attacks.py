import os
import argparse
import numpy as np
from PIL import Image
import cv2
import csv
import tensorflow as tf
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import TensorFlowV2Classifier
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
import foolbox as fb

def save_adv_tensor(tensor, out_path):
    adv_np = tf.squeeze(tensor)
    if isinstance(adv_np, tf.Tensor):
        adv_np = adv_np.numpy()
    adv_img = np.clip(adv_np * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(out_path, adv_img)

def main(args):
    eps = args.epsilon / 255.0

    keras_model = tf.keras.models.load_model('_mini_XCEPTION.102-0.66.hdf5', compile=False)
    keras_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    loss_object = tf.keras.losses.CategoricalCrossentropy()
    classifier = TensorFlowV2Classifier(
        model=keras_model,
        nb_classes=7,
        input_shape=(64, 64, 1),
        loss_object=loss_object,
        clip_values=(0.0, 1.0)
    )

    fmodel = fb.TensorFlowModel(keras_model, bounds=(0, 1))

    dataset_paths = {
        'FER': 'Datasets/FER-2013/test/',
        'RAF': 'Datasets/RAF-DB/DATASET/test/'
    }

    fer_label_map = {
        'angry': 0,
        'disgust': 1,
        'fear': 2,
        'happy': 3,
        'sad': 4,
        'surprise': 5,
        'neutral': 6
    }

    raf_labels = {}
    if 'RAF' in args.dataset or args.dataset == 'both':
        with open('Datasets/RAF-DB/test_labels.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                raf_labels[row['image']] = int(row['label'])

    datasets_to_use = ['FER', 'RAF'] if args.dataset == 'both' else [args.dataset]

    for dataset in datasets_to_use:
        print(f"\n=== Processing dataset: {dataset} ===")
        dataset_path = dataset_paths[dataset]

        for subdir, _, files in os.walk(dataset_path):
            for img_name in files:
                if not img_name.lower().endswith(('.jpg', '.png')):
                    continue

                img_path = os.path.join(subdir, img_name)

                try:
                    img = Image.open(img_path).convert("L").resize((64, 64))
                except Exception as e:
                    print(f"Failed to load image {img_path}: {e}")
                    continue

                img_np = np.asarray(img, dtype=np.float32) / 255.0
                x = np.expand_dims(img_np, axis=(0, -1))

                if dataset == 'FER':
                    label_name = os.path.basename(os.path.dirname(img_path)).lower()
                    label_idx = fer_label_map.get(label_name)
                    if label_idx is None:
                        print(f"Unknown FER label {label_name} for {img_path}")
                        continue
                else:
                    label_idx = raf_labels.get(img_name)
                    if label_idx is None:
                        print(f"Label not found for RAF image {img_name}")
                        continue

                # ART
                y_onehot = np.zeros((1, 7))
                y_onehot[0, label_idx] = 1.0

                # ART FGSM
                adv = FastGradientMethod(estimator=classifier, eps=eps).generate(x=x, y=y_onehot)
                out_dir = f'attacked_datasets/art_FGSM_{args.epsilon}/{dataset}'
                os.makedirs(out_dir, exist_ok=True)
                save_adv_tensor(adv[0], os.path.join(out_dir, img_name))

                # ART PGD
                adv = ProjectedGradientDescent(estimator=classifier, eps=eps, eps_step=2/255, max_iter=40).generate(x=x, y=y_onehot)
                out_dir = f'attacked_datasets/art_PGD_{args.epsilon}/{dataset}'
                os.makedirs(out_dir, exist_ok=True)
                save_adv_tensor(adv[0], os.path.join(out_dir, img_name))

                # CleverHans
                x_tf = tf.convert_to_tensor(x)

                # CleverHans FGSM
                adv = fast_gradient_method(keras_model, x_tf, eps=eps, norm=np.inf, targeted=False)
                adv_np = adv.numpy()
                out_dir = f'attacked_datasets/cleverhans_FGSM_{args.epsilon}/{dataset}'
                os.makedirs(out_dir, exist_ok=True)
                save_adv_tensor(adv_np, os.path.join(out_dir, img_name))

                # CleverHans PGD
                adv = projected_gradient_descent(keras_model, x_tf, eps=eps, eps_iter=min(eps, 1/255), nb_iter=40, norm=np.inf, targeted=False)
                adv_np = adv.numpy()
                out_dir = f'attacked_datasets/cleverhans_PGD_{args.epsilon}/{dataset}'
                os.makedirs(out_dir, exist_ok=True)
                save_adv_tensor(adv_np, os.path.join(out_dir, img_name))

                # Foolbox
                x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
                y_fb_tf = tf.convert_to_tensor([label_idx], dtype=tf.int32)

                # Foolbox FGSM
                atk = fb.attacks.FGSM()
                _, clipped_adv, _ = atk(fmodel, x_tf, y_fb_tf, epsilons=eps)
                out_dir = f'attacked_datasets/foolbox_FGSM_{args.epsilon}/{dataset}'
                os.makedirs(out_dir, exist_ok=True)
                save_adv_tensor(clipped_adv, os.path.join(out_dir, img_name))

                # Foolbox PGD
                atk = fb.attacks.LinfPGD(steps=40)
                _, clipped_adv, _ = atk(fmodel, x_tf, y_fb_tf, epsilons=eps)
                out_dir = f'attacked_datasets/foolbox_PGD_{args.epsilon}/{dataset}'
                os.makedirs(out_dir, exist_ok=True)
                save_adv_tensor(clipped_adv, os.path.join(out_dir, img_name))

                # Foolbox Deepfool
                atk = fb.attacks.deepfool.L2DeepFoolAttack(steps=50)
                _, clipped_adv, _ = atk(fmodel, x_tf, y_fb_tf, epsilons=eps)
                out_dir = f'attacked_datasets/foolbox_Deepfool_{args.epsilon}/{dataset}'
                os.makedirs(out_dir, exist_ok=True)
                save_adv_tensor(clipped_adv, os.path.join(out_dir, img_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['FER', 'RAF', 'both'], required=True,
                        help='Escolha o dataset: FER, RAF ou both')
    parser.add_argument('--epsilon', type=float, required=True,
                        help='Valor de epsilon (em escala 0-255)')
    args = parser.parse_args()
    main(args)
