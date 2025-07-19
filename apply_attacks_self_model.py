import os
import argparse
import numpy as np
from PIL import Image
import cv2
import csv
import torch
from torchvision import transforms
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
import foolbox as fb
from model import MiniXception

def preprocess(img):
    img = np.array(img).astype(np.float32)
    img = cv2.resize(img, (48, 48))
    img = img / 255.0
    img = (img - 0.5) * 2.0
    return img

def save_adv_tensor(tensor, out_path):
    adv_np = tensor.detach().cpu().numpy().squeeze()
    adv_np = ((adv_np / 2.0) + 0.5) * 255
    adv_np = np.clip(adv_np, 0, 255).astype(np.uint8)
    cv2.imwrite(out_path, adv_np)

def save_with_label_dir(base_out_dir, dataset, label_name, img_name, adv_tensor):
    if dataset == 'FER':
        out_dir = os.path.join(base_out_dir, dataset, label_name)
    else:
        out_dir = os.path.join(base_out_dir, dataset)
    os.makedirs(out_dir, exist_ok=True)
    save_adv_tensor(adv_tensor, os.path.join(out_dir, img_name))


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eps = args.epsilon / 255.0

    model = MiniXception(num_classes=7).to(device)
    model.load_state_dict(torch.load('trained_models/mini_xception_final.pth', map_location=device))
    model.eval()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) 

    classifier = PyTorchClassifier(
        model=model,
        loss=loss_fn,
        optimizer=optimizer,
        input_shape=(1, 48, 48),
        nb_classes=7,
        clip_values=(-1.0, 1.0),
        device_type='gpu' if device.type == 'cuda' else 'cpu'
    )

    fmodel = fb.PyTorchModel(model, bounds=(-1, 1), device=device)

    dataset_paths = {
        'FER': 'Datasets/FER-2013/test',
        'RAF': 'Datasets/RAF-DB/DATASET/test'
    }

    fer_label_map = {
        'angry': 0,
        'disgust': 1,
        'fear': 2,
        'happy': 3,
        'neutral': 4,
        'sad': 5,
        'surprise': 6
    }

    raf_labels = {}
    if 'RAF' in args.dataset or args.dataset == 'both':
        with open('Datasets/RAF-DB/test_labels.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                raf_labels[row['image']] = int(row['label']) - 1

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
                    img = Image.open(img_path).convert("L")
                except Exception as e:
                    print(f"Failed to load image {img_path}: {e}")
                    continue

                img_proc = preprocess(img)
                x = torch.tensor(img_proc).unsqueeze(0).unsqueeze(0).to(device) 

                label_name = None
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

                y = torch.tensor([label_idx], device=device)

                # One-hot para ART
                y_onehot = np.zeros((1, 7))
                y_onehot[0, label_idx] = 1.0

                # ART FGSM
                adv = FastGradientMethod(classifier, eps=eps).generate(x=x.cpu().numpy(), y=y_onehot)
                save_with_label_dir(f'attacked_datasets/self_model/art_FGSM_{args.epsilon}', dataset, label_name, img_name, torch.tensor(adv[0]))

                # ART PGD
                adv = ProjectedGradientDescent(classifier, eps=eps, eps_step=2/255, max_iter=40).generate(x=x.cpu().numpy(), y=y_onehot)
                save_with_label_dir(f'attacked_datasets/self_model/art_PGD_{args.epsilon}', dataset, label_name, img_name, torch.tensor(adv[0]))

                # CleverHans FGSM
                adv_ch = fast_gradient_method(model, x, eps=eps, norm=np.inf, targeted=False)
                save_with_label_dir(f'attacked_datasets/self_model/cleverhans_FGSM_{args.epsilon}', dataset, label_name, img_name, adv_ch)

                # CleverHans PGD
                adv_ch = projected_gradient_descent(model, x, eps=eps, eps_iter=min(eps, 1/255), nb_iter=40, norm=np.inf, targeted=False)
                save_with_label_dir(f'attacked_datasets/self_model/cleverhans_PGD_{args.epsilon}', dataset, label_name, img_name, adv_ch)

                # Foolbox FGSM
                atk = fb.attacks.FGSM()
                _, clipped_adv, _ = atk(fmodel, x, y, epsilons=eps)
                save_with_label_dir(f'attacked_datasets/self_model/foolbox_FGSM_{args.epsilon}', dataset, label_name, img_name, clipped_adv)

                # Foolbox PGD
                atk = fb.attacks.LinfPGD(steps=40)
                _, clipped_adv, _ = atk(fmodel, x, y, epsilons=eps)
                save_with_label_dir(f'attacked_datasets/self_model/foolbox_PGD_{args.epsilon}', dataset, label_name, img_name, clipped_adv)

                # Foolbox Deepfool
                atk = fb.attacks.deepfool.L2DeepFoolAttack(steps=50)
                _, clipped_adv, _ = atk(fmodel, x, y, epsilons=eps)
                save_with_label_dir(f'attacked_datasets/self_model/foolbox_Deepfool_{args.epsilon}', dataset, label_name, img_name, clipped_adv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['FER', 'RAF', 'both'], required=True, help='Escolha o dataset: FER, RAF ou both')
    parser.add_argument('--epsilon', type=float, required=True, help='Valor de epsilon (0-255)')
    args = parser.parse_args()
    main(args)
