import torch
import torchvision.transforms as T
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
import torchattacks
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import os

def run_attack(image_name, epsilon, save_img):
    img_pil = Image.open(f"../imgs/{image_name}").convert("RGB")
    resize = T.Resize((224, 224))
    img_resized_pil = resize(img_pil)
    raw_orig_np = np.array(img_resized_pil)

    to_tensor = T.ToTensor()
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    input_tensor = normalize(to_tensor(img_resized_pil)).unsqueeze(0)

    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
    attack = torchattacks.PGD(model, eps=epsilon/255, alpha=2/255, steps=4)
    attack.set_normalization_used(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    with torch.no_grad():
        label = torch.argmax(model(input_tensor), dim=1)
    adv_tensor = attack(input_tensor, label)

    mean = input_tensor.new_tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
    std  = input_tensor.new_tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
    adv_dn = adv_tensor * std + mean
    adv_clamped = torch.clamp(adv_dn, 0.0, 1.0)

    adv_np = adv_clamped.squeeze(0).permute(1,2,0).cpu().numpy()
    adv_img = (adv_np * 255).astype(np.uint8)

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(raw_orig_np)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(adv_img)
    plt.title(f"PGD Adversarial\n(ε={epsilon/255}, α=2/255, steps=4)")
    plt.axis("off")
    plt.show()

    if save_img:
        name = os.path.splitext(image_name)[0]
        out_path = f'../attacked_imgs/{name}_attacked_torchattacks.png'
        cv2.imwrite(out_path, cv2.cvtColor(adv_img, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image_name', type=str, help="Image filename")
    parser.add_argument('epsilon', type=float, help="Set epsilon for attack")
    parser.add_argument('--save', action='store_true', help="Save attacked image")
    args = parser.parse_args()
    run_attack(args.image_name, args.epsilon, args.save)
