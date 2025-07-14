import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F

from model import MiniXception

# ----------------------------
# 1) CONFIGURAÇÕES GLOBAIS
# ----------------------------
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
ATTACK_BASE_DIR = os.path.join(BASE_DIR, "attacked_datasets", "self_model")
OUTPUT_FILE     = os.path.join(BASE_DIR, "phi_results.txt")
MODEL_PATH      = os.path.join(BASE_DIR, "trained_models", "mini_xception_final.pth")
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Map de diretórios limpos por dataset
CLEAN_DIR_MAP = {
    'RAF': os.path.join(BASE_DIR, "Datasets", "RAF-DB", "DATASET", "test"),
    'FER': os.path.join(BASE_DIR, "Datasets", "FER-2013", "test")
}

# Labels (sem mudanças)
emotion_labels_model = [
    "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"
]
rafdb_id_to_label = {
    1: "surprise", 2: "fear", 3: "disgust",
    4: "happy",    5: "sad",  6: "angry",   7: "neutral"
}

# ----------------------------
# 2) MÉTRICA Φ (KL-DIVERGENCE)
# ----------------------------
def phi_kl(logits_clean: torch.Tensor,
           logits_adv: torch.Tensor,
           dim: int = -1,
           eps: float = 1e-8) -> torch.Tensor:
    p = F.softmax(logits_clean, dim=dim).clamp(min=eps)
    q = F.softmax(logits_adv,   dim=dim).clamp(min=eps)
    kl = torch.sum(p * (p.log() - q.log()), dim=dim)
    return kl.mean()

# ----------------------------
# 3) PREPROCESSAMENTO E PREDICT
# ----------------------------
def preprocess_input(x: np.ndarray, v2: bool = True) -> np.ndarray:
    x = x.astype("float32") / 255.0
    return (x - 0.5) * 2.0 if v2 else x


def preprocess(image: np.ndarray) -> torch.Tensor:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (48, 48))
    image = preprocess_input(image)
    tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0)
    return tensor.to(DEVICE)


def predict_logits(input_tensor: torch.Tensor):
    with torch.no_grad():
        logits = model(input_tensor)
        probs  = torch.softmax(logits, dim=1)
        idxs   = torch.argsort(probs, dim=1, descending=True)[0]
        top2   = [emotion_labels_model[i] for i in idxs.cpu().numpy()[:2]]
        return logits, top2[0], top2

# ----------------------------
# 4) EXECUÇÃO POR LOTE
# ----------------------------
def run_for_dataset(attack_name: str, ds_key: str) -> float:
    clean_dir = CLEAN_DIR_MAP[ds_key]
    adv_dir   = os.path.join(ATTACK_BASE_DIR, attack_name, ds_key)

    # Carrega lista clean
    clean_list = []
    for root, _, files in os.walk(clean_dir):
        for fn in files:
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                clean_list.append((os.path.join(root, fn), fn))

    adv_map = {}
    for root, _, files in os.walk(adv_dir):
        for fn in files:
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                if fn not in adv_map:
                    adv_map[fn] = os.path.join(root, fn)

    pairs = [(clean_path, adv_map[filename]) \
             for clean_path, filename in clean_list \
             if filename in adv_map]

    if not pairs:
        return float('nan')

    sum_phi = 0.0
    total   = 0

    for clean_p, adv_p in pairs:
        img_c = cv2.imread(clean_p)
        img_a = cv2.imread(adv_p)
        if img_c is None or img_a is None:
            continue
        t_c = preprocess(img_c)
        t_a = preprocess(img_a)
        logits_c, _, _ = predict_logits(t_c)
        logits_a, _, _ = predict_logits(t_a)
        phi_val = phi_kl(logits_c, logits_a).item()
        sum_phi += phi_val
        total   += 1

    return sum_phi / total if total > 0 else float('nan')


def main():
    global model
    model = MiniXception(num_classes=7).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    attacks = sorted(os.listdir(ATTACK_BASE_DIR))

    with open(OUTPUT_FILE, 'w') as f:
        f.write("Attack\tDataset\tAvgPhi\n")
        for atk in attacks:
            for ds in CLEAN_DIR_MAP.keys():
                avg_phi = run_for_dataset(atk, ds)
                f.write(f"{atk}\t{ds}\t{avg_phi:.4f}\n")
                print(f"{atk} | {ds} | Φ médio = {avg_phi:.4f}")

    print(f"Resultados salvos em: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
