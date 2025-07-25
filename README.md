# Avaliação de Ataques adversarios na detecção de emoções
Este repositório reúne todo o código, dados e resultados do projeto de Processamento de Imagem que avalia o impacto de ataques adversariais em modelos de classificação de expressões faciais. 
Foram usados os conjuntos de dados FER-2013 e RAF-DB e testados ataques FGSM, PGD e DeepFool em modelos pré-treinados e em fine‑tuning adversarial.
## Membros:
- Gabriel Santos de Andrade - RA: 815407
- Leonardo Prado Silva - RA: 813169

## 📂 Estrutura do Projeto

```
.
├── .vscode/                       # Configurações do VSCode (análise inicial + ε=1)
├── Datasets/                      # Dados originais (FER-2013, RAF-DB)
├── adversarial_attacks/           # Código‑base para geração de ataques
├── attacked_datasets/             # Imagens atacadas geradas
├── images/                        # Resultados e figuras para o relatório
├── trained_models/                # Pesos dos modelos treinados
├── .gitignore
├── _mini_XCEPTION.102-0.66.hdf5   # Modelo base pré-treinado (Mini‑Xception)
├── apply_attacks.py               # Gera imagens adversariais em RAF‑DB (e=5)
├── apply_attacks_self_model.py    # Usa o próprio modelo treinado para criar ataques
├── create_graph_phi.py            # Gera gráfico da métrica ϕ
├── evaluate_phi.py                # Calcula a métrica ϕ de robustez
├── fine_tuning_adversarial.py     # Fine‑tuning com ataques adversariais
├── model.py                       # Definição do modelo e pipeline de transfer learning
├── phi_results.txt                # Resultados ϕ (modelos limpos)
├── phi_results_adv.txt            # Resultados ϕ (modelos adversariais)
├── test_models_FER_2013.py        # Avaliação em FER‑2013 (modelos base)
├── test_models_RAF.py             # Avaliação em RAF‑DB (modelos base)
├── test_models_RAF_atk.py         # Avaliação em RAF‑DB atacado
├── test_trained_model_FER.py      # Avaliação de modelo treinado em FER
├── test_trained_model_RAF-DB.py   # Avaliação de modelo treinado em RAF
├── test_trained_model_adv_RAF-DB.py # Avaliação de modelo adversarial em RAF
└── train_model.py                 # Script de treinamento (limpo e adv)
```

## Dados:
- Modelos: Mini‑Xception e variantes fine‑tuned em FER‑2013 e RAF-DB.
- Ataques: FGSM, PGD e DeepFool, comparando forças (ε) diferentes.
- Métrica ϕ (evaluate_phi.py): Percentual de queda de acurácia entre modelos limpos e atacados.
- Fine‑tuning Adversarial: Especialização do modelo usando imagens adversariais durante o treinamento.

## Pré-requisitos
- Python 3.8+
- PyTorch
- OpenCV / Pillow
- NumPy, Pandas, Matplotlib
- tqdm
- (Opcional) CUDA Toolkit para treinamento em GPU
