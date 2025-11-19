# benchmark_nlp.py

import time
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from codecarbon import EmissionsTracker  # optionnel : mesurer l'énergie consommée

# ---------------------------------------------------------
# 🔥 Choix automatique du device : GPU Apple (mps), CUDA ou CPU
# ---------------------------------------------------------
if torch.backends.mps.is_available():
    DEVICE = "mps"         # GPU Apple (Mac M1/M2/M3)
elif torch.cuda.is_available():
    DEVICE = "cuda"        # GPU NVIDIA
else:
    DEVICE = "cpu"         # CPU
print("Using device:", DEVICE)

# 📁 Dossiers de sortie et dossier du meilleur modèle
OUTPUT_DIR = Path("outputs")
BEST_MODEL_DIR = OUTPUT_DIR / "best_model"


# ---------------------------------------------------------
# ⏱️ Fonction pour mesurer la latence moyenne d'inférence
# ---------------------------------------------------------
@torch.no_grad()  # désactive la grad, plus rapide et plus léger
def measure_latency(model, tokenizer, text="This is a test sentence.", warmup=10, runs=100):
    """
    Mesure la latence en millisecondes d'une prédiction textuelle.
    
    - warmup : chauffe le modèle pour stabiliser les performances
    - runs   : nombre de mesures pour faire la moyenne
    """

    model.eval()  # mode évaluation (désactive dropout, etc.)

    # Tokenisation du texte d'entrée → tenseurs PyTorch
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(DEVICE)

    # ---------------------------------------------------------
    # 🔥 Phase de warmup :
    # Première exécution souvent plus lente (cache, initialisation GPU...)
    # ---------------------------------------------------------
    for _ in range(warmup):
        _ = model(**inputs)

    # Synchronisation GPU (si CUDA) → garantit des mesures propres
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    # ---------------------------------------------------------
    # 🔍 Mesure réelle de latence
    # ---------------------------------------------------------
    start = time.perf_counter()
    for _ in range(runs):
        _ = model(**inputs)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    # Moyenne en millisecondes
    avg_latency_ms = (end - start) / runs * 1000
    return avg_latency_ms


# ---------------------------------------------------------
# 🚀 Fonction principale
# ---------------------------------------------------------
def main():
    # Chargement du tokenizer et du modèle sauvegardé dans train_nlp.py
    tokenizer = AutoTokenizer.from_pretrained(BEST_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BEST_MODEL_DIR).to(DEVICE)

    # ---------------------------------------------------------
    # ⚡ (Optionnel) Mesure d'énergie via CodeCarbon
    # ---------------------------------------------------------
    # tracker = EmissionsTracker()
    # tracker.start()

    # Mesure de la latence
    latency_ms = measure_latency(model, tokenizer)

    # emissions = tracker.stop()  # kg CO2eq (si activé)

    print(f"⏱ Latence moyenne par requête: {latency_ms:.3f} ms")

    # ---------------------------------------------------------
    # 📄 Sauvegarde des métriques système (latence, énergie...)
    # dans outputs/system_metrics.json
    # ---------------------------------------------------------
    sys_metrics_path = OUTPUT_DIR / "system_metrics.json"
    data = {
        "latency_ms": latency_ms,
        # "emissions_kg": emissions  # si CodeCarbon activé
    }

    # Si un fichier existe déjà → fusion des données
    if sys_metrics_path.exists():
        with open(sys_metrics_path) as f:
            old = json.load(f)
        old.update(data)
        data = old

    # Écriture JSON
    with open(sys_metrics_path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------
# ▶️ Lancer le benchmark
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
