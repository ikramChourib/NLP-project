# dashboard_nlp.py
# Pour exécuter ce dashboard :
# 👉 streamlit run dashboard_nlp.py

import json
from pathlib import Path

import streamlit as st
import pandas as pd

# ---------------------------------------------------------
# 📁 Dossier contenant les métriques générées par train_nlp.py et benchmark_nlp.py
# ---------------------------------------------------------
OUTPUT_DIR = Path("outputs")

# ---------------------------------------------------------
# 🎨 Titre du tableau de bord Streamlit
# ---------------------------------------------------------
st.title("Dashboard NLP – PyTorch + Transformers")

# Chemins vers les fichiers de métriques
metrics_path = OUTPUT_DIR / "metrics.json"          # métriques d'entraînement
sys_metrics_path = OUTPUT_DIR / "system_metrics.json"  # métriques système (latence, taille...)

# ---------------------------------------------------------
# 📊 Chargement et affichage des métriques d'entraînement
# ---------------------------------------------------------
if metrics_path.exists():
    # Lecture du fichier JSON contenant l'historique train/val
    with open(metrics_path) as f:
        history = json.load(f)

    # Création d'un DataFrame pour faciliter l'affichage
    df = pd.DataFrame({
        "epoch": range(1, len(history["train_loss"]) + 1),
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "train_acc": history["train_acc"],
        "val_acc": history["val_acc"],
    })

    # 🔹 Courbe de loss (train vs val)
    st.subheader("Courbes de loss")
    st.line_chart(df.set_index("epoch")[["train_loss", "val_loss"]])

    # 🔹 Courbe d'accuracy (train vs val)
    st.subheader("Courbes d'accuracy")
    st.line_chart(df.set_index("epoch")[["train_acc", "val_acc"]])

else:
    # Aucun fichier métrique trouvé → afficher un message d'avertissement
    st.warning("Aucune métrique d'entraînement trouvée. Lance d'abord train_nlp.py")


# ---------------------------------------------------------
# ⚙️ Affichage des métriques système : latence, taille du modèle, énergie
# ---------------------------------------------------------
if sys_metrics_path.exists():
    # Lecture du fichier JSON
    with open(sys_metrics_path) as f:
        sys_metrics = json.load(f)

    st.subheader("Perf système & modèle")

    # 🔹 Latence moyenne mesurée par benchmark_nlp.py
    if "latency_ms" in sys_metrics:
        st.metric("Latence moyenne (ms)", f"{sys_metrics['latency_ms']:.3f}")

    # 🔹 Taille du modèle sauvegardé
    if "model_size_mb" in sys_metrics:
        st.metric("Taille du modèle (MB)", f"{sys_metrics['model_size_mb']:.2f}")

    # 🔹 Optionnel : émissions CO₂ (si CodeCarbon activé)
    if "emissions_kg" in sys_metrics:
        st.metric("Émissions CO₂ (kg)", f"{sys_metrics['emissions_kg']:.4f}")

else:
    # Aucun fichier système trouvé → afficher une info
    st.info("Pas encore de metrics système. Lance benchmark_nlp.py")
