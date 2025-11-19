# api_nlp.py
# Pour lancer l'API :
# 👉 uvicorn api_nlp:app --host 0.0.0.0 --port 8000
#
# Pour tester via curl :
# curl -X POST "http://localhost:8000/predict"  -H "Content-Type: application/json"  -d '{"text": "Apple releases a new iPhone with amazing features."}'

from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------
# 🔥 Device pour faire tourner le modèle
# ---------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dossier contenant le meilleur modèle sauvegardé par train_nlp.py
OUTPUT_DIR = Path("outputs")
BEST_MODEL_DIR = OUTPUT_DIR / "best_model"

# ---------------------------------------------------------
# 🏷️ Dictionnaire des labels AG News
# 0 = World, 1 = Sports, etc.
# ---------------------------------------------------------
ID2LABEL = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}

# ---------------------------------------------------------
# 🚀 Création de l'application FastAPI
# ---------------------------------------------------------
app = FastAPI(title="NLP Classification API", version="1.0.0")

# ---------------------------------------------------------
# 🧠 Chargement du tokenizer et du meilleur modèle
# Ces fichiers ont été sauvegardés via train_nlp.py
# ---------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(BEST_MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(BEST_MODEL_DIR).to(DEVICE)
model.eval()   # mode inférence : pas de dropout ni de gradients


# ---------------------------------------------------------
# 📨 Modèle de données en entrée de l'API
# Permet de valider le JSON {"text": "..."}
# ---------------------------------------------------------
class TextInput(BaseModel):
    text: str


# ---------------------------------------------------------
# 📌 Route GET pour vérifier que l'API fonctionne
# ---------------------------------------------------------
@app.get("/")
def root():
    return {"message": "API NLP (AG News classification)"}


# ---------------------------------------------------------
# 🔮 Route POST : prédiction NLP
# - On reçoit un texte
# - On retourne la classe prédite + confiance + probas
# ---------------------------------------------------------
@app.post("/predict")
async def predict(input: TextInput):
    # 1️⃣ Tokenisation du texte d'entrée → tenseurs pour PyTorch
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True).to(DEVICE)

    # 2️⃣ Inférence du modèle (pas de gradient)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # scores non normalisés
        probs = F.softmax(logits, dim=-1)[0]  # conversion en probabilités

    # 3️⃣ Sélection de la meilleure classe
    conf, pred_idx = torch.max(probs, dim=-1)
    label = ID2LABEL[pred_idx.item()]

    # 4️⃣ Construction de la réponse JSON
    return {
        "text": input.text,
        "predicted_label": label,                     # nom de la classe
        "label_id": int(pred_idx.item()),             # index numérique
        "confidence": float(conf.item()),             # probabilité max
        "probabilities": {
            ID2LABEL[i]: float(p) for i, p in enumerate(probs.tolist())
        },  # probas pour toutes les classes
    }
