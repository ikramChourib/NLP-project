# train_nlp.py

import os
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.functional import cross_entropy

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

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

# ---------------------------------------------------------
# 📁 Dossier de sortie pour sauvegarder modèles et métriques
# ---------------------------------------------------------
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ---------------------------------------------------------
# 🧠 Modèle utilisé : DistilBERT
# NUM_LABELS = nombre de classes (AG News = 4)
# ---------------------------------------------------------
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 4


# ---------------------------------------------------------
# 📥 Chargement et préparation des données
# ---------------------------------------------------------
def load_data(batch_size=16, max_train_samples=3000, max_val_samples=500):
    # Chargement du dataset AG News depuis HuggingFace
    raw_datasets = load_dataset("ag_news")

    # On réduit la taille du dataset pour que l'entraînement soit rapide
    if max_train_samples is not None:
        raw_datasets["train"] = raw_datasets["train"].select(range(max_train_samples))
    if max_val_samples is not None:
        raw_datasets["test"] = raw_datasets["test"].select(range(max_val_samples))

    # Chargement du tokenizer correspondant au modèle BERT
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Fonction de prétraitement : transforme le texte → tokens
    def preprocess(batch):
        return tokenizer(batch["text"], truncation=True)

    # Application du tokenizer aux deux splits
    tokenized_train = raw_datasets["train"].map(preprocess, batched=True)
    tokenized_test = raw_datasets["test"].map(preprocess, batched=True)

    # On supprime la colonne texte (on garde que les tokens)
    tokenized_train = tokenized_train.remove_columns(["text"])
    tokenized_test = tokenized_test.remove_columns(["text"])

    # Pour utiliser PyTorch facilement
    tokenized_train.set_format("torch")
    tokenized_test.set_format("torch")

    # Le DataCollator gère automatiquement le padding pour les batches
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Création des DataLoaders (batchs)
    train_loader = DataLoader(
        tokenized_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=0,   # Important pour Mac (pas de multiprocessing)
    )

    val_loader = DataLoader(
        tokenized_test,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=0,
    )

    return train_loader, val_loader, tokenizer


# ---------------------------------------------------------
# 🎯 Fonction pour calculer l'accuracy
# ---------------------------------------------------------
def accuracy(preds, labels):
    preds_labels = preds.argmax(dim=-1)
    return (preds_labels == labels).float().mean().item()


# ---------------------------------------------------------
# 🔁 Une époque entière d'entraînement
# ---------------------------------------------------------
def train_one_epoch(model, train_loader, optimizer, scheduler, epoch):
    model.train()  # Mode entraînement
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for batch in train_loader:
        # On envoie les données vers le device (GPU/CPU)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        # 1) Reset des gradients
        optimizer.zero_grad()

        # 2) Forward pass (prédictions)
        outputs = model(**batch)
        logits = outputs.logits
        loss = outputs.loss  # La perte CrossEntropy est calculée automatiquement

        # 3) Backpropagation
        loss.backward()

        # 4) Mise à jour des poids
        optimizer.step()
        scheduler.step()

        # 5) Calcul de la métrique accuracy
        batch_size = batch["labels"].size(0)
        total_loss += loss.item() * batch_size
        total_acc += accuracy(logits.detach(), batch["labels"]) * batch_size
        total_samples += batch_size

    # Moyenne des métriques sur l'époque
    epoch_loss = total_loss / total_samples
    epoch_acc = total_acc / total_samples
    print(f"[Train] Epoch {epoch} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")

    return epoch_loss, epoch_acc


# ---------------------------------------------------------
# 🧪 Évaluation sur le dataset de validation
# ---------------------------------------------------------
@torch.no_grad()  # Pas de backpropagation
def evaluate(model, val_loader, epoch, split="val"):
    model.eval()  # Mode évaluation
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for batch in val_loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        loss = outputs.loss

        batch_size = batch["labels"].size(0)
        total_loss += loss.item() * batch_size
        total_acc += accuracy(logits, batch["labels"]) * batch_size
        total_samples += batch_size

    epoch_loss = total_loss / total_samples
    epoch_acc = total_acc / total_samples

    print(f"[{split.capitalize()}] Epoch {epoch} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")

    return epoch_loss, epoch_acc


# ---------------------------------------------------------
# 🚀 Fonction principale (orchestration de tout)
# ---------------------------------------------------------
def main():
    # Chargement dataset + tokenizer
    train_loader, val_loader, tokenizer = load_data(batch_size=16)

    # Chargement du modèle DistilBERT pour classification
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    ).to(DEVICE)

    # Nombre d'époques
    num_epochs = 10

    # Nombre total d'étapes (pour le scheduler)
    total_steps = num_epochs * len(train_loader)

    # Optimiseur AdamW (le standard pour Transformers)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Scheduler pour réduire le learning rate progressivement
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # Dictionnaire pour sauvegarder les métriques
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Pour sauvegarder le meilleur modèle
    best_acc = 0.0
    best_model_dir = OUTPUT_DIR / "best_model"
    best_model_dir.mkdir(exist_ok=True, parents=True)

    # ------------------------------
    # 🔁 Boucle d'entraînement
    # ------------------------------
    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scheduler, epoch)
        val_loss, val_acc = evaluate(model, val_loader, epoch, split="val")

        # Sauvegarde des métriques
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Sauvegarde du meilleur modèle
        if val_acc > best_acc:
            best_acc = val_acc
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            print(f"✅ Nouveau meilleur modèle sauvegardé (acc={best_acc:.4f})")

    # Sauvegarde des métriques dans metrics.json
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(history, f, indent=2)

    # ---------------------------------------------------------
    # 📦 Calcul de la taille totale du modèle sauvegardé
    # ---------------------------------------------------------
    total_size_bytes = 0
    for root, _, files in os.walk(best_model_dir):
        for file in files:
            fp = os.path.join(root, file)
            total_size_bytes += os.path.getsize(fp)

    model_size_mb = total_size_bytes / (1024 * 1024)
    print(f"📦 Taille du modèle: {model_size_mb:.2f} MB")

    # Sauvegarde dans system_metrics.json
    sys_metrics_path = OUTPUT_DIR / "system_metrics.json"
    sys_data = {"model_size_mb": model_size_mb}

    # Fusion si le fichier existe
    if sys_metrics_path.exists():
        with open(sys_metrics_path) as f:
            old = json.load(f)
        old.update(sys_data)
        sys_data = old

    with open(sys_metrics_path, "w") as f:
        json.dump(sys_data, f, indent=2)


# ---------------------------------------------------------
# ▶️ Lancement du script
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
