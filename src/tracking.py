# src/tracking.py
"""
Le "carnet de bord" (TP2) : enregistrer dans MLflow les mesures (metrics), des
fichiers texte et le modèle entraîné (artifact) du run courant. Les réglages
(params) sont enregistrés dès le début du run, dans main.py.

TP3 : un COMPTEUR de lectures du test, gardé sur
l'expérience MLflow. Chaque lecture d'un jeu scellé laisse une trace visible.
"""
import tempfile

import mlflow
from mlflow import MlflowClient

from src.config import settings


def log_run(metrics: dict, model=None, tokenizer=None, texts: dict | None = None):
    # 1. Les mesures (on ignore les valeurs manquantes)
    mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
    # 2. Des fichiers texte lisibles (prédictions, prompt...)
    for name, text in (texts or {}).items():
        mlflow.log_text(text, name)
    # 3. L'adaptateur LoRA + le tokenizer -> stockés dans RustFS (S3)
    if model is not None:
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            tokenizer.save_pretrained(tmp)
            mlflow.log_artifacts(tmp, artifact_path="lora_adapter")


def count_sealed_read(split: str) -> int:
    """Compte une lecture du test et renvoie le total.

    Le compteur est un tag de l'expérience principale (projet-nlu) :
    lectures_test. Il est visible dans MLflow (page Experiments, colonne Tags).
    """
    client = MlflowClient(settings.mlflow_tracking_uri)
    exp = client.get_experiment_by_name(settings.mlflow_experiment_name)
    if exp is None:
        exp_id = client.create_experiment(settings.mlflow_experiment_name)
        tags = {}
    else:
        exp_id, tags = exp.experiment_id, exp.tags
    key = f"lectures_{split}"
    total = int(tags.get(key, "0")) + 1
    client.set_experiment_tag(exp_id, key, str(total))
    return total


def warn_sealed_read(split: str, total: int) -> None:
    nom = "TEST SCELLÉ" if split == "test" else "JEU DE RÉSERVE"
    if total == 1:
        print(f"\n🔒 Première lecture du {nom}.")
    else:
        print(f"\n⚠️  {nom} lu {total} fois dans ce projet. Si une de ces lectures a "
              "influencé un choix (réglage, prompt...), son score n'est plus une "
              "mesure honnête : c'est l'overfitting adaptatif (vu au TP3).")
