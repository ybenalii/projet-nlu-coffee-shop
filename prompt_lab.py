# prompt_lab.py
"""
LE LABO DE PROMPT : mesurer le modèle de BASE (sans fine-tuning), guidé par un
prompt système, sur un jeu. Aucun entraînement : rapide.

  # fabriquer les prompts à partir de TES données
  uv run python prompt_lab.py --construire systeme          # depuis ontologie.yaml
  uv run python prompt_lab.py --construire exemples-train   # + 1 exemple par intention, pris dans le TRAIN

  # mesurer un prompt sur la validation (le test, lui, se lit avec juger.py)
  uv run python prompt_lab.py prompts/systeme.txt validation

Chaque mesure est tracée dans MLflow (expérience projet-prompt-lab), avec un
nom lisible « <prompt> @ <jeu> ».
"""
import sys
from pathlib import Path

import mlflow

from src.config import settings
from src.data import load_ontology, build_system_prompt, examples_for_prompt
from src.model import load_base_model
from src.evaluation import evaluate_split, print_split_report, load_prompt, rows_to_text
from src.tracking import log_run, count_sealed_read, warn_sealed_read

SYSTEME = Path("prompts/systeme.txt")
EXEMPLES = Path("prompts/exemples_train.txt")


def build_prompt(kind: str) -> Path:
    SYSTEME.parent.mkdir(exist_ok=True)
    text = build_system_prompt(load_ontology())
    if kind == "systeme":
        SYSTEME.write_text(text, encoding="utf-8")
        print(f"Prompt écrit : {SYSTEME}  (à partir de {settings.donnees_dir}/ontologie.yaml)")
        return SYSTEME
    pairs = examples_for_prompt("train", one_per_intent=True)
    text += "Exemples :\n" + "\n".join(f"{p} -> {r}" for p, r in pairs) + "\n"
    EXEMPLES.write_text(text, encoding="utf-8")
    print(f"Prompt écrit : {EXEMPLES}  ({len(pairs)} exemples tirés du jeu « train »)")
    return EXEMPLES


def measure(prompt_path: str, split: str) -> None:
    if split not in ("validation", "test"):
        sys.exit(f"Jeu attendu : validation (ou test) — reçu : {split}")
    if not Path(prompt_path).is_file():
        sys.exit(f"Fichier de prompt introuvable : {prompt_path}  "
                 "(fabrique-le avec  uv run python prompt_lab.py --construire systeme)")
    prompt = load_prompt(prompt_path)

    settings.configure_environment()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    if split == "test":
        warn_sealed_read(split, count_sealed_read(split))
    mlflow.set_experiment(settings.mlflow_prompt_experiment_name)

    model, tokenizer = load_base_model(settings.model_name)   # PAS de LoRA
    metrics, rows = evaluate_split(model, tokenizer, split, system=prompt)
    nom = Path(prompt_path).stem
    print_split_report(f"Modèle de base + prompt « {nom} » sur le jeu « {split} »",
                       metrics, rows)

    with mlflow.start_run(run_name=f"{nom} @ {split}"):
        mlflow.set_tag("groupe", settings.groupe)
        mlflow.log_params({"prompt": nom, "jeu": split, "model_name": settings.model_name})
        log_run({"intent_acc": metrics["intent_acc"], "slot_f1": metrics["slot_f1"],
                 "exact": metrics["exact"]},
                texts={"prompt.txt": prompt + "\n", f"predictions_{split}.txt": rows_to_text(rows)})
    print(f"\nMesure tracée dans MLflow (expérience {settings.mlflow_prompt_experiment_name}).")


def main(argv):
    if len(argv) == 2 and argv[0] == "--construire" and argv[1] in ("systeme", "exemples-train"):
        build_prompt(argv[1])
    elif len(argv) == 2 and not argv[0].startswith("--"):
        measure(argv[0], argv[1])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
