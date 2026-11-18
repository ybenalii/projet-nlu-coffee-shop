# main.py
"""
Exercice (projet NLU) — RÉGLER : entraîner le LoRA sur TES données (dossier
donnees/) et le MESURER SUR LA VALIDATION. Ce programme ne lit JAMAIS le test.

Enchaînement : charger -> LoRA -> entraîner (train + validation) ->
mesurer la tâche sur la VALIDATION -> tout tracer dans MLflow.

Pour JUGER un modèle (lire le test une fois) : uv run python juger.py

Pré-requis : `make setup` (MLflow + RustFS, comme au TP2).
Lancer :  uv run python main.py
"""
import mlflow

from src.config import settings
from src.data import get_train_dataset, get_validation_dataset, assert_disjoint
from src.model import load_base_model, apply_lora
from src.train import train
from src.evaluation import evaluate_split, print_split_report, rows_to_text
from src.tracking import log_run


def main():
    assert_disjoint()

    settings.configure_environment()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    params = {
        "model_name": settings.model_name,
        "lora_r": settings.lora_r,
        "lora_alpha": settings.lora_alpha,
        "lora_dropout": settings.lora_dropout,
        "train_max_steps": settings.train_max_steps,
        "train_learning_rate": settings.train_learning_rate,
        "task": "nlu",
        "donnees_dir": settings.donnees_dir,
    }

    # Un nom lisible dans MLflow, plutôt qu'un nom tiré au hasard
    run_name = f"lora {settings.train_max_steps} pas"
    with mlflow.start_run(run_name=run_name) as run:
        print(f"MLflow run: {run.info.run_id}  ({run_name})")
        mlflow.set_tag("groupe", settings.groupe)
        mlflow.log_params(params)

        model, tokenizer = load_base_model(settings.model_name)
        model = apply_lora(model, settings.lora_r, settings.lora_alpha, settings.lora_dropout)

        train_ds, val_ds = get_train_dataset(), get_validation_dataset()
        print(f"Train: {len(train_ds)} | Validation: {len(val_ds)}  (le test reste scellé)")

        print("\n=== ENTRAÎNEMENT (NLU) ===")
        model, train_metrics = train(model, tokenizer, train_ds, val_ds)

        # Mesurer la TÂCHE (pas seulement la perte) sur la validation
        val, val_rows = evaluate_split(model, tokenizer, "validation")
        print_split_report("Mesure sur la VALIDATION (sert à régler)", val, val_rows)

        metrics = {
            **train_metrics,
            "val_intent_acc": val["intent_acc"],
            "val_slot_f1": val["slot_f1"],
            "val_exact": val["exact"],
        }
        log_run(metrics, model, tokenizer,
                texts={"predictions_validation.txt": rows_to_text(val_rows)})

        print(f"\nRun enregistré dans MLflow -> {settings.mlflow_tracking_uri}/#/experiments")
        print("Change un réglage dans .env et relance, ou juge le meilleur run :"
              "  uv run python juger.py")


if __name__ == "__main__":
    main()
