# juger.py
"""
JUGER : appliquer la PORTE DE QUALITÉ sur le TEST SCELLÉ, une seule fois,
à un modèle déjà entraîné et choisi sur la VALIDATION.

  uv run python juger.py            # juge le meilleur run selon la VALIDATION
  uv run python juger.py <run_id>   # juge un run précis

Le modèle n'est pas réentraîné : son adaptateur LoRA est récupéré dans MLflow
(stocké dans RustFS). Les 3 colonnes (LoRA / base / base+prompt) et le
verdict PASS/FAIL sont ajoutés au même run dans MLflow.
"""
import sys

import mlflow
from mlflow import MlflowClient
from peft import PeftModel

from src.config import settings
from src.data import assert_disjoint
from src.model import load_base_model
from src.evaluation import evaluate_gate, load_prompt, rows_to_text
from src.tracking import log_run, count_sealed_read, warn_sealed_read


def choose_best_run(client: MlflowClient) -> str:
    """Le meilleur run selon la VALIDATION : val_intent_acc la plus haute,
    puis val_exact, puis la plus petite best_val_loss. Le test n'intervient pas."""
    exp = client.get_experiment_by_name(settings.mlflow_experiment_name)
    if exp is None:
        sys.exit("Aucune expérience trouvée : lance d'abord  uv run python main.py")
    runs = client.search_runs([exp.experiment_id],
                              filter_string="metrics.val_intent_acc >= 0",
                              order_by=["metrics.val_intent_acc DESC",
                                        "metrics.val_exact DESC",
                                        "metrics.best_val_loss ASC"])
    if not runs:
        sys.exit("Aucun run mesuré sur la validation : lance d'abord  uv run python main.py")
    print("Runs candidats, classés sur la VALIDATION :")
    for r in runs:
        m = r.data.metrics
        print(f"   {r.info.run_name:<16} val_intent_acc={m.get('val_intent_acc', 0):.0%}"
              f"  val_exact={m.get('val_exact', 0):.0%}  ({r.info.run_id[:8]})")
    return runs[0].info.run_id


def main():
    assert_disjoint()
    settings.configure_environment()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient(settings.mlflow_tracking_uri)

    run_id = sys.argv[1] if len(sys.argv) > 1 else choose_best_run(client)
    run = client.get_run(run_id)
    print(f"\nRun jugé : {run.info.run_name} ({run_id})")

    # 1. Récupérer l'adaptateur du run (dans RustFS, via MLflow)
    adapter_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="lora_adapter")
    model, tokenizer = load_base_model(settings.model_name)
    model = PeftModel.from_pretrained(model, adapter_dir)

    # 2. Le prompt de la colonne base+prompt (choisi sur la validation)
    prompt = load_prompt(settings.prompt_systeme_file)
    print(f"Prompt de la colonne base+prompt : {settings.prompt_systeme_file}")

    # 3. Lire le TEST (une lecture de plus au compteur)
    warn_sealed_read("test", count_sealed_read("test"))
    res = evaluate_gate(model, tokenizer, prompt)

    # 4. Ajouter le jugement au même run
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tags({"gate_passed": str(res["gate_passed"]),
                         "prompt_porte": settings.prompt_systeme_file})
        log_run({
            "test_lora_intent_acc": res["lora"]["intent_acc"],
            "test_lora_slot_f1": res["lora"]["slot_f1"],
            "test_lora_exact": res["lora"]["exact"],
            "test_base_intent_acc": res["base"]["intent_acc"],
            "test_base_prompt_intent_acc": res["base_prompt"]["intent_acc"],
            "test_margin_over_prompt": res["margin_over_prompt"],
            "gate_passed": 1.0 if res["gate_passed"] else 0.0,
        }, texts={"predictions_test.txt": rows_to_text(res["lora_rows"])})
    print(f"\nJugement ajouté au run dans MLflow -> {settings.mlflow_tracking_uri}/#/experiments")

    # 5. Porte bloquante : un FAIL arrête en erreur (code 1), comme en production
    if settings.eval_gate_strict and not res["gate_passed"]:
        sys.exit("Porte de qualité : FAIL (EVAL_GATE_STRICT=true) -> modèle refusé.")


if __name__ == "__main__":
    main()
