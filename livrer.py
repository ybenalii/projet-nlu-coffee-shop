# livrer.py
"""
LIVRER le projet : un petit fichier zip, sans le modèle de 1,5 Go.

  uv run python livrer.py            # le meilleur run selon la VALIDATION
  uv run python livrer.py <run_id>   # un run précis

Pourquoi c'est petit ? Le modèle de base (Qwen3-0.6B) est public : tout le
monde peut le retélécharger à l'identique. On ne livre donc que ce qui est À
NOUS :
  - l'ADAPTATEUR LoRA (le patch appris), converti en 16 bits : ~2 Mo ;
  - les DONNÉES (ontologie + CSV) et le PROMPT : quelques Ko ;
  - un RAPPORT (rapport.json + LISEZMOI.md) : modèle de base et version exacte,
    réglages, scores, nombre de lectures du test.

Résultat : livrables/livrable_<groupe>.zip, aussi ajouté au run dans MLflow.
"""
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import mlflow
import torch
from mlflow import MlflowClient
from safetensors.torch import load_file, save_file
from transformers import AutoConfig

from juger import choose_best_run
from src.config import settings
from src.data import SPLIT_NAMES

FORMAT = "livrable-projet-nlu/1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adaptateur_16_bits(src: Path, dst: Path) -> None:
    """Copie l'adaptateur en convertissant ses poids en float16 (taille / 2)."""
    dst.mkdir(parents=True)
    poids = load_file(src / "adapter_model.safetensors")
    save_file({k: (v.to(torch.float16) if v.is_floating_point() else v) for k, v in poids.items()},
              dst / "adapter_model.safetensors")
    shutil.copy(src / "adapter_config.json", dst / "adapter_config.json")


def lisezmoi(r: dict) -> str:
    m, t = r["mesures_validation"], r["mesures_test"]
    pct = lambda v: "—" if v is None else f"{v:.0%}"
    return f"""# Livrable — {r['groupe']}

Modèle de base : `{r['modele_base']['nom']}` (révision `{r['modele_base']['revision'] or 'non précisée'}`)
Adaptateur LoRA : `adaptateur/` ({r['adaptateur']['taille_octets'] / 1e6:.1f} Mo, float16)
Run MLflow : `{r['run']['nom']}` ({r['run']['id']})

| mesure | validation | test |
|---|---|---|
| intent_acc (LoRA) | {pct(m.get('val_intent_acc'))} | {pct(t.get('test_lora_intent_acc'))} |
| exact (LoRA) | {pct(m.get('val_exact'))} | {pct(t.get('test_lora_exact'))} |
| intent_acc (base+prompt) | — | {pct(t.get('test_base_prompt_intent_acc'))} |

Porte de qualité : {r['porte']}
Lectures du test pendant le projet : {r['lectures_test']}
Notes du groupe : {'donnees/NOTES.md' if r['notes'] else 'aucune'}

Pour évaluer ce livrable (le modèle de base est téléchargé automatiquement) :

    uv run python evaluer_livrable.py {r['fichier']}
    uv run python evaluer_livrable.py {r['fichier']} --demo
"""


def main():
    settings.configure_environment()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient(settings.mlflow_tracking_uri)
    run_id = sys.argv[1] if len(sys.argv) > 1 else choose_best_run(client)
    run = client.get_run(run_id)
    met, tags = run.data.metrics, run.data.tags
    juge = "gate_passed" in met
    if not juge:
        print("⚠️  Ce run n'a pas encore été jugé sur le test : lance  make juger  avant de livrer.")

    exp = client.get_experiment(run.info.experiment_id)
    prompt_file = Path(tags.get("prompt_porte", settings.prompt_systeme_file))
    nom = f"livrable_{settings.groupe}"
    sortie = Path("livrables")
    sortie.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        racine = tmp / nom
        # 1. L'adaptateur (récupéré dans MLflow / RustFS), en 16 bits
        src = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="lora_adapter",
                                                       dst_path=str(tmp / "telechargement")))
        adaptateur_16_bits(src, racine / "adaptateur")
        # 2. Les données et le prompt
        (racine / "donnees").mkdir()
        for f in ["ontologie.yaml"] + [f"{n}.csv" for n in SPLIT_NAMES]:
            shutil.copy(Path(settings.donnees_dir) / f, racine / "donnees" / f)
        notes = Path(settings.donnees_dir) / "NOTES.md"     # les remarques du groupe
        if notes.is_file():
            shutil.copy(notes, racine / "donnees" / "NOTES.md")
        if prompt_file.is_file():
            (racine / "prompts").mkdir()
            shutil.copy(prompt_file, racine / "prompts" / prompt_file.name)
        # 3. Le rapport
        poids = racine / "adaptateur" / "adapter_model.safetensors"
        rapport = {
            "format": FORMAT,
            "groupe": settings.groupe,
            "date": datetime.now().isoformat(timespec="seconds"),
            "fichier": f"{nom}.zip",
            "modele_base": {"nom": settings.model_name,
                            "revision": getattr(AutoConfig.from_pretrained(settings.model_name),
                                                "_commit_hash", None)},
            "run": {"id": run_id, "nom": run.info.run_name},
            "reglages": dict(run.data.params),
            "mesures_validation": {k: v for k, v in met.items()
                                   if k.startswith("val_") or k.endswith("_loss")},
            "mesures_test": {k: v for k, v in met.items() if k.startswith("test_")},
            "porte": ("PASS" if met.get("gate_passed") == 1.0 else "FAIL") if juge else "non jugé",
            "prompt_porte": f"prompts/{prompt_file.name}" if prompt_file.is_file() else None,
            "lectures_test": int(exp.tags.get("lectures_test", "0")),
            "notes": notes.is_file(),
            "adaptateur": {"taille_octets": poids.stat().st_size, "sha256": sha256(poids)},
        }
        (racine / "rapport.json").write_text(json.dumps(rapport, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
        (racine / "LISEZMOI.md").write_text(lisezmoi(rapport), encoding="utf-8")
        # 4. Le zip
        archive = sortie / f"{nom}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(racine.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(tmp))

    taille = archive.stat().st_size
    print(f"\nLivrable : {archive}  ({taille / 1e6:.1f} Mo)")
    print(f"  porte : {rapport['porte']}  |  lectures du test : {rapport['lectures_test']}")
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(str(archive), artifact_path="livrable")
    print("  ajouté au run dans MLflow (onglet Artifacts, dossier livrable/)")


if __name__ == "__main__":
    main()
