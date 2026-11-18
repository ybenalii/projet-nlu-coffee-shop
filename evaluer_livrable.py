# evaluer_livrable.py
"""
ÉVALUER un livrable, comme le fera la personne qui le reçoit.
Pas besoin de MLflow ni de Docker : seulement le zip et le modèle de base.

  uv run python evaluer_livrable.py livrables/livrable_groupe-x.zip
  uv run python evaluer_livrable.py livrable.zip --test test_croise.csv  # un test externe
  uv run python evaluer_livrable.py livrable.zip --demo                  # essayer des phrases

Ce que fait le script :
  1. dézippe le livrable et lit son rapport ;
  2. télécharge le modèle de base (même nom, même révision) et y greffe
     l'adaptateur livré ;
  3. applique la porte (LoRA / base / base+prompt du groupe) sur le test choisi.
     Une phrase du test qui apparaît aussi dans train ou validation du groupe
     est retirée du score (et signalée).
"""
import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import read_csv, _key
from src.evaluation import evaluate_gate, load_prompt
from src.generate import generate_answer


def extraire(archive: Path, dest: Path) -> Path:
    with zipfile.ZipFile(archive) as z:
        for m in z.namelist():                       # pas de chemin qui sort du dossier
            if Path(m).is_absolute() or ".." in Path(m).parts:
                raise SystemExit(f"Chemin suspect dans le zip : {m}")
        z.extractall(dest)
    racines = [p.parent for p in dest.rglob("rapport.json")]
    if len(racines) != 1:
        raise SystemExit("Livrable invalide : rapport.json introuvable.")
    return racines[0]


def charger(racine: Path, rapport: dict):
    base = rapport["modele_base"]
    kwargs = {"revision": base["revision"]} if base.get("revision") else {}
    tokenizer = AutoTokenizer.from_pretrained(base["nom"], **kwargs)
    model = AutoModelForCausalLM.from_pretrained(base["nom"], dtype=torch.float32, **kwargs)
    model = PeftModel.from_pretrained(model, racine / "adaptateur")
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser(description="Évaluer un livrable (projet NLU).")
    ap.add_argument("livrable")
    ap.add_argument("--test", help="un test externe (phrase;reponse) : par exemple un test croisé")
    ap.add_argument("--demo", action="store_true", help="taper des phrases et voir la réponse")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        racine = extraire(Path(args.livrable), Path(tmp))
        rapport = json.loads((racine / "rapport.json").read_text(encoding="utf-8"))
        print(f"Livrable du groupe : {rapport['groupe']}  (run {rapport['run']['nom']}, "
              f"porte déclarée : {rapport['porte']}, test lu {rapport['lectures_test']} fois)")
        taille = (racine / "adaptateur" / "adapter_model.safetensors").stat().st_size
        print(f"Adaptateur : {taille / 1e6:.1f} Mo  |  modèle de base : {rapport['modele_base']['nom']}")

        model, tokenizer = charger(racine, rapport)
        model.eval()

        if args.demo:
            print("\nTape une phrase (Entrée vide pour quitter).")
            while (phrase := input("> ").strip()):
                print("  ", generate_answer(model, tokenizer, phrase))
            return

        vus = {_key(p) for f in ("train.csv", "validation.csv")
               for p, _ in read_csv(racine / "donnees" / f)}
        if args.test:
            rows, titre = read_csv(args.test), f"TEST EXTERNE « {Path(args.test).name} »"
        else:
            rows, titre = read_csv(racine / "donnees" / "test.csv"), "TEST DU GROUPE"
        retenues = [(p, r) for p, r in rows if _key(p) not in vus]
        if len(retenues) < len(rows):
            print(f"⚠️  {len(rows) - len(retenues)} phrase(s) du test déjà présente(s) dans "
                  "train/validation du groupe : retirée(s) du score.")

        prompt = load_prompt(racine / rapport["prompt_porte"]) if rapport.get("prompt_porte") else None
        res = evaluate_gate(model, tokenizer, prompt, rows=retenues, title=titre)

        if not args.test and rapport["mesures_test"]:
            declare = rapport["mesures_test"].get("test_lora_intent_acc")
            if declare is not None and abs(declare - res["lora"]["intent_acc"]) > 1e-9:
                print(f"⚠️  intent_acc déclarée ({declare:.0%}) ≠ mesurée ({res['lora']['intent_acc']:.0%}). "
                      "Un petit écart peut venir du passage de l'adaptateur en 16 bits ; "
                      "un grand écart doit être expliqué par le groupe.")
            else:
                print("   Scores déclarés retrouvés à l'identique ✅")


if __name__ == "__main__":
    main()
