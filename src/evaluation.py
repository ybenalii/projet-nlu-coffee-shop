# src/evaluation.py
"""
TP3 — MESURER et JUGER.

1. evaluate_split : mesure un modèle sur UN jeu (validation ou test),
   avec ou sans prompt système. C'est l'outil de RÉGLAGE quand on l'utilise sur
   la validation.

2. evaluate_gate : LA PORTE DE QUALITÉ, sur le TEST SCELLÉ uniquement, en
   comparant TROIS colonnes :
     • LoRA        : notre modèle fine-tuné ;
     • base        : le modèle nu (aucune aide) ;
     • base+prompt : le modèle nu AVEC sa meilleure chance (le meilleur prompt
                     système, choisi sur la VALIDATION).
   Verdict (PASS/FAIL) = le LoRA réussit ET bat NETTEMENT base+prompt :
     intent_acc(LoRA) >= seuil  ET  intent_acc(LoRA) - intent_acc(base+prompt) >= marge

Astuce : pour les colonnes "base", on désactive temporairement l'adaptateur
LoRA du même modèle (disable_adapter) : pas besoin de recharger un 2e modèle.
"""
from contextlib import nullcontext
from pathlib import Path

from src.config import settings
from src.data import read_split, score
from src.generate import generate_answer


def load_prompt(path: str) -> str:
    """Lit un prompt système depuis un fichier texte (dossier prompts/)."""
    return Path(path).read_text(encoding="utf-8").strip()


def _avg(triples):
    """Moyenne (intent_acc, slot_f1, exact) sur une liste de (io, f1, ex)."""
    n = len(triples) or 1
    return {
        "intent_acc": sum(1 for io, _f, _e in triples if io) / n,
        "slot_f1": sum(f for _io, f, _e in triples) / n,
        "exact": sum(1 for _io, _f, e in triples if e) / n,
    }


def evaluate_rows(model, tokenizer, rows, system: str | None = None,
                  use_adapter: bool = True):
    """Mesure le modèle sur une liste de (phrase, reponse). Renvoie (métriques, lignes)."""
    # Sur un modèle LoRA, use_adapter=False donne le modèle de base
    ctx = nullcontext() if use_adapter or not hasattr(model, "disable_adapter") \
        else model.disable_adapter()
    out, triples = [], []
    with ctx:
        for phrase, gold in rows:
            pred = generate_answer(model, tokenizer, phrase, system=system)
            t = score(pred, gold)
            triples.append(t)
            out.append({"phrase": phrase, "attendu": gold, "obtenu": pred,
                        "intent_ok": t[0], "exact": t[2]})
    return _avg(triples), out


def evaluate_split(model, tokenizer, split: str, system: str | None = None,
                   use_adapter: bool = True):
    """Mesure le modèle sur un jeu du dossier donnees/ (validation ou test)."""
    return evaluate_rows(model, tokenizer, read_split(split), system, use_adapter)


def print_split_report(title: str, metrics: dict, rows=None, show_errors: bool = True):
    print(f"\n=== {title} ===")
    for m in ("intent_acc", "slot_f1", "exact"):
        print(f"{m:>12}  {metrics[m]*100:>4.0f}%")
    if rows and show_errors:
        erreurs = [r for r in rows if not r["exact"]]
        if erreurs:
            print(f"\n  Erreurs ({len(erreurs)}/{len(rows)}) :")
            for r in erreurs:
                marque = "intention ok" if r["intent_ok"] else "INTENTION FAUSSE"
                print(f"   - {r['phrase']}\n       attendu : {r['attendu']}\n"
                      f"       obtenu  : {r['obtenu'] or '(vide)'}   [{marque}]")


def rows_to_text(rows) -> str:
    """Les prédictions en clair, pour un artefact MLflow."""
    lignes = []
    for r in rows:
        ok = "OK " if r["exact"] else ("~  " if r["intent_ok"] else "XX ")
        lignes.append(f"{ok}{r['phrase']}\n    attendu : {r['attendu']}\n    obtenu  : {r['obtenu']}")
    return "\n".join(lignes) + "\n"


def evaluate_gate(model, tokenizer, system_prompt: str, rows=None, title="TEST SCELLÉ"):
    """La porte : 3 colonnes sur le test (par défaut donnees/test.csv) + verdict."""
    rows = read_split("test") if rows is None else rows
    lora, lora_rows = evaluate_rows(model, tokenizer, rows)
    base, _ = evaluate_rows(model, tokenizer, rows, use_adapter=False)
    base_prompt, _ = evaluate_rows(model, tokenizer, rows, system=system_prompt,
                                   use_adapter=False)
    res = {"lora": lora, "base": base, "base_prompt": base_prompt, "lora_rows": lora_rows}

    margin = lora["intent_acc"] - base_prompt["intent_acc"]
    res["margin_over_prompt"] = margin
    res["gate_passed"] = bool(lora["intent_acc"] >= settings.eval_gate_min_intent
                              and margin >= settings.eval_gate_margin
                              and lora["slot_f1"] >= settings.eval_gate_min_slot_f1)
    _print_gate_report(res, title)
    return res


def _print_gate_report(res, title="TEST SCELLÉ"):
    print(f"\n=== Résultats sur le {title} ({len(res['lora_rows'])} phrases) ===")
    cols = [("LoRA", "lora"), ("base", "base"), ("base+prompt", "base_prompt")]
    print(f"{'':>12}  " + "  ".join(f"{name:>12}" for name, _ in cols))
    for metric in ("intent_acc", "slot_f1", "exact"):
        line = "  ".join(f"{res[k][metric]*100:>11.0f}%" for _n, k in cols)
        print(f"{metric:>12}  {line}")
    print(f"\n   avance LoRA vs base+prompt : {res['margin_over_prompt']*100:+.0f} points")
    verdict = "PASS ✅" if res["gate_passed"] else "FAIL ❌"
    slots = (f", slot_f1 >= {settings.eval_gate_min_slot_f1:.0%}"
             if settings.eval_gate_min_slot_f1 > 0 else "")
    print(f"   GATE : {verdict}  (seuil intent >= {settings.eval_gate_min_intent:.0%}, "
          f"marge >= {settings.eval_gate_margin:.0%}{slots})")
