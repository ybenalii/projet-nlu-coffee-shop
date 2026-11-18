# src/data.py
"""
Exercice — TON assistant NLU, à partir de TES données (le sujet est libre).

Au TP3, les phrases étaient écrites dans ce fichier. Ici, elles sont dans des
FICHIERS, dans le dossier donnees/ (réglable par DONNEES_DIR dans .env) :

  ontologie.yaml   le domaine + la liste FERMÉE des intentions et des types de slots
  train.csv        pour APPRENDRE
  validation.csv   pour RÉGLER (nombre de pas, prompt...)
  test.csv         pour JUGER, une seule fois (juger.py)

Chaque CSV a deux colonnes, séparées par un point-virgule :
  phrase;reponse
  allume la lumière du salon;light_on | room=salon

La colonne « reponse » est EXACTEMENT ce que le modèle doit répondre :
  <intention> | <slot>=<valeur> | <slot>=<valeur>

Le code ne change pas d'un domaine à l'autre : seules les données changent.
"""
import csv
from pathlib import Path

import yaml
from datasets import Dataset

from src.config import settings

SPLIT_NAMES = ("train", "validation", "test")


# ── Lecture des fichiers ───────────────────────────────────────────────
def data_dir() -> Path:
    return Path(settings.donnees_dir)


def load_ontology(folder: Path | None = None) -> dict:
    """{'domaine': str, 'intentions': {nom: description}, 'slots': {nom: description}}"""
    path = (folder or data_dir()) / "ontologie.yaml"
    with open(path, encoding="utf-8") as f:
        onto = yaml.safe_load(f) or {}
    return {"domaine": str(onto.get("domaine") or "").strip(),
            "intentions": {k: str(v or "").strip() for k, v in (onto.get("intentions") or {}).items()},
            "slots": {k: str(v or "").strip() for k, v in (onto.get("slots") or {}).items()}}


def read_csv(path) -> list[tuple[str, str]]:
    """Lit un fichier phrase;reponse -> [(phrase, reponse)]."""
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:   # -sig : tolère le BOM d'Excel
        for rec in csv.DictReader(f, delimiter=";"):
            phrase = (rec.get("phrase") or "").strip()
            reponse = (rec.get("reponse") or "").strip()
            if phrase or reponse:
                rows.append((phrase, reponse))
    return rows


def read_split(name: str, folder: Path | None = None) -> list[tuple[str, str]]:
    if name not in SPLIT_NAMES:
        raise ValueError(f"Jeu inconnu : {name} (attendu : {', '.join(SPLIT_NAMES)})")
    return read_csv((folder or data_dir()) / f"{name}.csv")


def rows_to_dataset(rows) -> Dataset:
    return Dataset.from_dict({"instruction": [p for p, _r in rows],
                              "response": [r for _p, r in rows]})


def get_split(name: str) -> Dataset:
    return rows_to_dataset(read_split(name))


def get_train_dataset() -> Dataset:
    return get_split("train")


def get_validation_dataset() -> Dataset:
    return get_split("validation")


def get_test_dataset() -> Dataset:
    return get_split("test")


def _key(phrase: str) -> str:
    """Clé de comparaison : minuscules, espaces et ponctuation simplifiés."""
    return " ".join(phrase.lower().replace(",", " ").replace("?", " ").split())


def assert_disjoint() -> None:
    """Garde-fou : aucune phrase commune entre deux jeux."""
    sets = {n: {_key(p) for p, _r in read_split(n)} for n in SPLIT_NAMES}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        commun = sets[a] & sets[b]
        assert not commun, (f"Fuite de données entre {a} et {b} : {sorted(commun)[:3]}"
                            " -> lance  uv run python valider_donnees.py")


# ── Construire un prompt système à partir de l'ontologie ───────────────
def build_system_prompt(onto: dict) -> str:
    domaine = onto.get("domaine") or "un assistant"
    lignes = [
        f"Tu es l'analyseur de messages pour : {domaine}.",
        "Pour chaque message, réponds UNIQUEMENT par une ligne au format :",
        "<intention> | <slot>=<valeur> | <slot>=<valeur>" if onto["slots"] else "<intention>",
        "Intentions autorisées :",
        *[f"- {n} : {d}" if d else f"- {n}" for n, d in onto["intentions"].items()],
    ]
    if onto["slots"]:
        lignes += [
            "Types de slots autorisés :",
            *[f"- {n} : {d}" if d else f"- {n}" for n, d in onto["slots"].items()],
            "La valeur d'un slot est recopiée du message, en minuscules, sans article. "
            "S'il n'y a aucun slot, réponds seulement l'intention.",
        ]
    return "\n".join(lignes) + "\n"


def examples_for_prompt(name: str, one_per_intent: bool = False):
    """Paires (phrase, reponse) d'un jeu, pour les écrire dans un prompt."""
    pairs, seen = [], set()
    for phrase, reponse in read_split(name):
        intent = parse_target(reponse)[0]
        if one_per_intent and intent in seen:
            continue
        seen.add(intent)
        pairs.append((phrase, reponse))
    return pairs


# ── Parsing + scoring de la sortie (identique au TP3) ──────────────────
_ARTICLES = {"le", "la", "les", "un", "une", "des"}


def _norm_value(v: str) -> str:
    """minuscules + retrait d'un article en tête (la valeur ; pas le type)."""
    v = v.strip().lower()
    if v.startswith("l'"):
        v = v[2:].strip()
    toks = v.split()
    if toks and toks[0] in _ARTICLES:
        v = " ".join(toks[1:])
    return v.strip()


def parse_target(text: str):
    """Ligne -> (intent, set[(type,valeur)]) normalisés."""
    line = text.strip().splitlines()[0] if text.strip() else ""
    parts = [p.strip() for p in line.split("|")]
    intent = parts[0].strip().lower() if parts else ""
    slots = set()
    for p in parts[1:]:
        if "=" in p:
            t, v = p.split("=", 1)
            slots.add((t.strip().lower(), _norm_value(v)))
    return intent, slots


def score(pred: str, gold: str):
    """(intent_ok, slot_f1, exact_match)."""
    pi, ps = parse_target(pred)
    gi, gs = parse_target(gold)
    intent_ok = (pi == gi)
    tp = len(ps & gs)
    prec = tp / len(ps) if ps else (1.0 if not gs else 0.0)
    rec = tp / len(gs) if gs else 1.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else (1.0 if not gs and not ps else 0.0)
    exact = intent_ok and (ps == gs)
    return intent_ok, f1, exact


def format_chat(example, tokenizer) -> str:
    """Met un exemple au format conversation (renvoie une chaîne)."""
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
