# valider_donnees.py
"""
Vérifie TES données avant d'entraîner (dossier donnees/, ou DONNEES_DIR).

  uv run python valider_donnees.py
  uv run python valider_donnees.py --test-croise test_pour_groupe_b.csv --ontologie ontologie_groupe_b.yaml

Erreurs (❌, à corriger : l'entraînement serait faux ou impossible) :
  - ontologie encore au stade du modèle, fichier manquant, mauvais séparateur ;
  - réponse mal formée, intention ou type de slot absent de ontologie.yaml ;
  - une même phrase dans deux jeux (fuite de données).
Avertissements (⚠️, à regarder) :
  - valeur de slot introuvable dans la phrase (faute de frappe ?) ;
  - phrase ou réponse trop longue pour le petit modèle ;
  - doublons, jeux trop petits, intention trop rare, ontologie très grande.

Code de sortie : 1 s'il y a au moins une erreur, 0 sinon.
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

from src.config import settings
from src.data import SPLIT_NAMES, read_csv, _key, _norm_value

TAILLE_MIN = {"train": 40, "validation": 10, "test": 10, "test croisé": 15}
NOM_OK = re.compile(r"^[a-z][a-z0-9_]*$")
PHRASE_MAX = 300      # caractères : au-delà, l'exemple risque d'être tronqué à l'entraînement
REPONSE_MAX = 25      # mots : la génération s'arrête à 40 tokens


class Rapport:
    def __init__(self):
        self.erreurs, self.avertissements = [], []

    def erreur(self, msg):
        self.erreurs.append(msg)

    def avert(self, msg):
        self.avertissements.append(msg)


def verifier_ontologie(path: Path, r: Rapport):
    if not path.is_file():
        r.erreur(f"{path} introuvable.")
        return None
    try:
        onto = _charger_yaml(path)
    except Exception as e:  # YAML mal formé
        r.erreur(f"{path} illisible : {e}")
        return None
    if "A_REMPLIR" in onto["domaine"] or any("a_remplir" in n for n in [*onto["intentions"], *onto["slots"]]):
        r.erreur("ontologie.yaml est encore le MODÈLE : décris ton sujet "
                 "(ou pars de l'exemple :  make exemple-domotique).")
        return None
    if not onto["domaine"]:
        r.avert("ontologie.yaml : ajoute une ligne  domaine: …  (elle sert au prompt système).")
    if len(onto["intentions"]) < 2:
        r.erreur("ontologie.yaml : il faut au moins 2 intentions.")
    elif len(onto["intentions"]) > 12:
        r.avert(f"ontologie.yaml : {len(onto['intentions'])} intentions. Au-delà de 12, il faut beaucoup "
                "plus de données et d'entraînement : regroupe ou réduis si tu peux.")
    if len(onto["slots"]) > 8:
        r.avert(f"ontologie.yaml : {len(onto['slots'])} types de slots (conseillé : 8 au plus).")
    for kind in ("intentions", "slots"):
        for nom, desc in onto[kind].items():
            if not NOM_OK.match(str(nom)):
                r.erreur(f"ontologie.yaml : nom « {nom} » invalide (minuscules, chiffres et _ seulement).")
            if not desc:
                r.avert(f"ontologie.yaml : « {nom} » n'a pas de description (elle sert au prompt système).")
    return onto


def _charger_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        onto = yaml.safe_load(f) or {}
    return {"domaine": str(onto.get("domaine") or "").strip(),
            "intentions": {k: str(v or "").strip() for k, v in (onto.get("intentions") or {}).items()},
            "slots": {k: str(v or "").strip() for k, v in (onto.get("slots") or {}).items()}}


def verifier_csv(path: Path, label: str, onto: dict, r: Rapport):
    if not path.is_file():
        r.erreur(f"{path} introuvable.")
        return []
    first = path.read_text(encoding="utf-8-sig").splitlines()[:1]
    if not first or first[0].strip().lower() != "phrase;reponse":
        r.erreur(f"{path.name} : la 1re ligne doit être exactement  phrase;reponse  "
                 f"(trouvé : {first[0] if first else 'fichier vide'}). Séparateur = point-virgule.")
        return []
    rows = read_csv(path)
    if not rows:
        r.erreur(f"{path.name} est vide : écris au moins {TAILLE_MIN[label]} phrases.")
        return []
    for n, (phrase, reponse) in enumerate(rows, start=2):   # ligne 1 = en-tête
        ou = f"{path.name} ligne {n}"
        if not phrase or not reponse:
            r.erreur(f"{ou} : phrase ou réponse vide (un point-virgule manquant ?).")
            continue
        if len(phrase) > PHRASE_MAX:
            r.avert(f"{ou} : phrase de {len(phrase)} caractères (conseillé : moins de {PHRASE_MAX}).")
        if len(reponse.split()) > REPONSE_MAX:
            r.avert(f"{ou} : réponse de {len(reponse.split())} mots : trop longue pour être générée en entier.")
        parts = [p.strip() for p in reponse.split("|")]
        intent = parts[0]
        if intent not in onto["intentions"]:
            indice = "  Il manque un « | » ?" if ("=" in intent or " " in intent) else ""
            r.erreur(f"{ou} : intention « {intent} » absente de l'ontologie.{indice}")
        types_vus = []
        for p in parts[1:]:
            if "=" not in p:
                r.erreur(f"{ou} : « {p} » n'est pas de la forme slot=valeur.")
                continue
            t, v = (x.strip() for x in p.split("=", 1))
            types_vus.append(t)
            if t not in onto["slots"]:
                r.erreur(f"{ou} : type de slot « {t} » absent de l'ontologie.")
            if not v:
                r.erreur(f"{ou} : slot « {t} » sans valeur.")
            elif v != v.lower():
                r.avert(f"{ou} : valeur « {v} » avec des majuscules (le score compare en minuscules).")
            elif _norm_value(v) not in phrase.lower():
                r.avert(f"{ou} : valeur « {v} » introuvable dans la phrase (faute de frappe ?).")
        for t, k in Counter(types_vus).items():
            if k > 1:
                r.avert(f"{ou} : slot « {t} » présent {k} fois.")
    for d in [p for p, k in Counter(_key(p) for p, _ in rows).items() if k > 1]:
        r.avert(f"{path.name} : phrase en double « {d} ».")
    if len(rows) < TAILLE_MIN[label]:
        r.avert(f"{path.name} : {len(rows)} phrases seulement (conseillé : au moins {TAILLE_MIN[label]}).")
    return rows


def tableau(onto, jeux: dict, r: Rapport, conseils: bool = True):
    noms = list(jeux)
    print(f"{'intention':<24}" + "".join(f"{n:>13}" for n in noms))
    comptes = {n: Counter(rep.split("|")[0].strip() for _, rep in rows) for n, rows in jeux.items()}
    for intent in onto["intentions"]:
        print(f"{intent:<24}" + "".join(f"{comptes[n][intent]:>13}" for n in noms))
        if not conseils:
            continue
        if comptes["train"][intent] < 5:
            r.avert(f"intention « {intent} » : {comptes['train'][intent]} exemple(s) dans train "
                    "(conseillé : 8 ou plus).")
        for n in ("validation", "test"):
            if comptes[n][intent] == 0:
                r.avert(f"intention « {intent} » : absente de {n}.csv (elle ne sera pas mesurée).")
    print(f"{'TOTAL':<24}" + "".join(f"{len(jeux[n]):>13}" for n in noms))


def valider_projet() -> Rapport:
    folder = Path(settings.donnees_dir)
    print(f"Vérification des données de : {folder}/\n")
    r = Rapport()
    onto = verifier_ontologie(folder / "ontologie.yaml", r)
    if onto is None:
        return r
    print(f"Domaine : {onto['domaine'] or '(non précisé)'}  |  {len(onto['intentions'])} intentions, "
          f"{len(onto['slots'])} types de slots\n")
    jeux = {n: verifier_csv(folder / f"{n}.csv", n, onto, r) for n in SPLIT_NAMES}
    cles = {n: {_key(p) for p, _ in rows} for n, rows in jeux.items()}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        for p in sorted(cles[a] & cles[b]):
            r.erreur(f"Fuite de données : « {p} » est à la fois dans {a} et dans {b}.")
    tableau(onto, jeux, r)
    return r


def valider_test_croise(test: Path, ontologie: Path) -> Rapport:
    """Le test écrit pour un AUTRE groupe, à partir de sa seule ontologie."""
    print(f"Vérification du test croisé : {test}  (ontologie : {ontologie})\n")
    r = Rapport()
    onto = verifier_ontologie(ontologie, r)
    if onto is None:
        return r
    rows = verifier_csv(test, "test croisé", onto, r)
    tableau(onto, {"test croisé": rows}, r, conseils=False)
    absentes = [i for i in onto["intentions"] if not any(rep.split("|")[0].strip() == i for _, rep in rows)]
    for i in absentes:
        r.avert(f"intention « {i} » absente du test croisé.")
    return r


def finir(r: Rapport) -> int:
    print()
    for m in r.erreurs:
        print(f"❌ {m}")
    for m in r.avertissements:
        print(f"⚠️  {m}")
    print(f"\nBilan : {len(r.erreurs)} erreur(s), {len(r.avertissements)} avertissement(s).")
    if r.erreurs:
        print("Corrige les erreurs avant de continuer.")
        return 1
    print("Données utilisables ✅" + ("  (regarde quand même les avertissements)" if r.avertissements else ""))
    return 0


def main():
    ap = argparse.ArgumentParser(description="Vérifier des données (projet ou test croisé).")
    ap.add_argument("--test-croise", type=Path, help="un test écrit pour un autre groupe")
    ap.add_argument("--ontologie", type=Path, help="l'ontologie de ce groupe (avec --test-croise)")
    args = ap.parse_args()
    if args.test_croise:
        if not args.ontologie:
            ap.error("--test-croise demande aussi --ontologie")
        return finir(valider_test_croise(args.test_croise, args.ontologie))
    return finir(valider_projet())


if __name__ == "__main__":
    sys.exit(main())
