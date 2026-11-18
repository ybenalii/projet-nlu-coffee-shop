# importer.py
"""
IMPORTER un jeu de données trouvé en ligne, pour une tâche de CLASSIFICATION
(une phrase -> une intention, sans slot), et le mettre au format du projet.

  # depuis Hugging Face (téléchargé automatiquement)
  uv run python importer.py --hf tblard/allocine --texte review --etiquette label --par-intention 30

  # depuis un fichier téléchargé à la main (csv, tsv, jsonl, json ou parquet)
  uv run python importer.py --fichier train.csv --texte text --etiquette category --nb-intentions 8
  uv run python importer.py --fichier avis.csv --texte review --etiquette polarity --noms 0=negatif,1=positif

Ce que fait le script :
  - garde les phrases courtes (--max-caracteres, 200 par défaut) et sans doublon ;
  - garde quelques intentions (--intentions a,b,c ou --nb-intentions K : les K plus fréquentes) ;
  - tire au hasard --par-intention phrases par intention (12 par défaut) et les répartit
    en train / validation / test (8 / 2 / 2 pour 12), sans phrase commune ;
  - écrit donnees/train.csv, validation.csv, test.csv et un squelette de donnees/ontologie.yaml
    que TU dois compléter (le domaine et une description par intention).

Vérifie TOUJOURS la licence du jeu : elle doit autoriser cet usage (et la citer).
Pour des slots, il faut annoter à la main : ajoute-les ensuite dans les CSV.
"""
import argparse
import csv
import random
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

from datasets import ClassLabel, load_dataset

from src.config import settings


def nom_propre(label) -> str:
    """'Card arrival' -> 'card_arrival' ; 'Négatif' -> 'negatif'."""
    s = unicodedata.normalize("NFKD", str(label)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    if not s or not s[0].isalpha():
        s = f"classe_{s}"
    return s


def charger(args):
    if args.hf:
        ds = load_dataset(args.hf, args.config, split=args.split)
    else:
        path = Path(args.fichier)
        suffixe = path.suffix.lower()
        if suffixe in (".csv", ".tsv"):
            ds = load_dataset("csv", data_files=str(path), split="train",
                              sep="\t" if suffixe == ".tsv" else (args.sep or ","))
        elif suffixe in (".json", ".jsonl"):
            ds = load_dataset("json", data_files=str(path), split="train")
        elif suffixe == ".parquet":
            ds = load_dataset("parquet", data_files=str(path), split="train")
        else:
            sys.exit(f"Format non reconnu : {suffixe} (csv, tsv, json, jsonl ou parquet)")
    for col in (args.texte, args.etiquette):
        if col not in ds.column_names:
            sys.exit(f"Colonne « {col} » introuvable. Colonnes disponibles : {', '.join(ds.column_names)}")
    return ds


def noms_des_etiquettes(ds, args) -> dict:
    """Valeur brute -> nom d'intention lisible."""
    if args.noms:
        return {k.strip(): nom_propre(v) for k, v in (x.split("=", 1) for x in args.noms.split(","))}
    feat = ds.features[args.etiquette]
    if isinstance(feat, ClassLabel):        # cas fréquent sur Hugging Face : 0, 1, 2... + des noms
        return {str(i): nom_propre(n) for i, n in enumerate(feat.names)}
    return {}


def main():
    ap = argparse.ArgumentParser(description="Importer un jeu de classification au format du projet.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf", help="nom d'un jeu Hugging Face, ex. PolyAI/banking77")
    src.add_argument("--fichier", help="un fichier local csv / tsv / jsonl / json / parquet")
    ap.add_argument("--config", help="sous-ensemble Hugging Face (ex. une langue)")
    ap.add_argument("--split", default="train", help="partie du jeu Hugging Face à lire (train par défaut)")
    ap.add_argument("--sep", help="séparateur d'un .csv (virgule par défaut)")
    ap.add_argument("--texte", required=True, help="colonne qui contient la phrase")
    ap.add_argument("--etiquette", required=True, help="colonne qui contient l'intention (la classe)")
    ap.add_argument("--noms", help="noms des classes, ex. 0=negatif,1=positif")
    ap.add_argument("--intentions", help="intentions à garder, séparées par des virgules")
    ap.add_argument("--nb-intentions", type=int, help="garder les K intentions les plus fréquentes")
    ap.add_argument("--par-intention", type=int, default=12, help="phrases tirées par intention (12)")
    ap.add_argument("--max-caracteres", type=int, default=200, help="longueur maximale d'une phrase (200)")
    ap.add_argument("--graine", type=int, default=42, help="graine du tirage au hasard (reproductible)")
    ap.add_argument("--forcer", action="store_true", help="écraser des données déjà présentes")
    args = ap.parse_args()

    dest = Path(settings.donnees_dir)
    train = dest / "train.csv"
    if train.is_file() and len(train.read_text(encoding="utf-8").splitlines()) > 1 and not args.forcer:
        sys.exit(f"{train} contient déjà des données. Ajoute --forcer pour les remplacer.")

    ds = charger(args)
    noms = noms_des_etiquettes(ds, args)

    # 1. Regrouper les phrases par intention (courtes, sans doublon)
    par_intention, vues = defaultdict(list), set()
    for ex in ds:
        texte = " ".join(str(ex[args.texte] or "").replace(";", ",").split())
        if not texte or len(texte) > args.max_caracteres or texte.lower() in vues:
            continue
        vues.add(texte.lower())
        brut = ex[args.etiquette]
        par_intention[noms.get(str(brut), nom_propre(brut))].append(texte)

    # 2. Choisir les intentions
    if args.intentions:
        garder = [nom_propre(x) for x in args.intentions.split(",")]
        inconnues = [g for g in garder if g not in par_intention]
        if inconnues:
            sys.exit(f"Intentions introuvables : {inconnues}. Disponibles : {sorted(par_intention)}")
    else:
        classement = sorted(par_intention, key=lambda k: (-len(par_intention[k]), k))
        k = args.nb_intentions or len(classement)
        if k > 12 and not args.nb_intentions:
            sys.exit(f"{len(classement)} intentions trouvées : c'est trop pour ce projet. "
                     "Choisis-en avec --nb-intentions 8 ou --intentions a,b,c.\n"
                     f"Les plus fréquentes : {', '.join(classement[:15])}")
        garder = sorted(classement[:k])

    # 3. Tirer et répartir : train / validation / test, par intention
    rng = random.Random(args.graine)
    n_eval = max(1, args.par_intention // 6)
    jeux = {"train": [], "validation": [], "test": []}
    for intent in garder:
        phrases = par_intention[intent][:]
        rng.shuffle(phrases)
        phrases = phrases[:args.par_intention]
        if len(phrases) < 3:
            print(f"⚠️  « {intent} » : seulement {len(phrases)} phrase(s) utilisable(s).")
        jeux["test"] += [(p, intent) for p in phrases[:n_eval]]
        jeux["validation"] += [(p, intent) for p in phrases[n_eval:2 * n_eval]]
        jeux["train"] += [(p, intent) for p in phrases[2 * n_eval:]]

    dest.mkdir(exist_ok=True)
    for nom, rows in jeux.items():
        rng.shuffle(rows)
        with open(dest / f"{nom}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter=";", lineterminator="\n")
            w.writerow(["phrase", "reponse"])
            w.writerows(rows)

    source = args.hf or Path(args.fichier).name
    onto = dest / "ontologie.yaml"
    if args.forcer or not onto.is_file() or "A_REMPLIR" in onto.read_text(encoding="utf-8"):
        lignes = [f"# Importé depuis : {source}  (vérifie et cite sa licence)",
                  "# À COMPLÉTER : le domaine, puis une description courte pour chaque intention.",
                  "", "domaine: A_REMPLIR une phrase qui décrit l'usage", "", "intentions:"]
        lignes += [f"  {i}: " for i in garder]
        lignes += ["", "slots: {}", ""]
        onto.write_text("\n".join(lignes), encoding="utf-8")
        print(f"Squelette écrit : {onto}  ->  complète le domaine et les descriptions.")

    print(f"Importé depuis {source} : {len(garder)} intentions "
          f"({', '.join(garder)})")
    print("  " + "  |  ".join(f"{n} : {len(r)} phrases" for n, r in jeux.items()))
    print("Ensuite :  make valider")


if __name__ == "__main__":
    main()
