# Exercice — Ton propre assistant : fine-tuner un modèle sur tes données

**Public :** débutant. **Pré-requis :** TP3 fait (régler / juger / overfitting
adaptatif).

**Idée :** au TP3, les phrases venaient d'un corpus préparé pour toi. Cette fois,
**c'est toi qui choisis l'usage** et qui constitues les données. Tu peux :

- inventer un assistant (pour une voiture, un restaurant, une mairie…) ;
- adapter un jeu de données trouvé en ligne ;
- partir de l'exemple fourni (la domotique).

Puis tu refais tout le cycle du TP3 :

1. écrire les données ;
2. les vérifier ;
3. régler sur la validation ;
4. juger une fois sur le test ;
5. **livrer** ton modèle en un fichier de quelques mégaoctets, au lieu d'un modèle
   de 1,5 Go.

> Convention : un cadre `comme ceci` = commande à taper dans un terminal,
> **depuis le dossier du projet** (celui qui contient `main.py` et `.env`).

---

## 0. Le déroulé

| Étape | Commande | Chapitre |
|---|---|---|
| choisir le sujet et écrire l'ontologie | — | 1, 2 |
| écrire, adapter ou importer les données | `make valider` | 3 |
| écrire le **test croisé** d'un autre groupe | `valider_donnees.py --test-croise` | 4 |
| régler le LoRA sur la validation | `make run` | 5 |
| régler le prompt sur la validation | `make prompt-systeme`, `make prompt-exemples` | 6 |
| juger, une fois | `make juger` | 7 |
| livrer | `make livrer`, `make evaluer` | 8 |

**Ce que tu livres :** un seul fichier, `livrables/livrable_<ton-groupe>.zip`,
d'environ 2 à 3 Mo, fabriqué par `make livrer` (chapitre 8). Il contient ton
adaptateur LoRA, tes données, ton meilleur prompt, tes notes et un rapport.

**Pourquoi ce format :** avec ce seul fichier, quelqu'un d'autre doit pouvoir
**reconstruire ton modèle**, le **mesurer** sur un test à lui et **retrouver tes
scores**. Ton modèle sera mesuré ainsi sur un **test croisé**, écrit par un autre
groupe, que tu ne verras jamais.

Pour démarrer :

1. Arrête la plateforme du TP3 (`make clean` dans le dossier `tp3`).
2. Mets le nom de ton groupe dans `.env`, par exemple `GROUPE=groupe-alice-bob`.
3. Lance :

   ```
   make setup
   uv sync
   make test
   ```

---

## 1. Choisir ton sujet

### 1.1 Les règles du jeu

Le projet reprend exactement la tâche du TP3 : **un message court → une réponse
courte et structurée**.

```
intention | slot=valeur | slot=valeur        (ou seulement : intention)
```

Ton sujet doit respecter ces contraintes :

- **Entrée courte** : une phrase ou un message de moins de 300 caractères. Pas de
  documents, pas de longs avis.
- **Sortie fermée** : une liste fixe de **3 à 12 intentions**, et **0 à 8 types
  de slots**. Sans slot, c'est une **classification**, et c'est permis.
- **Des intentions qui ne se chevauchent pas** : donne 10 phrases à deux membres
  du groupe. S'ils ne choisissent pas la même intention, l'ontologie n'est pas
  assez claire.
- **Assez de données** : au moins **8 phrases par intention** pour apprendre (15,
  c'est mieux), plus 2 à 4 par intention pour la validation et pour le test. Pas
  plus de 300 phrases d'apprentissage : on entraîne sur un ordinateur portable.
- **Français de préférence**. L'anglais est accepté.
- **Pas de données personnelles** : pas de vrais noms, adresses, numéros, ni
  données de santé. Invente.
- **Hors sujet** : rédiger un texte, résumer, traduire, répondre librement. La
  porte de qualité mesure une **étiquette exacte**, pas un texte.

> **Le bon sujet, au sens du TP3 :** une tâche où un simple prompt a du mal, parce
> qu'elle suit des étiquettes précises, des conventions « maison » ou un jargon.
> Si un prompt fait aussi bien que ton LoRA, la porte dira FAIL. C'est une
> conclusion **honnête et intéressante**, pas un échec.

### 1.2 Des idées

| Usage | Message | Réponse |
|---|---|---|
| **Domotique** (exemple fourni) | ferme les volets du salon | `shutter_close \| room=salon` |
| Commandes vocales d'une voiture | mets la clim à vingt et un degrés | `clim_regler \| temperature=vingt et un degrés` |
| Réservation au restaurant | une table pour quatre samedi soir | `reserver \| personnes=quatre \| date=samedi soir` |
| Signalements à la mairie | un lampadaire est cassé rue des lilas | `eclairage_public \| lieu=rue des lilas` |
| Support d'une application | je n'arrive plus à me connecter | `connexion_impossible` |
| Messages à la vie scolaire | Léo sera absent demain matin | `absence \| date=demain matin` |
| Commandes d'un robot ou d'un jeu | avance de trois cases | `deplacer \| direction=avant \| distance=trois cases` |
| Médiathèque | est-ce que Dune est disponible | `disponibilite \| titre=dune` |
| Avis clients d'une boutique | livraison rapide mais carton abîmé | `avis_mitige \| aspect=emballage` |

Tu peux aussi partir d'un usage qui te concerne : une association, un club, un
stage, un projet personnel…

### 1.3 Partir de données en ligne

Un jeu de données public peut fournir des phrases, mais il faut presque toujours
l'**adapter** : le traduire, le réduire, choisir quelques intentions, ajouter des
slots. **Lis toujours sa licence**, et cite la source dans tes notes.

| Jeu | Langue | Contenu | Licence | Idée d'usage |
|---|---|---|---|---|
| [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets) (fichier `banking_data/train.csv`) | anglais | 10 000 messages d'un support bancaire, 77 intentions | CC BY 4.0 | garder 6 à 8 intentions, et les traduire toi-même en français |
| [CLINC150](https://huggingface.co/datasets/clinc_oos) | anglais | 150 intentions dans 10 domaines (voyage, cuisine, banque…) | CC BY 3.0 ([source](https://github.com/clinc/oos-eval)) | les intentions d'un seul domaine |
| [MTOP](https://huggingface.co/datasets/mteb/MTOPIntentClassification) | **français** (et 5 autres langues) | commandes d'assistant : alarmes, messages, musique, météo… | CC BY-SA 4.0 d'après la [source](https://huggingface.co/datasets/tasksource/mtop) | un seul domaine ; les slots sont à annoter toi-même |
| [Allociné](https://huggingface.co/datasets/tblard/allocine) | **français** | 200 000 critiques de films, positives ou négatives | MIT | ne garder que les critiques **courtes** ; ajouter un slot à la main (par ex. l'acteur cité) |
| [Snips SLU](https://github.com/YuexinLi/spoken-language-understanding-research-datasets) | anglais, **français** | lumières connectées, enceinte musicale | **recherche uniquement**, sur formulaire | lis ses conditions avant : un usage hors recherche n'est pas permis |
| [data.gouv.fr](https://www.data.gouv.fr) | français | données publiques (FAQ, catégories de signalements…) | souvent Licence Ouverte | pour **inventer** des intentions réalistes |

Pour une **classification** (sans slot), `importer.py` convertit un jeu au format
du projet. Il tire au hasard le même nombre de phrases par intention et écrit un
squelette d'ontologie à compléter. Exemples :

```
# Banking77 : télécharge d'abord banking_data/train.csv depuis GitHub
uv run python importer.py --fichier train.csv --texte text --etiquette category \
    --intentions card_arrival,lost_or_stolen_card,pin_blocked,exchange_rate,top_up_failed,cash_withdrawal_charge \
    --par-intention 20

# depuis Hugging Face, téléchargé automatiquement
uv run python importer.py --hf tblard/allocine --texte review --etiquette label --par-intention 30 --max-caracteres 150
uv run python importer.py --hf clinc_oos --config small --texte text --etiquette intent --intentions book_flight,book_hotel,...

# depuis ton propre fichier (csv, tsv, jsonl ou parquet)
uv run python importer.py --fichier mes_donnees.csv --texte message --etiquette categorie
```

Options utiles :

- `--intentions a,b,c` : choisir les intentions à garder ;
- `--par-intention 30` : nombre de phrases tirées par intention ;
- `--noms 0=negatif,1=positif` : donner un nom aux classes numérotées.

Ensuite, complète `donnees/ontologie.yaml` (le domaine et une description par
intention), puis lance `make valider`.

> ⚠️ Un jeu importé est souvent **trop facile à tester** : les phrases de train et
> de test se ressemblent, parce qu'elles viennent de la même source. Ajoute des
> phrases **écrites par toi**, surtout dans le test. Le test croisé (chapitre 4)
> le montrera.

---

## 2. Écrire l'ontologie

Ouvre **`donnees/ontologie.yaml`** : c'est un **modèle à remplacer**. Pour partir
de l'exemple domotique, tape `make exemple-domotique`.

```yaml
domaine: les réservations d'un restaurant
intentions:
  reserver: réserver une table
  annuler: annuler une réservation
  horaires: demander les horaires d'ouverture
slots:
  personnes: le nombre de couverts (quatre, deux personnes)
  date: le jour et le moment (samedi soir, demain midi)
```

Quelques règles :

- les noms sont en **minuscules**, avec des **chiffres** et des **_** seulement ;
- chaque intention et chaque slot a une **description courte** : elle sert au
  prompt système (chapitre 6) ;
- `domaine` est une phrase qui décrit ton usage ;
- sans slot, écris `slots: {}`.

> ⚠️ Une fois le test croisé échangé (chapitre 4), **l'ontologie est figée** : le
> test d'un autre groupe s'appuie dessus.

---

## 3. Les données

Les données sont dans `donnees/`, en fichiers **CSV** à deux colonnes séparées par
un **point-virgule**. La colonne `reponse` contient **exactement** ce que le modèle
doit répondre :

```
phrase;reponse
une table pour quatre samedi soir;reserver | personnes=quatre | date=samedi soir
vous fermez à quelle heure;horaires
```

**Les règles d'une réponse :**

- la **valeur d'un slot est recopiée du message**, en minuscules, sans article ;
- s'il n'y a aucun slot, on écrit seulement l'intention ;
- on n'invente rien qui ne soit pas dans le message.

| Fichier | Rôle | Minimum | Conseillé |
|---|---|---|---|
| `train.csv` | apprendre | 8 par intention | 15 par intention |
| `validation.csv` | régler | 2 par intention | 4 par intention |
| `test.csv` | juger (une fois) | 2 par intention | 4 par intention |

### Écrire de bonnes données

- **Varie les verbes, les tournures** (ordres, questions, politesse, fautes de
  frappe réalistes) et les valeurs de slots.
- **Mets des messages sans slot**, et d'autres avec plusieurs slots.
- **Écris le test en dernier, et par quelqu'un d'autre** : un membre du groupe
  écrit `test.csv` **sans regarder** `train.csv`.
- Aucune phrase ne doit apparaître dans deux fichiers.

> **Outils :** VS Code suffit. Avec un tableur, enregistre en **CSV, séparateur
> point-virgule, encodage UTF-8**.

### Vérifier

```
make valider
```

Le vérificateur affiche le domaine et le nombre de phrases par intention et par
fichier. Il signale :

- des **erreurs** ❌ : ontologie encore au stade du modèle, intention ou slot hors
  ontologie, « | » oublié, séparateur virgule, phrase présente dans deux fichiers ;
- des **avertissements** ⚠️ : valeur introuvable dans le message, phrase trop
  longue, intention trop rare, ontologie très grande.

Recommence jusqu'à **0 erreur**. `make run` refuse d'entraîner tant qu'il en reste.

---

## 4. Le test croisé : écrire le test d'un autre groupe

Au TP3, tu as vu qu'un test qu'on a soi-même écrit, et relu pendant les réglages,
peut tromper. Le vrai juge est un test **écrit par quelqu'un d'autre** et
**jamais montré**. Dans ce projet, chaque groupe écrit ce test pour **un autre
groupe**.

1. Dès que ton ontologie est **figée**, remets `donnees/ontologie.yaml`. Elle
   part chez un autre groupe, qui écrira **ton** test croisé.
2. En échange, tu reçois l'ontologie d'un autre groupe. **Rien d'autre** : ni ses
   données, ni ses phrases. Tu es dans la situation d'un vrai utilisateur, qui
   ne sait que ce que l'assistant est censé comprendre.
3. Écris **à la main** au moins **15 messages** pour cette ontologie, avec leur
   réponse, dans un fichier `test_croise_<son-groupe>.csv`. Couvre toutes ses
   intentions.
4. Vérifie-le :

   ```
   uv run python valider_donnees.py --test-croise test_croise_<son-groupe>.csv --ontologie ontologie_<son-groupe>.yaml
   ```

5. Remets-le. Le groupe concerné ne le verra **jamais** : c'est ce qui en fait un
   vrai juge.

> Écris des phrases **à toi**. Ne les copie pas d'un jeu en ligne, car le groupe
> l'a peut-être utilisé : une phrase du test croisé déjà présente dans ses données
> est **retirée du score**.

Si tu as choisi la **domotique**, ton test croisé existe déjà : il est écrit sur
l'ontologie de l'exemple, et tu ne le verras pas non plus. Tu écris quand même un
test croisé pour un autre groupe.

---

## 5. Régler le LoRA, sur la validation

Comme au TP3 (chapitre 4) :

```
make run
```

Fais varier `TRAIN_MAX_STEPS` dans `.env` : **30, 60, 100, 150**. Avec beaucoup
d'intentions ou de données, il faudra peut-être plus.

Dans MLflow (http://localhost:5000, mode **Model training**, expérience
**`projet-nlu`**), ajoute avec **Columns** :

- `val_intent_acc` et `val_exact` ;
- le paramètre `train_max_steps`.

👀 **À regarder :** la liste des **erreurs sur la validation**, à la fin de chaque
run.

- Une intention souvent confondue avec une autre ? Ajoute au train des exemples
  qui les distinguent, ou précise l'ontologie (avant le test croisé seulement).
- Un slot souvent raté ? Ajoute des exemples qui l'utilisent.

> **Attention :** corriger le train en lisant les erreurs de la validation est
> **permis**, c'est son rôle. Mais la validation s'use à son tour. Corrige des
> **familles** d'erreurs, pas des phrases une par une.

---

## 6. Le prompt, sur la validation

Comme au TP3 (chapitre 5). Le prompt de départ est **fabriqué à partir de ton
ontologie** : domaine, intentions, slots et leurs descriptions.

```
make prompt-systeme
make prompt-exemples
```

Compare dans MLflow (expérience `projet-prompt-lab`), puis indique le meilleur
dans `.env`, par exemple :

```
PROMPT_SYSTEME_FILE=prompts/exemples_train.txt
```

Tu peux aussi écrire ton propre prompt dans `prompts/`.

---

## 7. Juger, une fois

Quand tes données et tes réglages sont **définitifs** :

```
make juger
```

Tu obtiens les trois colonnes (LoRA, base, base+prompt) sur **ton** test, avec le
verdict PASS/FAIL.

> Si tu modifies tes données **après** avoir lu le test, ton score n'est plus
> honnête (TP3). Le compteur `lectures_test` le montre, et il est dans ton
> livrable : dis-le dans tes notes.

---

## 8. Livrer : quoi, et pourquoi

Un modèle livré à une équipe ou à un client part **sans toi** : sans ton
ordinateur, sans ton MLflow, sans tes explications. Ce qui n'est pas dans le
livrable n'existe pas. Ton livrable doit donc permettre à quelqu'un d'autre de :

1. **reconstruire** ton modèle ;
2. le **mesurer** sur un test qu'il apporte ;
3. **vérifier** que tes scores déclarés sont vrais ;
4. **comprendre** tes choix et leurs limites.

Avant de le fabriquer, crée `donnees/NOTES.md`. Indique la source des données et
leur licence si tu en as importé, tes choix, ce qui ne marche pas, et combien de
fois tu as lu le test. Puis lance :

```
make livrer
```

👀 **À regarder :** `Livrable : livrables/livrable_<groupe>.zip (2.3 Mo)`, ou à
peu près.

### Ce qu'il contient, et à quoi sert chaque pièce

| Dans le livrable | Taille | Pourquoi |
|---|---|---|
| `adaptateur/` : le patch LoRA | ~2,3 Mo | pour **reconstruire** ton modèle : c'est **la seule chose apprise**. Converti en 16 bits, il est 2 fois plus léger |
| `rapport.json` + `LISEZMOI.md` | 2 Ko | pour **reconstruire** et **vérifier** : modèle de base et **version exacte**, réglages, scores, lectures du test. Fabriqués automatiquement |
| `donnees/` : ontologie + 3 CSV | quelques Ko | pour **vérifier** : on peut refaire ton entraînement, et écarter du test croisé les phrases déjà vues par ton modèle |
| `prompts/` : ton meilleur prompt | 1 Ko | pour **mesurer** honnêtement : il donne la colonne base+prompt, que ton LoRA doit battre |
| `donnees/NOTES.md` | 1 Ko | pour **comprendre** : source et licence, choix, limites, lectures du test |
| ~~le modèle Qwen3-0.6B~~ | ~~1,5 Go~~ | **pas livré** : il est public, et sa version exacte est notée dans le rapport |

C'est l'intérêt de LoRA, vu au TP0 : le modèle de base reste **gelé**. On ne
livre que le **patch**, et n'importe qui peut le regreffer sur le même modèle de
base.

### Vérifier ton livrable avant de le remettre

```
make evaluer
uv run python evaluer_livrable.py livrables/livrable_<groupe>.zip --demo
```

Ces commandes font exactement ce que fera la personne qui reçoit ton livrable :
elles l'ouvrent **sans MLflow ni Docker**, retéléchargent le modèle de base, y
greffent ton adaptateur et remesurent ton test. Si elles échouent ici, elles
échoueront chez elle.

Les scores doivent correspondre à ceux de `make juger`. Un petit écart reste
possible, à cause du passage en 16 bits.

**Remets uniquement le fichier `.zip`.** Tout le reste y est déjà.

---

## 9. Ce que ton livrable doit montrer

- **Un sujet clair** : une ontologie fermée, bien dimensionnée, des intentions
  qui ne se chevauchent pas. Sinon, personne ne peut écrire ton test croisé.
- **Des données soignées** : variées, propres, `make valider` sans erreur,
  source et licence citées. C'est d'elles que vient la qualité du modèle.
- **Une démarche honnête** : réglages et prompt choisis sur la validation, test
  lu peu de fois. Le rapport le montre, avec `lectures_test`.
- **Un livrable qui fonctionne** : léger, il se réévalue sans erreur et l'on y
  retrouve les scores déclarés.
- **Un résultat qui tient** sur le test croisé, cohérent avec ton propre test.
  C'est la preuve que ton modèle généralise.
- **Du recul** dans tes notes : limites, et écart entre ton test et le test
  croisé.

Et hors du livrable : **le test croisé que tu as écrit** pour un autre groupe,
soigné et varié. C'est lui qui juge leur travail.

---

## 10. À retenir

- **Les données sont le vrai travail.** Le cycle est celui du TP3 ; ce qui
  change, c'est le dossier `donnees/`.
- **Un test écrit par quelqu'un d'autre**, et que tu ne vois pas, est le seul
  vrai juge.
- **On livre le patch, pas le modèle :** quelques mégaoctets, plus la référence
  exacte d'un modèle de base public.

---

## 11. Bonus (au choix)

- **Un second domaine** : entraîne un second adaptateur sur d'autres données, avec
  le **même** modèle de base. Deux patchs de 2 Mo, un seul modèle de 1,5 Go.
- **L'essayer dans Ollama** : fusionne ton adaptateur et convertis-le, comme dans
  le bonus du TP1. Le résultat (~1,2 Go) reste **chez toi** : on ne le livre pas.

---

## 12. Dépannage

- **« ontologie.yaml est encore le MODÈLE ».** Remplace le domaine et les
  intentions d'exemple par les tiens, ou tape `make exemple-domotique`.
- **« la 1re ligne doit être exactement phrase;reponse ».** Ton tableur a
  enregistré avec des virgules. Réenregistre en CSV, séparateur
  **point-virgule**, encodage UTF-8.
- **« intention … absente de l'ontologie. Il manque un « | » ? »** Tu as écrit
  `reserver personnes=quatre` au lieu de `reserver | personnes=quatre`.
- **`importer.py` : « Colonne … introuvable ».** Le message liste les colonnes
  disponibles : relance avec les bons noms dans `--texte` et `--etiquette`.
- **`importer.py` : « … intentions trouvées : c'est trop ».** Garde-en quelques-unes
  avec `--nb-intentions 8` ou `--intentions a,b,c`.
- **`juger.py` : « Aucun run mesuré sur la validation ».** Lance d'abord `make run`.
- **`livrer.py` : « Ce run n'a pas encore été jugé ».** Lance `make juger` avant
  `make livrer`.
- **Page « Something went wrong » dans MLflow.** Clique sur **`Model training`**
  en haut à gauche avant d'ouvrir l'expérience.
- **Port 5000, 9000 ou 9001 occupé.** La plateforme du TP3 tourne encore :
  `make clean` dans le dossier `tp3`.
