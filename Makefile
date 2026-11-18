.PHONY: setup health ui exemple-domotique valider run prompt-systeme prompt-exemples juger livrer evaluer test clean

setup:          ## Démarrer MLflow + RustFS (Docker) et attendre qu'ils soient prêts
	docker compose up -d
	@echo "Attente de MLflow (1re fois : quelques minutes, le temps d'installer boto3)..."
	@for i in $$(seq 1 60); do curl -sf http://localhost:5000/health >/dev/null && echo "MLflow est prêt : http://localhost:5000" && break; sleep 5; done
	docker compose ps -a

health:         ## État des services (-a : affiche aussi rustfs-setup, terminé)
	docker compose ps -a

ui:             ## Rappel des URLs
	@echo "MLflow : http://localhost:5000  (mode Model training)"
	@echo "RustFS : http://localhost:9001  (rustfsadmin / rustfsadmin)"

exemple-domotique: ## Partir de l'exemple domotique (copie exemples/domotique/ dans donnees/)
	@test "$$(wc -l < donnees/train.csv)" -le 1 || (echo "donnees/train.csv contient déjà tes données : rien n'est écrasé." && exit 1)
	cp exemples/domotique/* donnees/
	@echo "Exemple domotique copié dans donnees/ : complète train.csv, validation.csv et test.csv."

valider:        ## Vérifier TES données (ontologie + CSV) avant d'entraîner
	uv run python valider_donnees.py

run:            ## RÉGLER : entraîner + mesurer sur la VALIDATION (ne lit pas le test)
	uv run python valider_donnees.py
	uv run python main.py

prompt-systeme: ## Labo : prompt généré depuis l'ontologie, sur la VALIDATION
	uv run python prompt_lab.py --construire systeme
	uv run python prompt_lab.py prompts/systeme.txt validation

prompt-exemples: ## Labo : prompt + 1 exemple par intention (TRAIN), sur la VALIDATION
	uv run python prompt_lab.py --construire exemples-train
	uv run python prompt_lab.py prompts/exemples_train.txt validation

juger:          ## JUGER : porte de qualité sur TON test (meilleur run selon la validation)
	uv run python juger.py

livrer:         ## LIVRER : livrables/livrable_<groupe>.zip (adaptateur + données + rapport)
	uv run python livrer.py

evaluer:        ## Vérifier son livrable comme le fera la personne qui le reçoit
	uv run python evaluer_livrable.py livrables/livrable_$$(grep '^GROUPE=' .env | cut -d= -f2).zip

test:           ## Tests unitaires (lecture des données, score)
	uv run --with pytest pytest -q

clean:          ## Tout arrêter (les runs sont conservés)
	docker compose down
