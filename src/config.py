# src/config.py
"""
Configuration du projet, lue depuis .env.

TP2 : les réglages MLflow/RustFS et configure_environment(), qui pose les
identifiants S3 dans l'environnement (boto3/MLflow les lisent pour envoyer les
artefacts).
TP3 : les réglages de la porte de qualité et le fichier du prompt système.
Exercice (sujet libre) : le dossier des données (DONNEES_DIR) et le nom du groupe.
"""
import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )

    # Le groupe (apparaît dans MLflow et dans le nom du livrable)
    groupe: str = Field(default="groupe-x")
    # Les données : ontologie.yaml + train.csv / validation.csv / test.csv
    donnees_dir: str = Field(default="donnees")

    # Modèle
    model_name: str = Field(default="Qwen/Qwen3-0.6B")

    # LoRA
    lora_r: int = Field(default=8, ge=1, le=128)
    lora_alpha: int = Field(default=16)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0)

    # Entraînement
    train_max_steps: int = Field(default=150, ge=1)   # la NLU demande ~150 pas
    train_learning_rate: float = Field(default=2e-4, gt=0.0)
    train_eval_steps: int = Field(default=10, ge=1)

    # --- Porte de qualité (nouveau au TP3) ---
    eval_gate_min_intent: float = Field(default=0.70, ge=0.0, le=1.0)  # seuil LoRA
    eval_gate_margin: float = Field(default=0.30, ge=0.0, le=1.0)      # avance / base+prompt
    eval_gate_min_slot_f1: float = Field(default=0.0, ge=0.0, le=1.0)  # 0 = pas d'exigence sur les slots
    eval_gate_strict: bool = Field(default=False)   # True -> un FAIL arrête en erreur
    # Le prompt système de la colonne base+prompt (choisi sur la VALIDATION)
    prompt_systeme_file: str = Field(default="prompts/systeme.txt")

    # --- MLflow / RustFS (TP2) ---
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    mlflow_experiment_name: str = Field(default="projet-nlu")
    mlflow_prompt_experiment_name: str = Field(default="projet-prompt-lab")
    mlflow_s3_endpoint_url: str = Field(default="http://localhost:9000")
    aws_access_key_id: str = Field(default="rustfsadmin")
    aws_secret_access_key: str = Field(default="rustfsadmin")

    def configure_environment(self):
        """Pose les identifiants S3 (RustFS) dans l'environnement (pour boto3/MLflow)."""
        os.environ["AWS_ACCESS_KEY_ID"] = self.aws_access_key_id
        os.environ["AWS_SECRET_ACCESS_KEY"] = self.aws_secret_access_key
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = self.mlflow_s3_endpoint_url


settings = Settings()
