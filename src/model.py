# src/model.py
"""Charger le modèle de base et y greffer le LoRA. (Deux responsabilités
séparées, contrairement au TP0 où tout était mélangé dans le script.)"""
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


def load_base_model(model_name: str):
    """Charge le modèle + tokenizer. float32 sans device_map = rapide sur CPU."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    # Le modèle est maintenant en cache : on coupe les accès à Hugging Face pour la
    # suite. Sinon, chaque sauvegarde de checkpoint interroge le Hub (pour vérifier
    # la taille du vocabulaire) : lent, et un avertissement si le réseau hoquette.
    os.environ["HF_HUB_OFFLINE"] = "1"
    return model, tokenizer


def apply_lora(model, r: int, alpha: int, dropout: float):
    """Greffe le patch LoRA (on n'entraîne que ~0,2 % des paramètres)."""
    config = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model
