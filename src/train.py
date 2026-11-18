# src/train.py
"""
L'entraînement, avec une NOUVEAUTÉ par rapport au TP0 : un jeu de VALIDATION.

Le Trainer évalue la perte sur la validation tous les N pas (eval_loss), et
`load_best_model_at_end=True` recharge le MEILLEUR état (pas forcément le
dernier). On ne se contente plus de la dernière version : on garde la meilleure,
mesurée sur des exemples non vus.
"""
from transformers import (
    Trainer, TrainingArguments, DataCollatorForLanguageModeling,
)

from src.config import settings
from src.data import format_chat


def _tokenize(dataset, tokenizer):
    textes = [format_chat(ex, tokenizer) for ex in dataset]
    enc = [tokenizer(t, truncation=True, max_length=192) for t in textes]
    return dataset.from_list([
        {"input_ids": e["input_ids"], "attention_mask": e["attention_mask"]} for e in enc
    ])


def train(model, tokenizer, train_ds, val_ds):
    train_tok = _tokenize(train_ds, tokenizer)
    val_tok = _tokenize(val_ds, tokenizer)

    args = TrainingArguments(
        output_dir="./sortie_entrainement",
        per_device_train_batch_size=2,
        learning_rate=settings.train_learning_rate,
        max_steps=settings.train_max_steps,
        logging_steps=1,
        # --- évaluation sur la VALIDATION ---
        eval_strategy="steps",
        eval_steps=settings.train_eval_steps,
        save_strategy="steps",
        save_steps=settings.train_eval_steps,
        load_best_model_at_end=True,        # recharge le meilleur état
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        dataloader_pin_memory=False,
        report_to="none",
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_tok, eval_dataset=val_tok,
        data_collator=collator,
    )
    result = trainer.train()

    best = trainer.state.best_metric
    if best is not None:
        print(f"\nMeilleure perte de validation : {best:.4f}")
    # On renvoie les MESURES pour pouvoir les loguer (TP2)
    # Attention : result.training_loss est la MOYENNE sur toutes les étapes.
    # La perte du dernier pas se lit dans l'historique des logs du Trainer.
    pertes = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    metrics = {
        "best_val_loss": float(best) if best is not None else None,
        "final_train_loss": float(pertes[-1]) if pertes else None,
        "mean_train_loss": float(result.training_loss),
    }
    return model, metrics
