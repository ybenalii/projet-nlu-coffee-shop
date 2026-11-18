# src/generate.py
"""Générer une réponse à une question. Option `system` pour donner une consigne
au modèle de base (la colonne 'base+prompt' de la porte)."""


def generate_answer(model, tokenizer, question: str, system: str | None = None,
                    max_new_tokens: int = 40) -> str:
    # Mode évaluation : coupe le dropout du LoRA, pour des réponses (et donc
    # des scores) identiques d'un lancement à l'autre.
    model.eval()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})
    entree = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False,
        return_tensors="pt", return_dict=True,
    )
    sortie = model.generate(
        **entree, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    longueur = entree["input_ids"].shape[-1]
    return tokenizer.decode(sortie[0][longueur:], skip_special_tokens=True).strip()
