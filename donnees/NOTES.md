# Notes

## Topic

Taking orders at the counter of a coffee shop. The model turns what the customer says (in French) into a ticket for the barista.

8 intents: order_coffee, order_tea, order_other_drink, order_food, modify_order, cancel_order, pay, menu_question.
6 slots: item, quantity, size, milk, extra, service.

## Data

Handwritten sentences, no personal data.

- train: 120 (15 per intent)
- validation: 24 (3 per intent)
- test: 24 (3 per intent)

`make valider`: 0 errors, 0 warnings.

## Conventions

- Anything espresso-based, hot or iced (latte, noisette, mocha, iced latte), is coffee. Chai latte and matcha latte are tea.
- Price of one item → menu_question. Total of the order → pay.
- "vous avez du déca" is a question, "un déca" is an order.
- quantity only when there is more than one item. One extra per sentence. pay has no slot.
- Slot values are copied from the sentence, lowercase, without article.
- One sentence = one item.

## Tuning (validation set)

| max_steps | val_intent_acc | val_slot_f1 | val_exact | best_val_loss |
| --- | --- | --- | --- | --- |
| 30 | 0 % | 17 % | 0 % | 3.20 |
| 60 | 8 % | 40 % | 8 % | 2.21 |
| 100 | 42 % | 51 % | 25 % | 1.54 |
| 150 | 67 % | 71 % | 38 % | 1.28 |

Scores are still rising at 150 steps (only 2.5 passes over the 120 training sentences), so longer runs are tested before reading the test set. The best checkpoint on the validation set is kept, so more steps cannot make the selected model worse on validation.

Selected: … steps.

Observations on the validation predictions: at 30 steps the model mostly repeats the sentence instead of answering with a label. At 100 steps it often falls back on order_food and sometimes invents labels outside the ontology (order_juice, "modify | milk | almond").

No work on the system prompt: the base+prompt column uses the prompt generated from the ontology.

## Test results

| | LoRA | base | base+prompt |
| --- | --- | --- | --- |
| intent_acc | | | |
| slot_f1 | | | |
| exact | | | |

Gate: …
Test reads: …
Data changed after reading the test: …

## Limits

- 24 test sentences: one error ≈ 4 points.
- No order with several items in the same sentence.
- An off-topic sentence ("bonjour") will still be classified into one of the 8 intents.
