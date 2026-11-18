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

| max_steps | val_intent_acc | val_exact | best_val_loss |
| --- | --- | --- | --- |
| 30 | | | |
| 60 | | | |
| 100 | | | |
| 150 | | | |

Selected: … steps.

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
