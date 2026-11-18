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
| 250 | 83 % | – | 58 % | – |

At 150 steps the model had seen each of the 120 training sentences only 2.5 times and the scores were still rising, so a 250-step run (about 4 passes) was added. The best checkpoint on the validation set is kept.

Selected: 250 steps (best val_intent_acc).

Observations on the validation predictions: at 30 steps the model mostly repeats the sentence instead of answering with a label. At 100 steps it often falls back on order_food and sometimes invents labels outside the ontology (order_juice, "modify | milk | almond").

No work on the system prompt: the base+prompt column uses the prompt generated from the ontology.

## Test results

| | LoRA | base | base+prompt |
| --- | --- | --- | --- |
| intent_acc | 67 % | 0 % | 0 % |
| slot_f1 | 68 % | 17 % | 9 % |
| exact | 46 % | 0 % | 0 % |

Gate: FAIL. The margin condition is met (+67 points over base+prompt, 30 required) but intent_acc is 67 % for a 70 % threshold: one more correct sentence out of 24 would have passed. Without fine-tuning the base model finds no intent at all, even with the system prompt, so fine-tuning is clearly useful here; it is not yet reliable enough for the threshold.

The gap between validation (83 %) and test (67 %) is expected: settings were chosen on the validation set, and both sets are small (24 sentences, about 4 points per sentence).

Test reads: 3. The first two runs of make juger were interrupted before any result was displayed; only the last one completed. No choice was made from a test score: 250 steps was selected on the validation set.
Data changed after reading the test: no.

## Limits

- 24 test sentences: one error ≈ 4 points.
- Scores were still rising at 250 steps; longer training or more data would probably pass the gate.
- No order with several items in the same sentence.
- An off-topic sentence ("bonjour") will still be classified into one of the 8 intents.
