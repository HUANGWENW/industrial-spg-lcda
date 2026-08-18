# MVP experiment registry

| ID | Configuration | Purpose |
|---|---|---|
| E0 | Smoke test | Validate environment and configuration loading |
| E1 | Synthetic-only | Establish detector baseline |
| E2 | Domain randomization | Measure rendering diversity benefit |
| E3 | Generic CLIP prompt | Control for generic language guidance |
| E4 | Structured prompt without shift loss | Measure prompt-content benefit |
| E5 | E1 + SPG-PDA | Measure simulator-grounded shift alignment |
| E6 | E5 + IHN-PB | Measure industrial hard-negative suppression |
| E7 | E6 + LC-SCDA | Measure localization consistency and full method |

Formal runs use at least three seeds and evaluate mAP50, AP75, precision, recall,
F1, false positives per 100 images, hard-negative false-positive rate, IoU, and
box-center error.

