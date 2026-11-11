# SMS Spam Classifier: TF-IDF vs DistilBERT

Compares a TF-IDF + logistic regression baseline with a fine-tuned DistilBERT on the UCI SMS Spam Collection, with a Gradio demo.

## Results

Held-out test set: 1,034 unique messages (131 spam), same split for both models.

| | TF-IDF + LogReg | DistilBERT |
|---|---|---|
| ROC AUC | 0.989 | **0.993** |
| Log loss | 0.165 | **0.061** |
| Spam precision | 0.90 | **1.00** |
| Spam recall | **0.92** | 0.89 |
| Errors | 24 | **14** |

DistilBERT is better calibrated and never flagged a genuine message as spam, at the cost of missing a few more spam messages. The baseline is already strong, so the gain in AUC is small, and a transformer is far heavier to train and run.

## Method

- 5,572 messages, 403 exact duplicates removed **before** splitting so identical text cannot appear in both train and test (5,169 remain, 12.6% spam).
- Stratified 80/20 split, fixed seed.
- DistilBERT fine-tuned for 3 epochs (AdamW, lr 2e-5). No hyperparameters were tuned on the test set.
- Accuracy is not the headline metric because predicting "not spam" every time scores about 87%.

## Error analysis

[Your sentence or two on the spam both models missed.]

## Limitations

- One random split and only 131 spam messages, so a difference of a few errors is suggestive, not conclusive. Cross-validation would firm this up.
- Near-duplicate messages (not just exact ones) may remain.
- The dataset is small, old and English-only, so results may not transfer to modern spam.

## Reproduce

1. Create a Python 3.11 environment and install PyTorch plus `requirements.txt`.
2. Download the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) into `data/`.
3. `python src/baseline.py`, then `python src/finetune.py`
4. `python app.py` for the demo (needs the model saved by `finetune.py`).