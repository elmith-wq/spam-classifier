"""Fine-tune DistilBERT on SMS spam and compare with TF-IDF (identical split)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL, EPOCHS, BATCH, LR, MAXLEN = "distilbert-base-uncased", 3, 32, 2e-5, 128
OUT = Path("results")
OUT.mkdir(exist_ok=True)
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Same dedupe + split as baseline.py
df = pd.read_csv("data/SMSSpamCollection", sep="\t", names=["label", "text"])
df = df.drop_duplicates("text").reset_index(drop=True)
df["y"] = (df.label == "spam").astype(int)
train, test = train_test_split(df, test_size=0.2, stratify=df.y, random_state=42)

# Baseline
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
lr = LogisticRegression(max_iter=1000, class_weight="balanced")
lr.fit(vec.fit_transform(train.text), train.y)
p_tfidf = lr.predict_proba(vec.transform(test.text))[:, 1]

# DistilBERT
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
texts, labels = train.text.tolist(), train.y.values


def encode(batch_texts):
    return tok(batch_texts, truncation=True, max_length=MAXLEN,
               padding=True, return_tensors="pt").to(device)


for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(len(texts)).tolist()
    total = 0.0
    for i in range(0, len(perm), BATCH):
        idx = perm[i:i + BATCH]
        enc = encode([texts[j] for j in idx])
        y = torch.tensor(labels[idx], device=device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            loss = model(**enc, labels=y).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(idx)
    print(f"epoch {epoch}: train_loss={total / len(texts):.4f}")

model.eval()
probs = []
with torch.no_grad():
    tt = test.text.tolist()
    for i in range(0, len(tt), 64):
        logits = model(**encode(tt[i:i + 64])).logits.float()
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
p_bert = np.concatenate(probs)

# Compare
y = test.y.values
EPS = 1e-4
rows = {}
for name, p in [("TF-IDF + LogReg", p_tfidf), ("DistilBERT", p_bert)]:
    pred = (p >= 0.5).astype(int)
    rows[name] = {
        "auc": round(roc_auc_score(y, p), 4),
        "log_loss": round(log_loss(y, np.clip(p, EPS, 1 - EPS)), 4),
        "spam_precision": round(precision_score(y, pred), 4),
        "spam_recall": round(recall_score(y, pred), 4),
        "errors": int((pred != y).sum()),
    }
print(pd.DataFrame(rows).T.to_string())
(OUT / "metrics.json").write_text(json.dumps(rows, indent=2))

# Messages either model gets wrong, for error analysis
res = test.assign(p_tfidf=p_tfidf.round(3), p_bert=p_bert.round(3))
wrong = res[((res.p_tfidf >= 0.5) != (res.y == 1)) | ((res.p_bert >= 0.5) != (res.y == 1))]
wrong[["label", "text", "p_tfidf", "p_bert"]].to_csv(OUT / "errors.csv", index=False)
print(f"\n{len(wrong)} messages misclassified by at least one model -> results/errors.csv")

model.save_pretrained("outputs/distilbert-spam")
tok.save_pretrained("outputs/distilbert-spam")