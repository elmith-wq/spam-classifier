"""TF-IDF + logistic regression baseline for SMS spam."""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/SMSSpamCollection", sep="\t", names=["label", "text"])
print(len(df), "messages;", df.text.duplicated().sum(), "exact duplicates")

# Dedupe BEFORE splitting so the same text can't be in train and test
df = df.drop_duplicates("text").reset_index(drop=True)
df["y"] = (df.label == "spam").astype(int)
print(len(df), "after dedupe;", f"{df.y.mean():.1%} spam")

train, test = train_test_split(df, test_size=0.2, stratify=df.y, random_state=42)

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
Xtr, Xte = vec.fit_transform(train.text), vec.transform(test.text)

clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, train.y)
prob = clf.predict_proba(Xte)[:, 1]
print("ROC AUC:", round(roc_auc_score(test.y, prob), 4))
print(classification_report(test.y, prob >= 0.5, target_names=["ham", "spam"]))