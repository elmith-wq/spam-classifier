import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PATH = "outputs/distilbert-spam"
tok = AutoTokenizer.from_pretrained(PATH)
model = AutoModelForSequenceClassification.from_pretrained(PATH).eval()


def predict(text):
    enc = tok(text, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        p = torch.softmax(model(**enc).logits, dim=1)[0]
    return {"spam": float(p[1]), "not spam": float(p[0])}


gr.Interface(
    predict, gr.Textbox(lines=3, label="SMS message"), gr.Label(),
    title="SMS Spam Detector",
    description="DistilBERT fine-tuned on the UCI SMS Spam Collection.",
    examples=[["WINNER!! You have been selected for a free prize. Call now to claim."],
              ["Hey, are we still meeting at 6 tonight?"]],
).launch()