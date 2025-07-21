from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import os

# === Uygulama ===
app = FastAPI()

# === Model yolu ===
MODEL_PATH = "./app/model"  # model dosyaların burada olmalı

# === Model ve Tokenizer yükleniyor ===
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
model.eval()

# === API input modeli ===
class QARequest(BaseModel):
    article: str
    question: str
    options: list[str]  # ['A seçeneği', 'B seçeneği', ...]

# === Prediction endpoint ===
@app.post("/predict")
def predict(req: QARequest):
    input_text = f"article: {req.article.strip()} question: {req.question.strip()} options: {' || '.join(req.options)}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_length=10)
        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return {"answer": answer}
