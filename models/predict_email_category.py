from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import json

MODEL_DIR = os.path.join(os.path.dirname(__file__), '../bert_manual_300_model/checkpoint-264')
LABEL_MAP_PATH = os.path.join(os.path.dirname(__file__), '../bert_manual_300_model/label_mapping.json')

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# 라벨 매핑 로드
with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
    label_map = json.load(f)
    # 숫자인 키만 int로 변환
    id2label = {int(k): v for k, v in label_map.items() if k.isdigit()}

def predict_category(subject, body):
    text = f"{subject} {body}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).squeeze()
        pred_id = int(torch.argmax(probs).item())
        confidence = float(probs[pred_id].item())
        label = id2label.get(pred_id, "Unknown")
    return label, confidence 