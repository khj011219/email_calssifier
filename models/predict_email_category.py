from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import json

MODEL_DIR = os.path.join(os.path.dirname(__file__), '../bert_manual_300_model/checkpoint-264')
LABEL_MAP_PATH = os.path.join(os.path.dirname(__file__), '../bert_manual_300_model/label_mapping.json')

# 전역 변수로 모델과 토크나이저 저장
_tokenizer = None
_model = None
_id2label = None

def _load_model():
    """모델을 지연 로딩하는 함수"""
    global _tokenizer, _model, _id2label
    
    if _tokenizer is None:
        try:
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
            _model.eval()
            
            # 라벨 매핑 로드
            with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
                label_map = json.load(f)
                # 숫자인 키만 int로 변환
                _id2label = {int(k): v for k, v in label_map.items() if k.isdigit()}
        except Exception as e:
            print(f"모델 로딩 오류: {e}")
            # 기본값 설정
            _id2label = {0: "Work", 1: "Personal", 2: "Advertisement"}

def predict_category(subject, body):
    """이메일 분류 예측"""
    # 모델이 로드되지 않았으면 로드
    _load_model()
    
    if _tokenizer is None or _model is None:
        # 모델 로딩 실패 시 기본값 반환
        return "Work", 0.5
    
    text = f"{subject} {body}"
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    
    with torch.no_grad():
        outputs = _model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).squeeze()
        pred_id = int(torch.argmax(probs).item())
        confidence = float(probs[pred_id].item())
        label = _id2label.get(pred_id, "Unknown")
    
    return label, confidence 