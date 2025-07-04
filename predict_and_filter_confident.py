import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from sklearn.metrics import classification_report
import os

def load_trained_model():
    """학습된 모델과 토크나이저를 로드합니다."""
    print("학습된 모델 로딩 중...")
    
    # 모델과 토크나이저 로드 (최신 checkpoint 사용)
    model_path = "./bert_manual_300_model/checkpoint-207"  # 최신 checkpoint
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # 라벨 매핑 로드
    with open("./bert_manual_300_model/label_mapping.json", 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)
    
    id_to_label = label_mapping['id_to_label']
    label_to_id = label_mapping['label_to_id']
    
    print(f"라벨 매핑: {id_to_label}")
    return model, tokenizer, id_to_label, label_to_id

def predict_with_confidence(model, tokenizer, texts, batch_size=8, max_length=512):
    """텍스트에 대해 예측과 신뢰도를 계산합니다."""
    model.eval()
    predictions = []
    confidences = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # 토큰화
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # 예측
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            
            # 예측 클래스와 신뢰도
            pred_labels = torch.argmax(probabilities, dim=1)
            max_probs = torch.max(probabilities, dim=1)[0]
            
            predictions.extend(pred_labels.cpu().numpy())
            confidences.extend(max_probs.cpu().numpy())
    
    return predictions, confidences

def filter_high_confidence_data(df, predictions, confidences, confidence_threshold=0.8):
    """높은 신뢰도를 가진 예측만 필터링합니다."""
    print(f"신뢰도 임계값: {confidence_threshold}")
    
    # 신뢰도가 임계값을 넘는 인덱스 찾기
    high_conf_indices = [i for i, conf in enumerate(confidences) if conf >= confidence_threshold]
    
    print(f"전체 데이터: {len(df)}")
    print(f"높은 신뢰도 데이터: {len(high_conf_indices)}")
    print(f"필터링 비율: {len(high_conf_indices)/len(df)*100:.2f}%")
    
    # 필터링된 데이터 생성
    filtered_df = df.iloc[high_conf_indices].copy()
    filtered_df['predicted_label'] = [predictions[i] for i in high_conf_indices]
    filtered_df['confidence'] = [confidences[i] for i in high_conf_indices]
    
    return filtered_df

def analyze_filtered_data(filtered_df, id_to_label):
    """필터링된 데이터를 분석합니다."""
    print("\n=== 필터링된 데이터 분석 ===")
    
    # 라벨별 분포
    label_counts = filtered_df['predicted_label'].value_counts()
    print("\n예측 라벨 분포:")
    for label_id, count in label_counts.items():
        label_name = id_to_label[str(label_id)]
        print(f"  {label_name}: {count}개")
    
    # 신뢰도 분포
    print(f"\n신뢰도 통계:")
    print(f"  평균: {filtered_df['confidence'].mean():.4f}")
    print(f"  최소: {filtered_df['confidence'].min():.4f}")
    print(f"  최대: {filtered_df['confidence'].max():.4f}")
    
    return filtered_df

def main():
    print("=== 신뢰도 기반 데이터 필터링 시작 ===")
    
    # 1. 학습된 모델 로드
    model, tokenizer, id_to_label, label_to_id = load_trained_model()
    
    # 2. 파싱된 데이터 로드
    print("\n파싱된 데이터 로딩 중...")
    df = pd.read_csv('dataset/emails_processed.csv', nrows=1000)
    print(f"파싱된 데이터 크기: {len(df)} (상위 1000개만 사용)")
    print(f"결측 라벨 수: {df['label'].isna().sum()}")
    
    # 본문과 라벨만 사용
    texts = df['body'].astype(str).tolist()
    labels = df['label'].tolist() if 'label' in df.columns else None
    
    # 3. 예측 수행
    print("\n예측 수행 중...")
    predictions, confidences = predict_with_confidence(model, tokenizer, texts)
    
    # 4. 높은 신뢰도 데이터 필터링
    print("\n높은 신뢰도 데이터 필터링 중...")
    confidence_threshold = 0.9  # 90% 신뢰도 임계값
    filtered_df = filter_high_confidence_data(df, predictions, confidences, confidence_threshold)
    
    # 5. 필터링된 데이터 분석
    filtered_df = analyze_filtered_data(filtered_df, id_to_label)
    
    # 6. 결과 저장
    print("\n결과 저장 중...")
    
    # 예측 라벨을 실제 라벨명으로 변환
    filtered_df['predicted_label_name'] = filtered_df['predicted_label'].apply(
        lambda x: id_to_label[str(x)]
    )
    
    # 최종 데이터셋 (본문 + 예측 결과)
    final_df = filtered_df[['body', 'predicted_label_name', 'confidence']].copy()
    final_df.columns = ['body', 'label', 'confidence']
    
    # 저장
    output_path = 'dataset/filtered_high_confidence.csv'
    final_df.to_csv(output_path, index=False)
    
    print(f"\n=== 필터링 완료 ===")
    print(f"필터링된 데이터가 '{output_path}'에 저장되었습니다.")
    print(f"총 {len(final_df)}개의 높은 신뢰도 샘플이 선택되었습니다.")
    
    # 신뢰도별 분포 시각화
    print(f"\n신뢰도 분포:")
    confidence_ranges = [(0.8, 0.85), (0.85, 0.9), (0.9, 0.95), (0.95, 1.0)]
    for low, high in confidence_ranges:
        count = len(final_df[(final_df['confidence'] >= low) & (final_df['confidence'] < high)])
        print(f"  {low:.2f}-{high:.2f}: {count}개")
    
    return final_df

if __name__ == "__main__":
    main() 