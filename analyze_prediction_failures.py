import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

def analyze_prediction_failures():
    """예측 실패 원인을 분석합니다."""
    
    print("="*60)
    print("예측 실패 원인 분석")
    print("="*60)
    
    # 1. 데이터 품질 분석
    print("\n1. 데이터 품질 분석")
    print("-" * 30)
    
    # 원본 데이터 로드
    df_original = pd.read_csv('dataset/test_labeled_dataset.csv')
    df_oversampled = pd.read_csv('dataset/test_labeled_dataset_oversampled.csv')
    
    print(f"원본 데이터: {len(df_original)}개")
    print(f"오버샘플링 데이터: {len(df_oversampled)}개")
    
    # 텍스트 길이 분석
    df_original['text_length'] = df_original['body'].str.len()
    print(f"\n텍스트 길이 통계:")
    print(f"평균: {df_original['text_length'].mean():.1f}")
    print(f"중간값: {df_original['text_length'].median():.1f}")
    print(f"최소: {df_original['text_length'].min()}")
    print(f"최대: {df_original['text_length'].max()}")
    
    # 라벨별 텍스트 길이
    print(f"\n라벨별 평균 텍스트 길이:")
    for label in df_original['label'].unique():
        avg_length = df_original[df_original['label'] == label]['text_length'].mean()
        print(f"{label}: {avg_length:.1f}")
    
    # 2. 텍스트 내용 분석
    print(f"\n2. 텍스트 내용 분석")
    print("-" * 30)
    
    # 특수 문자, URL, 이메일 패턴 분석
    def analyze_text_patterns(text):
        patterns = {
            'urls': len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)),
            'emails': len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)),
            'numbers': len(re.findall(r'\d+', text)),
            'uppercase_words': len(re.findall(r'\b[A-Z]{2,}\b', text)),
            'special_chars': len(re.findall(r'[^\w\s]', text))
        }
        return patterns
    
    # 샘플 분석
    sample_texts = df_original.sample(min(10, len(df_original)))
    print(f"샘플 텍스트 패턴 분석 (10개):")
    for idx, row in sample_texts.iterrows():
        patterns = analyze_text_patterns(row['body'])
        print(f"라벨: {row['label']}, 길이: {len(row['body'])}, URL: {patterns['urls']}, 이메일: {patterns['emails']}")
    
    # 3. 라벨링 품질 분석
    print(f"\n3. 라벨링 품질 분석")
    print("-" * 30)
    
    # 라벨별 키워드 분석
    def extract_keywords(text, top_n=10):
        # 간단한 키워드 추출 (실제로는 더 정교한 방법 사용)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return Counter(words).most_common(top_n)
    
    print("라벨별 주요 키워드:")
    for label in df_original['label'].unique():
        label_texts = df_original[df_original['label'] == label]['body'].tolist()
        all_words = []
        for text in label_texts:
            all_words.extend([word for word in re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())])
        
        word_counts = Counter(all_words)
        print(f"\n{label} 라벨 주요 키워드:")
        for word, count in word_counts.most_common(10):
            print(f"  {word}: {count}")
    
    # 4. 모델 예측 분석
    print(f"\n4. 모델 예측 분석")
    print("-" * 30)
    
    # 가장 성능이 좋은 모델로 예측 분석
    model_path = "bert_manual_300_model/checkpoint-264"
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model.eval()
        model.to(device)
        
        label2id = {"Work": 0, "Personal": 1, "Advertisement": 2}
        id2label = {0: "Work", 1: "Personal", 2: "Advertisement"}
        
        # 예측 수행
        predictions = []
        confidences = []
        true_labels = []
        
        for idx, row in df_original.iterrows():
            text = row['body']
            true_label = row['label']
            
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                predicted_id = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_id].item()
            
            predicted_label = id2label[predicted_id]
            
            predictions.append(predicted_label)
            confidences.append(confidence)
            true_labels.append(true_label)
        
        # 예측 결과 분석
        df_analysis = pd.DataFrame({
            'true_label': true_labels,
            'predicted_label': predictions,
            'confidence': confidences,
            'text_length': df_original['text_length'],
            'correct': [t == p for t, p in zip(true_labels, predictions)]
        })
        
        print(f"전체 정확도: {df_analysis['correct'].mean():.4f}")
        
        # 라벨별 정확도
        print(f"\n라벨별 정확도:")
        for label in df_analysis['true_label'].unique():
            label_data = df_analysis[df_analysis['true_label'] == label]
            accuracy = label_data['correct'].mean()
            avg_confidence = label_data['confidence'].mean()
            print(f"{label}: 정확도 {accuracy:.4f}, 평균 확신도 {avg_confidence:.4f}")
        
        # 오분류된 케이스 분석
        incorrect_cases = df_analysis[~df_analysis['correct']]
        print(f"\n오분류된 케이스 수: {len(incorrect_cases)}")
        
        if len(incorrect_cases) > 0:
            print(f"\n오분류 패턴 분석:")
            confusion = pd.crosstab(incorrect_cases['true_label'], incorrect_cases['predicted_label'])
            print(confusion)
            
            # 가장 확신도가 높은 오분류 케이스
            high_confidence_errors = incorrect_cases.nlargest(5, 'confidence')
            print(f"\n높은 확신도로 오분류된 케이스 (상위 5개):")
            for idx, row in high_confidence_errors.iterrows():
                original_idx = df_original.index[idx]
                text_preview = df_original.loc[original_idx, 'body'][:100] + "..."
                print(f"실제: {row['true_label']} -> 예측: {row['predicted_label']} (확신도: {row['confidence']:.4f})")
                print(f"텍스트: {text_preview}")
                print()
        
        # 텍스트 길이와 정확도 관계
        print(f"\n텍스트 길이와 정확도 관계:")
        df_analysis['length_bin'] = pd.cut(df_analysis['text_length'], bins=5)
        length_accuracy = df_analysis.groupby('length_bin')['correct'].mean()
        print(length_accuracy)
        
    except Exception as e:
        print(f"모델 분석 중 오류 발생: {e}")
    
    # 5. 개선 제안
    print(f"\n5. 개선 제안")
    print("-" * 30)
    
    print("1. 데이터 품질 개선:")
    print("   - 라벨링 기준 통일 및 검증")
    print("   - 텍스트 전처리 강화 (특수문자, URL, 이메일 처리)")
    print("   - 더 많은 다양한 도메인의 데이터 수집")
    
    print("\n2. 모델 개선:")
    print("   - 더 큰 모델 사용 (RoBERTa-large, DeBERTa 등)")
    print("   - 데이터 증강 기법 적용")
    print("   - 앙상블 모델 사용")
    
    print("\n3. 평가 방법 개선:")
    print("   - 도메인별 성능 분석")
    print("   - 에러 케이스 심층 분석")
    print("   - 인간 평가와의 비교")

if __name__ == "__main__":
    analyze_prediction_failures() 