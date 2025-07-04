import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import json
import os

def load_and_prepare_balanced_data():
    """균형 잡힌 데이터를 로드하고 전처리합니다."""
    print("균형 잡힌 데이터 로딩 중...")
    
    # 균형 잡힌 데이터 로드
    df = pd.read_csv('dataset/manual_labeled_balanced.csv')
    
    # 라벨 매핑 생성
    unique_labels = sorted(df['label'].unique())
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    
    print(f"균형 잡힌 데이터 수: {len(df)}")
    print(f"라벨 분포:")
    print(df['label'].value_counts())
    print(f"라벨 매핑: {label_to_id}")
    
    # train_test_split 수행 (80:20 비율)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['body'].tolist(),
        df['label'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )
    
    # 라벨을 숫자로 변환
    train_labels = [label_to_id[label] for label in train_labels]
    test_labels = [label_to_id[label] for label in test_labels]
    
    return train_texts, test_texts, train_labels, test_labels, label_to_id, id_to_label

def create_dataset_dict(texts, labels):
    """텍스트와 라벨을 데이터셋 딕셔너리로 변환합니다."""
    return {
        'text': texts,
        'label': labels
    }

def tokenize_function(examples, tokenizer, max_length=512):
    """텍스트를 토큰화하는 함수입니다."""
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors=None
    )

def compute_metrics(pred):
    """평가 메트릭을 계산합니다."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    print("=== 균형 잡힌 데이터로 BERT 모델 학습 시작 ===")
    
    # 1. 균형 잡힌 데이터 로드 및 전처리
    print("\n1. 균형 잡힌 데이터 로드 및 전처리 중...")
    train_texts, test_texts, train_labels, test_labels, label_to_id, id_to_label = load_and_prepare_balanced_data()
    
    # 2. BERT 모델 및 토크나이저 로드
    model_name = "bert-base-uncased"
    print(f"\n2. 모델 로딩 중: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(label_to_id)
    )
    
    # 3. 데이터셋 생성 및 변환
    print("\n3. 데이터셋 생성 및 변환 중...")
    
    # 훈련 데이터셋 생성
    train_dataset_dict = create_dataset_dict(train_texts, train_labels)
    train_dataset = Dataset.from_dict(train_dataset_dict)
    
    # 테스트 데이터셋 생성
    test_dataset_dict = create_dataset_dict(test_texts, test_labels)
    test_dataset = Dataset.from_dict(test_dataset_dict)
    
    print(f"훈련 데이터셋 크기: {len(train_dataset)}")
    print(f"테스트 데이터셋 크기: {len(test_dataset)}")
    
    # 4. 토큰화 함수 정의 및 적용
    print("\n4. 토큰화 중...")
    
    def tokenize_batch(examples):
        return tokenize_function(examples, tokenizer, max_length=512)
    
    # label 컬럼만 남기고 나머지 삭제
    remove_cols = [col for col in train_dataset.column_names if col != 'label']

    train_dataset = train_dataset.map(
        tokenize_batch, 
        batched=True, 
        batch_size=16,
        remove_columns=remove_cols
    )
    
    test_dataset = test_dataset.map(
        tokenize_batch, 
        batched=True, 
        batch_size=16,
        remove_columns=remove_cols
    )
    
    print("토큰화 완료!")
    print(f"토큰화된 훈련 데이터셋 특징: {train_dataset.features}")
    
    # 5. 데이터 콜레이터 설정
    print("\n5. 데이터 콜레이터 설정 중...")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 6. 학습 인수 설정
    print("\n6. 학습 인수 설정 중...")
    training_args = TrainingArguments(
        output_dir="./bert_balanced_model",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        logging_dir="./logs_balanced",
        logging_steps=10,
        save_total_limit=2,
        warmup_steps=100,
        logging_first_step=True,
    )
    
    # 7. 트레이너 초기화
    print("\n7. 트레이너 초기화 중...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # 8. 모델 학습
    print("\n8. 모델 학습 시작...")
    trainer.train()
    
    # 9. 모델 평가
    print("\n9. 모델 평가 중...")
    results = trainer.evaluate()
    print(f"평가 결과: {results}")
    
    # 10. 예측 수행 및 상세 분석
    print("\n10. 예측 수행 중...")
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    
    # 분류 리포트 생성
    print("\n=== 분류 리포트 ===")
    print(classification_report(test_labels, pred_labels, target_names=list(label_to_id.keys())))
    
    # 11. 결과 저장
    print("\n11. 결과 저장 중...")
    results_dict = {
        'model_name': model_name,
        'data_source': 'balanced_oversampled',
        'total_samples': len(train_texts) + len(test_texts),
        'train_samples': len(train_texts),
        'test_samples': len(test_texts),
        'num_labels': len(label_to_id),
        'label_mapping': label_to_id,
        'evaluation_results': results,
        'classification_report': classification_report(test_labels, pred_labels, target_names=list(label_to_id.keys()), output_dict=True)
    }
    
    # 결과 디렉토리 생성
    os.makedirs('./bert_balanced_model', exist_ok=True)
    
    with open('./bert_balanced_model/evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)
    
    # 라벨 매핑 저장
    with open('./bert_balanced_model/label_mapping.json', 'w', encoding='utf-8') as f:
        json.dump({
            'label_to_id': label_to_id,
            'id_to_label': id_to_label
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== 학습 완료 ===")
    print(f"모델이 './bert_balanced_model' 디렉토리에 저장되었습니다.")
    print(f"평가 결과가 'evaluation_results.json'에 저장되었습니다.")
    print(f"라벨 매핑이 'label_mapping.json'에 저장되었습니다.")

if __name__ == "__main__":
    main() 