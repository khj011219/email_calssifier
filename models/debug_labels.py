from datasets import load_from_disk
import numpy as np
import json

def debug_labels():
    """데이터셋의 라벨을 디버깅"""
    print("데이터셋 라벨 디버깅을 시작합니다...")
    
    # 데이터셋 로드
    datasets = load_from_disk('data/tokenized')
    
    # 라벨 맵 로드
    with open('data/tokenized/label_map.json', 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    
    print(f"라벨 맵: {label_map}")
    
    # 학습 데이터 확인
    train_dataset = datasets["train"]
    print(f"\n학습 데이터 크기: {len(train_dataset)}")
    
    # 라벨 타입 확인
    labels = train_dataset['label']
    print(f"라벨 타입: {type(labels[0])}")
    print(f"라벨 예시: {labels[:10]}")
    
    # 고유 라벨 확인
    unique_labels = np.unique(labels)
    print(f"고유 라벨: {unique_labels}")
    print(f"고유 라벨 타입: {[type(x) for x in unique_labels]}")
    
    # 라벨 분포 확인
    for label in unique_labels:
        count = sum(1 for x in labels if x == label)
        print(f"라벨 {label} (타입: {type(label)}): {count}개")
    
    # 검증 데이터도 확인
    val_dataset = datasets["validation"]
    val_labels = val_dataset['label']
    val_unique = np.unique(val_labels)
    print(f"\n검증 데이터 고유 라벨: {val_unique}")

if __name__ == "__main__":
    debug_labels() 