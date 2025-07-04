import pandas as pd
import numpy as np
from sklearn.utils import resample
import random

def create_balanced_dataset():
    """오버샘플링을 통해 균형 잡힌 데이터셋을 생성합니다."""
    print("=== 균형 잡힌 데이터셋 생성 시작 ===")
    
    # 1. 원본 데이터 로드
    print("\n1. 원본 데이터 로드 중...")
    df = pd.read_csv('dataset/manual_labeled_300.csv')
    
    print(f"원본 데이터 크기: {len(df)}")
    print("원본 라벨 분포:")
    print(df['label'].value_counts())
    
    # 2. 클래스별 데이터 분리
    print("\n2. 클래스별 데이터 분리 중...")
    work_df = df[df['label'] == 'Work']
    personal_df = df[df['label'] == 'Personal']
    advertisement_df = df[df['label'] == 'Advertisement']
    
    print(f"Work: {len(work_df)}개")
    print(f"Personal: {len(personal_df)}개")
    print(f"Advertisement: {len(advertisement_df)}개")
    
    # 3. 오버샘플링 수행
    print("\n3. 오버샘플링 수행 중...")
    
    # 가장 많은 클래스(Work)의 크기에 맞춰 오버샘플링
    target_size = len(work_df)
    
    # Personal 오버샘플링
    personal_upsampled = resample(
        personal_df,
        replace=True,
        n_samples=target_size,
        random_state=42
    )
    
    # Advertisement 오버샘플링
    advertisement_upsampled = resample(
        advertisement_df,
        replace=True,
        n_samples=target_size,
        random_state=42
    )
    
    print(f"오버샘플링 후 Personal: {len(personal_upsampled)}개")
    print(f"오버샘플링 후 Advertisement: {len(advertisement_upsampled)}개")
    
    # 4. 균형 잡힌 데이터셋 생성
    print("\n4. 균형 잡힌 데이터셋 생성 중...")
    balanced_df = pd.concat([work_df, personal_upsampled, advertisement_upsampled])
    
    # 데이터 순서 섞기
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("균형 잡힌 데이터셋 라벨 분포:")
    print(balanced_df['label'].value_counts())
    
    # 5. 결과 저장
    print("\n5. 결과 저장 중...")
    output_path = 'dataset/manual_labeled_balanced.csv'
    balanced_df.to_csv(output_path, index=False)
    
    print(f"\n=== 균형 잡힌 데이터셋 생성 완료 ===")
    print(f"저장 경로: {output_path}")
    print(f"총 데이터 수: {len(balanced_df)}")
    print(f"클래스별 데이터 수: {len(balanced_df) // 3}개씩")
    
    return balanced_df

def analyze_balanced_dataset(df):
    """균형 잡힌 데이터셋을 분석합니다."""
    print("\n=== 균형 잡힌 데이터셋 분석 ===")
    
    # 클래스별 분포
    print("클래스별 분포:")
    for label in df['label'].unique():
        count = len(df[df['label'] == label])
        percentage = count / len(df) * 100
        print(f"  {label}: {count}개 ({percentage:.1f}%)")
    
    # 텍스트 길이 통계
    df['text_length'] = df['body'].str.len()
    print(f"\n텍스트 길이 통계:")
    print(f"  평균: {df['text_length'].mean():.1f}자")
    print(f"  최소: {df['text_length'].min()}자")
    print(f"  최대: {df['text_length'].max()}자")
    
    # 클래스별 텍스트 길이
    print(f"\n클래스별 평균 텍스트 길이:")
    for label in df['label'].unique():
        avg_length = df[df['label'] == label]['text_length'].mean()
        print(f"  {label}: {avg_length:.1f}자")

if __name__ == "__main__":
    # 균형 잡힌 데이터셋 생성
    balanced_df = create_balanced_dataset()
    
    # 데이터셋 분석
    analyze_balanced_dataset(balanced_df) 