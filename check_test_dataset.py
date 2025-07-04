import pandas as pd
import numpy as np
from collections import Counter
from sklearn.utils import resample

def check_test_dataset():
    """test_labeled_dataset.csv 파일의 구조와 라벨 분포를 확인합니다."""
    
    # 데이터 로드
    df = pd.read_csv('dataset/test_labeled_dataset.csv')
    
    print("="*50)
    print("test_labeled_dataset.csv 분석 결과")
    print("="*50)
    
    print(f"\n📊 기본 정보:")
    print(f"   전체 데이터 수: {len(df):,}개")
    print(f"   컬럼명: {df.columns.tolist()}")
    
    print(f"\n📋 라벨 분포:")
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {label}: {count:,}개 ({percentage:.1f}%)")
    
    print(f"\n📝 데이터 샘플:")
    print("첫 번째 행:")
    print(f"   라벨: {df.iloc[0]['label']}")
    print(f"   본문 길이: {len(df.iloc[0]['body'])} 문자")
    print(f"   본문 미리보기: {df.iloc[0]['body'][:100]}...")
    
    print(f"\n마지막 행:")
    print(f"   라벨: {df.iloc[-1]['label']}")
    print(f"   본문 길이: {len(df.iloc[-1]['body'])} 문자")
    print(f"   본문 미리보기: {df.iloc[-1]['body'][:100]}...")
    
    # 본문 길이 통계
    body_lengths = df['body'].str.len()
    print(f"\n📏 본문 길이 통계:")
    print(f"   평균 길이: {body_lengths.mean():.1f} 문자")
    print(f"   중간값: {body_lengths.median():.1f} 문자")
    print(f"   최소 길이: {body_lengths.min()} 문자")
    print(f"   최대 길이: {body_lengths.max()} 문자")
    
    # 결측값 확인
    print(f"\n🔍 결측값 확인:")
    print(f"   body 컬럼 결측값: {df['body'].isnull().sum()}개")
    print(f"   label 컬럼 결측값: {df['label'].isnull().sum()}개")
    
    # 고유 라벨 확인
    unique_labels = df['label'].unique()
    print(f"\n🏷️ 고유 라벨: {unique_labels.tolist()}")
    
    return df

def oversample_test_dataset():
    """test_labeled_dataset.csv를 오버샘플링하여 각 라벨의 샘플 수를 맞춥니다."""
    df = pd.read_csv('dataset/test_labeled_dataset.csv')
    max_count = df['label'].value_counts().max()
    dfs = []
    for label in df['label'].unique():
        df_label = df[df['label'] == label]
        if len(df_label) < max_count:
            df_label_upsampled = resample(
                df_label,
                replace=True,
                n_samples=max_count,
                random_state=42
            )
            dfs.append(df_label_upsampled)
        else:
            dfs.append(df_label)
    df_oversampled = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    df_oversampled.to_csv('dataset/test_labeled_dataset_oversampled.csv', index=False)
    print(f"오버샘플링 완료! 각 라벨 {max_count}개, 총 {len(df_oversampled)}개 샘플")
    print(df_oversampled['label'].value_counts())
    return df_oversampled

if __name__ == "__main__":
    check_test_dataset()
    oversample_test_dataset() 