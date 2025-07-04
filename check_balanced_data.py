import pandas as pd
import numpy as np

# 균형잡힌 데이터 로드
df = pd.read_csv('dataset/emails_balanced.csv')

print("=== 균형잡힌 데이터셋 분석 ===")
print(f"총 데이터 수: {len(df)}개")
print(f"컬럼: {list(df.columns)}")

print("\n=== 라벨 분포 ===")
label_counts = df['label'].value_counts()
print(label_counts)
print(f"\n라벨별 비율:")
for label, count in label_counts.items():
    ratio = count / len(df) * 100
    print(f"{label}: {count}개 ({ratio:.1f}%)")

print("\n=== 라벨별 샘플 확인 ===")
for label in df['label'].unique():
    print(f"\n--- {label} 라벨 샘플 ---")
    sample = df[df['label'] == label].head(2)
    for idx, row in sample.iterrows():
        print(f"제목: {row['subject'][:100]}...")
        print(f"발신자: {row['from']}")
        print(f"수신자: {row['to'][:100]}...")
        print(f"본문 일부: {row['body'][:200]}...")
        print("-" * 50)

print("\n=== 데이터 품질 확인 ===")
print(f"빈 제목: {df['subject'].isna().sum()}개")
print(f"빈 본문: {df['body'].isna().sum()}개")
print(f"빈 발신자: {df['from'].isna().sum()}개")
print(f"빈 수신자: {df['to'].isna().sum()}개")

# 라벨별 평균 텍스트 길이
print("\n=== 라벨별 평균 텍스트 길이 ===")
for label in df['label'].unique():
    label_data = df[df['label'] == label]
    avg_subject_len = label_data['subject'].str.len().mean()
    avg_body_len = label_data['body'].str.len().mean()
    print(f"{label}:")
    print(f"  평균 제목 길이: {avg_subject_len:.1f}자")
    print(f"  평균 본문 길이: {avg_body_len:.1f}자") 