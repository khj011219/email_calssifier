import pandas as pd
import numpy as np

# 수정된 라벨링 결과 확인
df = pd.read_csv('dataset/emails_labeled.csv')

print("=== 수정된 라벨링 결과 분석 ===")
print(f"총 데이터 수: {len(df)}개")

print("\n=== 라벨 분포 ===")
label_counts = df['label'].value_counts()
print(label_counts)
print(f"\n라벨별 비율:")
for label, count in label_counts.items():
    ratio = count / len(df) * 100
    print(f"{label}: {count}개 ({ratio:.1f}%)")

print("\n=== 라벨별 샘플 확인 ===")
for label in df['label'].unique():
    print(f"\n--- {label} 라벨 샘플 (3개) ---")
    sample = df[df['label'] == label].head(3)
    for idx, row in sample.iterrows():
        print(f"제목: {row['subject'][:80]}...")
        print(f"발신자: {row['from']}")
        print(f"수신자: {row['to'][:80]}...")
        print(f"본문 일부: {row['body'][:150]}...")
        print("-" * 60)

# 이전에 문제가 있었던 샘플들 확인
print("\n=== 이전 문제 샘플 재확인 ===")

# 농담/유머 이메일 확인
joke_samples = df[df['subject'].str.contains('joke|funny|humor|pun', case=False, na=False)]
if not joke_samples.empty:
    print(f"\n농담/유머 관련 이메일 ({len(joke_samples)}개):")
    for idx, row in joke_samples.head(2).iterrows():
        print(f"제목: {row['subject']}")
        print(f"라벨: {row['label']}")
        print(f"발신자: {row['from']}")
        print("-" * 40)

# Enron 뉴스 관련 이메일 확인
news_samples = df[df['subject'].str.contains('enron|news|report', case=False, na=False)]
if not news_samples.empty:
    print(f"\nEnron 뉴스 관련 이메일 ({len(news_samples)}개):")
    for idx, row in news_samples.head(2).iterrows():
        print(f"제목: {row['subject']}")
        print(f"라벨: {row['label']}")
        print(f"발신자: {row['from']}")
        print("-" * 40)

# 광고 관련 이메일 확인
ad_samples = df[df['label'] == 'Advertisement']
if not ad_samples.empty:
    print(f"\n광고 라벨 이메일 샘플 ({len(ad_samples)}개):")
    for idx, row in ad_samples.head(2).iterrows():
        print(f"제목: {row['subject']}")
        print(f"발신자: {row['from']}")
        print(f"본문 일부: {row['body'][:100]}...")
        print("-" * 40)

print("\n=== 데이터 품질 확인 ===")
print(f"빈 제목: {df['subject'].isna().sum()}개")
print(f"빈 본문: {df['body'].isna().sum()}개")
print(f"빈 발신자: {df['from'].isna().sum()}개")
print(f"빈 수신자: {df['to'].isna().sum()}개")
print(f"라벨이 없는 데이터: {df['label'].isna().sum()}개") 