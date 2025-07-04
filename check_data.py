import pandas as pd

# 데이터 로드
df = pd.read_csv('dataset/manual_labeled_300.csv')

print(f"총 행 수: {len(df)}")
print(f"컬럼: {df.columns.tolist()}")
print(f"라벨 분포:")
print(df['label'].value_counts())
print(f"\n라벨 종류: {df['label'].unique()}")
print(f"\n첫 번째 행 예시:")
print(f"Body: {df['body'].iloc[0][:200]}...")
print(f"Label: {df['label'].iloc[0]}") 