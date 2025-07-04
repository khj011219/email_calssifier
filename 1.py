import pandas as pd

# 라벨링된 데이터 로드
df = pd.read_csv('dataset/emails_labeled.csv')

# 원본 데이터 분포 확인
print("원본 데이터 분포:")
print(df['label'].value_counts())
print(f"\n총 데이터 수: {len(df)}개")

# 각 라벨별로 100개씩만 샘플링
balanced_df = pd.DataFrame()

for label in df['label'].unique():
    label_data = df[df['label'] == label]
    
    # 해당 라벨의 데이터가 100개보다 적으면 모두 사용
    if len(label_data) <= 50:
        sampled_data = label_data
    else:
        # 100개보다 많으면 랜덤 샘플링
        sampled_data = label_data.sample(n=50, random_state=42)
    
    balanced_df = pd.concat([balanced_df, sampled_data], ignore_index=True)

# 결과 확인
print("\n균형잡힌 데이터 분포:")
print(balanced_df['label'].value_counts())
print(f"\n총 데이터 수: {len(balanced_df)}개")

# 균형잡힌 데이터셋 저장
balanced_df.to_csv('dataset/balanced.csv', index=False, encoding='utf-8-sig')
print(f"\n균형잡힌 데이터셋이 'dataset/balanced.csv'에 저장되었습니다.")

# 저장된 파일의 상위 5개 행 확인
print("\n저장된 파일의 상위 5개 행:")
print(balanced_df.head())
