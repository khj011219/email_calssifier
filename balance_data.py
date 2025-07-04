import pandas as pd
import numpy as np

def balance_data(input_file, output_file, samples_per_class=100):
    """
    각 라벨별로 지정된 개수만큼만 남기도록 데이터를 균형있게 만듭니다.
    
    Args:
        input_file: 입력 CSV 파일 경로
        output_file: 출력 CSV 파일 경로
        samples_per_class: 각 클래스당 샘플 개수
    """
    print(f"데이터 균형 조정을 시작합니다...")
    print(f"입력 파일: {input_file}")
    print(f"각 클래스당 샘플 개수: {samples_per_class}")
    
    # 데이터 로드
    df = pd.read_csv(input_file)
    print(f"원본 데이터 크기: {len(df)}개")
    
    # 원본 라벨 분포 확인
    print("\n원본 라벨 분포:")
    original_counts = df['label'].value_counts()
    print(original_counts)
    
    # 각 라벨별로 지정된 개수만큼 샘플링
    balanced_dfs = []
    
    for label in df['label'].unique():
        label_df = df[df['label'] == label].copy()
        
        if len(label_df) >= samples_per_class:
            # 충분한 데이터가 있으면 랜덤 샘플링
            sampled_df = label_df.sample(n=samples_per_class, random_state=42)
        else:
            # 부족한 경우 전체 사용 (경고 메시지 출력)
            sampled_df = label_df
            print(f"⚠️  {label} 클래스: {len(label_df)}개만 있어서 전체 사용")
        
        balanced_dfs.append(sampled_df)
        print(f"{label}: {len(sampled_df)}개 선택")
    
    # 균형잡힌 데이터프레임 생성
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # 순서 섞기 (클래스별로 그룹화되지 않도록)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 결과 확인
    print(f"\n균형 조정 후 데이터 크기: {len(balanced_df)}개")
    print("\n균형 조정 후 라벨 분포:")
    balanced_counts = balanced_df['label'].value_counts()
    print(balanced_counts)
    
    # 파일 저장
    balanced_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n균형잡힌 데이터가 '{output_file}'에 저장되었습니다.")
    
    return balanced_df

if __name__ == "__main__":
    input_file = 'dataset/emails_labeled.csv'
    output_file = 'dataset/emails_balanced.csv'
    
    # 각 라벨별로 100개씩만 남기기
    balanced_df = balance_data(input_file, output_file, samples_per_class=100)
    
    print("\n✅ 데이터 균형 조정 완료!")
    print(f"총 {len(balanced_df)}개의 균형잡힌 데이터가 생성되었습니다.") 