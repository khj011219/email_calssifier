import pandas as pd
import numpy as np

def extract_random_100_for_labeling():
    """기존 300개를 제외하고 랜덤으로 100개를 추출합니다."""
    print("=== 새로운 라벨링용 100개 데이터 추출 시작 ===")
    
    # 1. 기존 수작업 라벨링 데이터 로드
    print("\n1. 기존 수작업 라벨링 데이터 로드 중...")
    manual_df = pd.read_csv('dataset/manual_labeled_300.csv')
    print(f"기존 수작업 데이터: {len(manual_df)}개")
    
    # 2. 원본 이메일 데이터 로드
    print("\n2. 원본 이메일 데이터 로드 중...")
    original_df = pd.read_csv('dataset/emails_processed.csv')
    print(f"원본 데이터: {len(original_df)}개")
    
    # 3. 기존 300개와 겹치지 않는 데이터 찾기
    print("\n3. 중복 제거 중...")
    
    # 기존 300개의 본문을 set으로 변환 (빠른 검색을 위해)
    manual_bodies = set(manual_df['body'].astype(str))
    
    # 중복되지 않는 인덱스 찾기
    non_duplicate_indices = []
    for idx, body in enumerate(original_df['body'].astype(str)):
        if body not in manual_bodies:
            non_duplicate_indices.append(idx)
    
    print(f"중복되지 않는 데이터: {len(non_duplicate_indices)}개")
    
    # 4. 랜덤으로 100개 선택
    print("\n4. 랜덤으로 100개 선택 중...")
    np.random.seed(42)
    
    if len(non_duplicate_indices) >= 100:
        selected_indices = np.random.choice(non_duplicate_indices, size=100, replace=False)
    else:
        print(f"경고: 중복되지 않는 데이터가 100개보다 적습니다. ({len(non_duplicate_indices)}개)")
        selected_indices = non_duplicate_indices
    
    # 선택된 데이터 추출
    selected_df = original_df.iloc[selected_indices].copy()
    selected_df = selected_df.reset_index(drop=True)
    
    print(f"선택된 데이터: {len(selected_df)}개")
    
    # 5. 라벨링용 파일 생성
    print("\n5. 라벨링용 파일 생성 중...")
    output_path = 'dataset/new_100_for_labeling.csv'
    
    # 라벨링을 위한 컬럼만 포함
    labeling_df = selected_df[['body']].copy()
    labeling_df['label'] = ''  # 빈 라벨 컬럼 추가
    
    labeling_df.to_csv(output_path, index=False)
    
    print(f"\n=== 새로운 라벨링용 데이터 추출 완료 ===")
    print(f"저장 경로: {output_path}")
    print(f"총 {len(labeling_df)}개의 새로운 데이터가 준비되었습니다.")
    
    # 6. 데이터 샘플 출력
    print(f"\n=== 데이터 샘플 (처음 5개) ===")
    for i in range(min(5, len(labeling_df))):
        print(f"\n--- 샘플 {i+1} ---")
        text = labeling_df.iloc[i]['body']
        print(f"텍스트 길이: {len(text)}자")
        print(f"텍스트 미리보기: {text[:200]}...")
    
    return labeling_df

if __name__ == "__main__":
    # 새로운 라벨링용 데이터 추출
    new_data = extract_random_100_for_labeling() 