import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_new_data_for_labeling():
    """기존 300개 수작업 데이터를 제외하고 새로운 100개 데이터를 추출합니다."""
    print("=== 새로운 라벨링용 데이터 추출 시작 ===")
    
    # 1. 기존 수작업 라벨링 데이터 로드
    print("\n1. 기존 수작업 라벨링 데이터 로드 중...")
    manual_df = pd.read_csv('dataset/manual_labeled_300.csv')
    print(f"기존 수작업 데이터: {len(manual_df)}개")
    
    # 2. 원본 이메일 데이터 로드
    print("\n2. 원본 이메일 데이터 로드 중...")
    original_df = pd.read_csv('dataset/emails_processed.csv')
    print(f"원본 데이터: {len(original_df)}개")
    
    # 3. 중복 제거 (기존 수작업 데이터와 겹치는 것 제외)
    print("\n3. 중복 제거 중...")
    
    # 텍스트 유사도 기반으로 중복 제거
    def remove_similar_texts(original_texts, manual_texts, threshold=0.8):
        """유사한 텍스트를 제거합니다."""
        print("텍스트 유사도 계산 중...")
        
        # TF-IDF 벡터화
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # 원본 텍스트 벡터화
        original_vectors = vectorizer.fit_transform(original_texts)
        
        # 수작업 텍스트 벡터화 (같은 벡터라이저 사용)
        manual_vectors = vectorizer.transform(manual_texts)
        
        # 코사인 유사도 계산
        similarities = cosine_similarity(original_vectors, manual_vectors)
        
        # 유사도가 threshold 이상인 인덱스 찾기
        similar_indices = np.where(similarities >= threshold)[0]
        unique_similar_indices = np.unique(similar_indices)
        
        print(f"유사한 텍스트 발견: {len(unique_similar_indices)}개")
        
        # 유사하지 않은 인덱스만 반환
        all_indices = np.arange(len(original_texts))
        dissimilar_indices = np.setdiff1d(all_indices, unique_similar_indices)
        
        return dissimilar_indices
    
    # 중복 제거 수행
    original_texts = original_df['body'].astype(str).tolist()
    manual_texts = manual_df['body'].astype(str).tolist()
    
    dissimilar_indices = remove_similar_texts(original_texts, manual_texts, threshold=0.8)
    
    # 중복되지 않는 데이터 추출
    new_df = original_df.iloc[dissimilar_indices].copy()
    print(f"중복 제거 후 데이터: {len(new_df)}개")
    
    # 4. 새로운 100개 데이터 추출
    print("\n4. 새로운 100개 데이터 추출 중...")
    
    # 랜덤 샘플링 (다양성 확보)
    np.random.seed(42)
    sample_indices = np.random.choice(len(new_df), size=min(100, len(new_df)), replace=False)
    sample_df = new_df.iloc[sample_indices].copy()
    
    # 인덱스 리셋
    sample_df = sample_df.reset_index(drop=True)
    
    print(f"새로운 라벨링용 데이터: {len(sample_df)}개")
    
    # 5. 결과 저장
    print("\n5. 결과 저장 중...")
    output_path = 'dataset/new_data_for_labeling.csv'
    
    # 라벨링을 위한 컬럼만 포함
    labeling_df = sample_df[['body']].copy()
    labeling_df['label'] = ''  # 빈 라벨 컬럼 추가
    
    labeling_df.to_csv(output_path, index=False)
    
    print(f"\n=== 새로운 라벨링용 데이터 추출 완료 ===")
    print(f"저장 경로: {output_path}")
    print(f"총 {len(labeling_df)}개의 새로운 데이터가 준비되었습니다.")
    
    # 6. 데이터 샘플 출력
    print(f"\n=== 데이터 샘플 (처음 3개) ===")
    for i in range(min(3, len(labeling_df))):
        print(f"\n--- 샘플 {i+1} ---")
        text = labeling_df.iloc[i]['body']
        print(f"텍스트 길이: {len(text)}자")
        print(f"텍스트 미리보기: {text[:200]}...")
    
    return labeling_df

def analyze_new_data(df):
    """새로운 데이터를 분석합니다."""
    print("\n=== 새로운 데이터 분석 ===")
    
    # 텍스트 길이 통계
    df['text_length'] = df['body'].str.len()
    print(f"텍스트 길이 통계:")
    print(f"  평균: {df['text_length'].mean():.1f}자")
    print(f"  최소: {df['text_length'].min()}자")
    print(f"  최대: {df['text_length'].max()}자")
    
    # 길이별 분포
    print(f"\n텍스트 길이 분포:")
    length_ranges = [(0, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, float('inf'))]
    for low, high in length_ranges:
        if high == float('inf'):
            count = len(df[df['text_length'] >= low])
            print(f"  {low}자 이상: {count}개")
        else:
            count = len(df[(df['text_length'] >= low) & (df['text_length'] < high)])
            print(f"  {low}-{high}자: {count}개")

if __name__ == "__main__":
    # 새로운 라벨링용 데이터 추출
    new_data = extract_new_data_for_labeling()
    
    # 데이터 분석
    analyze_new_data(new_data) 