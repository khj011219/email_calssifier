import pandas as pd
import re
from collections import Counter

def analyze_email_content(subject, body):
    """
    이메일의 제목과 내용을 분석하여 라벨을 추정합니다.
    """
    # 텍스트를 소문자로 변환
    text = (subject + " " + body).lower()
    
    # 키워드 기반 분류
    work_keywords = [
        'meeting', 'project', 'report', 'deadline', 'client', 'business', 
        'work', 'office', 'schedule', 'conference', 'presentation', 'review',
        'approval', 'contract', 'agreement', 'proposal', 'budget', 'quarterly',
        'annual', 'financial', 'revenue', 'sales', 'marketing', 'hr', 'hr@',
        'enron', 'energy', 'trading', 'market', 'investment', 'stock',
        'earnings', 'quarter', 'fiscal', 'board', 'executive', 'ceo', 'cfo'
    ]
    
    personal_keywords = [
        'family', 'kids', 'children', 'wife', 'husband', 'son', 'daughter',
        'birthday', 'party', 'dinner', 'lunch', 'weekend', 'vacation', 'holiday',
        'christmas', 'thanksgiving', 'wedding', 'anniversary', 'baby', 'pregnant',
        'school', 'college', 'university', 'graduation', 'home', 'house',
        'personal', 'private', 'friend', 'friends', 'dinner', 'movie', 'game'
    ]
    
    advertisement_keywords = [
        'sale', 'discount', 'offer', 'limited time', 'special', 'deal',
        'promotion', 'coupon', 'free', 'buy now', 'order', 'subscribe',
        'newsletter', 'marketing', 'advertisement', 'ad', 'spam', 'unsubscribe',
        'click here', 'visit', 'website', 'www.', 'http', 'commercial',
        'product', 'service', 'price', 'cost', 'money', 'save', 'earn'
    ]
    
    # 각 카테고리별 점수 계산
    work_score = sum(1 for keyword in work_keywords if keyword in text)
    personal_score = sum(1 for keyword in personal_keywords if keyword in text)
    ad_score = sum(1 for keyword in advertisement_keywords if keyword in text)
    
    # 점수 정규화 (텍스트 길이로 나누기)
    text_length = len(text.split())
    if text_length > 0:
        work_score = work_score / text_length * 1000
        personal_score = personal_score / text_length * 1000
        ad_score = ad_score / text_length * 1000
    
    # 가장 높은 점수의 카테고리 선택
    scores = {
        'Work': work_score,
        'Personal': personal_score,
        'Advertisement': ad_score
    }
    
    max_score = max(scores.values())
    predicted_label = max(scores, key=scores.get)
    
    # 점수가 너무 낮으면 애매한 것으로 판단
    if max_score < 0.1:
        return 'Unclear', scores
    
    return predicted_label, scores

def improve_labeling(input_file, output_file):
    """
    이메일 데이터를 다시 분석하고 더 정확한 라벨링을 수행합니다.
    """
    print("이메일 라벨링 개선을 시작합니다...")
    
    # 데이터 로드
    df = pd.read_csv(input_file)
    print(f"총 {len(df)}개의 이메일을 분석합니다.")
    
    # 새로운 라벨과 점수 저장
    new_labels = []
    confidence_scores = []
    unclear_count = 0
    
    for idx, row in df.iterrows():
        subject = str(row['subject']) if pd.notna(row['subject']) else ""
        body = str(row['body']) if pd.notna(row['body']) else ""
        
        predicted_label, scores = analyze_email_content(subject, body)
        
        if predicted_label == 'Unclear':
            unclear_count += 1
        
        new_labels.append(predicted_label)
        confidence_scores.append(max(scores.values()))
        
        if idx % 50 == 0:
            print(f"진행률: {idx}/{len(df)} ({idx/len(df)*100:.1f}%)")
    
    # 결과 분석
    print(f"\n=== 라벨링 결과 ===")
    print(f"총 이메일 수: {len(df)}")
    print(f"애매한 이메일 수: {unclear_count}")
    print(f"명확한 이메일 수: {len(df) - unclear_count}")
    
    # 새로운 라벨 분포
    label_counts = Counter(new_labels)
    print(f"\n새로운 라벨 분포:")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count}개 ({count/len(df)*100:.1f}%)")
    
    # 원본 라벨과 새로운 라벨 비교
    df['new_label'] = new_labels
    df['confidence_score'] = confidence_scores
    
    print(f"\n=== 라벨 변경 분석 ===")
    changed_count = 0
    for idx, row in df.iterrows():
        if row['label'] != row['new_label']:
            changed_count += 1
            if changed_count <= 10:  # 처음 10개만 출력
                print(f"  {row['label']} → {row['new_label']} (점수: {row['confidence_score']:.3f})")
    
    print(f"총 {changed_count}개의 라벨이 변경되었습니다.")
    
    # 애매한 이메일 제거하고 명확한 것만 저장
    clear_df = df[df['new_label'] != 'Unclear'].copy()
    clear_df = clear_df.drop(['new_label', 'confidence_score'], axis=1)
    clear_df['label'] = df[df['new_label'] != 'Unclear']['new_label']
    
    print(f"\n명확한 라벨의 이메일만 저장: {len(clear_df)}개")
    
    # 각 라벨별로 100개씩 균형 조정
    balanced_dfs = []
    for label in clear_df['label'].unique():
        label_df = clear_df[clear_df['label'] == label].copy()
        if len(label_df) >= 100:
            sampled_df = label_df.sample(n=100, random_state=42)
        else:
            sampled_df = label_df
            print(f"⚠️  {label} 클래스: {len(label_df)}개만 있어서 전체 사용")
        
        balanced_dfs.append(sampled_df)
        print(f"{label}: {len(sampled_df)}개 선택")
    
    final_df = pd.concat(balanced_dfs, ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 최종 결과 저장
    final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n=== 최종 결과 ===")
    print(f"최종 데이터 크기: {len(final_df)}개")
    print(f"최종 라벨 분포:")
    final_counts = final_df['label'].value_counts()
    print(final_counts)
    
    print(f"\n개선된 데이터가 '{output_file}'에 저장되었습니다.")
    
    return final_df

if __name__ == "__main__":
    input_file = 'dataset/emails_balanced.csv'
    output_file = 'dataset/emails_improved.csv'
    
    improved_df = improve_labeling(input_file, output_file)
    
    print("\n✅ 라벨링 개선 완료!") 