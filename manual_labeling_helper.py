import pandas as pd
import os

def create_manual_labeling_file(input_file, output_file, sample_size=50):
    """
    수동 라벨링을 위한 파일을 생성합니다.
    각 라벨별로 샘플을 추출하여 사용자가 직접 확인할 수 있도록 합니다.
    """
    print("수동 라벨링을 위한 파일을 생성합니다...")
    
    # 데이터 로드
    df = pd.read_csv(input_file)
    
    # 각 라벨별로 샘플 추출
    samples = []
    for label in df['label'].unique():
        label_df = df[df['label'] == label].copy()
        if len(label_df) >= sample_size:
            sampled = label_df.sample(n=sample_size, random_state=42)
        else:
            sampled = label_df
        
        sampled['original_label'] = sampled['label']
        sampled['new_label'] = ''  # 수동 입력용
        sampled['notes'] = ''      # 메모용
        samples.append(sampled)
    
    # 모든 샘플 합치기
    manual_df = pd.concat(samples, ignore_index=True)
    
    # 순서 섞기
    manual_df = manual_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 수동 라벨링용 컬럼 추가
    manual_df['reviewed'] = False
    
    # 파일 저장
    manual_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"수동 라벨링 파일이 '{output_file}'에 생성되었습니다.")
    print(f"총 {len(manual_df)}개의 이메일이 포함되어 있습니다.")
    
    # 라벨별 샘플 수 출력
    print("\n라벨별 샘플 수:")
    for label in manual_df['original_label'].unique():
        count = len(manual_df[manual_df['original_label'] == label])
        print(f"  {label}: {count}개")
    
    return manual_df

def review_email_content(df, index):
    """
    특정 이메일의 내용을 보기 좋게 출력합니다.
    """
    row = df.iloc[index]
    print(f"\n{'='*80}")
    print(f"이메일 #{index + 1}")
    print(f"{'='*80}")
    print(f"파일: {row['file']}")
    print(f"제목: {row['subject']}")
    print(f"보낸사람: {row['from']}")
    print(f"받는사람: {row['to']}")
    print(f"원본 라벨: {row['original_label']}")
    print(f"현재 라벨: {row['new_label']}")
    print(f"메모: {row['notes']}")
    print(f"{'='*80}")
    print("내용:")
    print(f"{'='*80}")
    print(row['body'][:1000])  # 너무 길면 1000자까지만
    print(f"{'='*80}")

def interactive_labeling(manual_file):
    """
    대화형 라벨링 인터페이스를 제공합니다.
    """
    df = pd.read_csv(manual_file)
    
    print("대화형 라벨링을 시작합니다.")
    print("명령어:")
    print("  n: 다음 이메일")
    print("  p: 이전 이메일")
    print("  w: Work로 라벨 변경")
    print("  a: Advertisement로 라벨 변경")
    print("  s: Personal로 라벨 변경")
    print("  u: Unclear로 라벨 변경")
    print("  m: 메모 추가")
    print("  q: 종료")
    print("  s: 저장")
    
    current_index = 0
    
    while True:
        review_email_content(df, current_index)
        
        command = input("\n명령어를 입력하세요: ").lower().strip()
        
        if command == 'q':
            break
        elif command == 'n':
            current_index = min(current_index + 1, len(df) - 1)
        elif command == 'p':
            current_index = max(current_index - 1, 0)
        elif command == 'w':
            df.at[current_index, 'new_label'] = 'Work'
            df.at[current_index, 'reviewed'] = True
            print("라벨을 Work로 변경했습니다.")
        elif command == 'a':
            df.at[current_index, 'new_label'] = 'Advertisement'
            df.at[current_index, 'reviewed'] = True
            print("라벨을 Advertisement로 변경했습니다.")
        elif command == 's':
            df.at[current_index, 'new_label'] = 'Personal'
            df.at[current_index, 'reviewed'] = True
            print("라벨을 Personal로 변경했습니다.")
        elif command == 'u':
            df.at[current_index, 'new_label'] = 'Unclear'
            df.at[current_index, 'reviewed'] = True
            print("라벨을 Unclear로 변경했습니다.")
        elif command == 'm':
            note = input("메모를 입력하세요: ")
            df.at[current_index, 'notes'] = note
            print("메모가 추가되었습니다.")
        elif command == 'save':
            df.to_csv(manual_file, index=False, encoding='utf-8-sig')
            print("변경사항이 저장되었습니다.")
        else:
            print("잘못된 명령어입니다.")

def manual_labeling():
    # 데이터 불러오기
    try:
        df = pd.read_csv('dataset/emails_processed.csv')
        print(f"총 {len(df)}개의 이메일을 불러왔습니다.")
    except FileNotFoundError:
        print("dataset/emails_processed.csv 파일을 찾을 수 없습니다.")
        return
    
    # 100개 샘플 추출 (300개에서 100개로 변경)
    sample_df = df.sample(n=100, random_state=123).reset_index(drop=True)  # 다른 random_state 사용
    print(f"100개 샘플을 추출했습니다.")
    
    labels = []
    print("\n" + "="*60)
    print("수동 라벨링을 시작합니다.")
    print("각 이메일의 본문을 읽고 라벨을 입력하세요:")
    print("w: Work (업무 관련)")
    print("a: Advertisement (광고)")
    print("p: Personal (개인적)")
    print("q: 종료")
    print("="*60)
    
    for i, row in sample_df.iterrows():
        print(f"\n[{i+1}/100]")  # 300에서 100으로 변경
        print("-" * 60)
        
        # 본문 출력 (너무 길면 잘라서)
        body = str(row['body'])
        if len(body) > 1000:
            body = body[:1000] + "..."
        print(body)
        print("-" * 60)
        
        while True:
            key = input("라벨 (w/a/p/q): ").strip().lower()
            if key in ['w', 'a', 'p']:
                label_map = {'w': 'Work', 'a': 'Advertisement', 'p': 'Personal'}
                labels.append({
                    'body': row['body'],
                    'label': label_map[key]
                })
                print(f"라벨링 완료: {label_map[key]}")
                break
            elif key == 'q':
                print("라벨링을 중단합니다.")
                break
            else:
                print("w/a/p/q 중 하나를 입력하세요.")
        
        if key == 'q':
            break
    
    # 결과 저장 (파일명 변경)
    if labels:
        labeled_df = pd.DataFrame(labels)
        output_file = 'dataset/test_labeled_dataset.csv'  # 파일명 변경
        labeled_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"\n라벨링 완료!")
        print(f"총 {len(labels)}개의 이메일을 라벨링했습니다.")
        print(f"결과가 '{output_file}'에 저장되었습니다.")
        
        # 라벨 분포 출력
        label_counts = labeled_df['label'].value_counts()
        print("\n라벨 분포:")
        for label, count in label_counts.items():
            print(f"{label}: {count}개")
    else:
        print("라벨링된 데이터가 없습니다.")

if __name__ == "__main__":
    manual_labeling() 