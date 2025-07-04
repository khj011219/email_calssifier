import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import os

def preprocess_for_kobert(data_dir, output_dir):
    """
    Loads split data, tokenizes it for KoBERT, and saves it in Hugging Face
    Dataset format.
    """
    try:
        print("KoBERT 데이터 전처리를 시작합니다...")
        
        # 1. 데이터 로드
        print("분할된 CSV 파일을 로드합니다...")
        train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
        val_df = pd.read_csv(os.path.join(data_dir, 'validation.csv'))

        # 라벨을 정수로 변환하기 위한 맵핑 생성
        labels = train_df['label'].unique()
        label2id = {label: i for i, label in enumerate(labels)}
        id2label = {i: label for i, label in enumerate(labels)}
        
        print(f"라벨 맵핑: {label2id}")

        # 라벨을 정수로 변환
        train_df['label'] = train_df['label'].map(label2id)
        val_df['label'] = val_df['label'].map(label2id)

        # 2. Huggingface Dataset 형식으로 변환
        print("Pandas DataFrame을 Huggingface Dataset으로 변환합니다...")
        train_dataset = Dataset.from_pandas(train_df)
        eval_dataset = Dataset.from_pandas(val_df)
        
        raw_datasets = DatasetDict({
            'train': train_dataset,
            'validation': eval_dataset
        })

        # 3. 토크나이저 로드 (transformers의 AutoTokenizer 사용)
        print("KoBERT 토크나이저를 transformers에서 로드합니다 (`monologg/kobert`)...")
        tokenizer = AutoTokenizer.from_pretrained('monologg/kobert', trust_remote_code=True)
        
        # 4. 토큰화 함수 정의
        def tokenize_function(examples):
            # 'text' 필드가 비어있거나 NaN인 경우를 처리
            texts = [str(text) if pd.notna(text) else "" for text in examples["text"]]
            return tokenizer(texts, padding="max_length", truncation=True, max_length=512)

        # 5. 데이터셋 토큰화
        print("데이터셋을 토큰화합니다 (이 과정은 시간이 걸릴 수 있습니다)...")
        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

        # 불필요한 열 제거 및 포맷 설정
        remove_cols = ["text"]
        if "__index_level_0__" in tokenized_datasets["train"].column_names:
            remove_cols.append("__index_level_0__")
        tokenized_datasets = tokenized_datasets.remove_columns(remove_cols)
        tokenized_datasets.set_format("torch")

        print("전처리된 데이터를 디스크에 저장합니다...")
        # 6. 결과 저장
        os.makedirs(output_dir, exist_ok=True)
        tokenized_datasets.save_to_disk(output_dir)
        # 토크나이저 저장 (예외 발생 시 안내 메시지 출력)
        try:
            tokenizer.save_pretrained(output_dir)
            print(f"토크나이저가 {output_dir}에 저장되었습니다.")
        except Exception as e:
            print(f"토크나이저 저장 중 오류 발생: {e}")
            print("vocab.txt, tokenizer_config.json 등 파일이 정상적으로 다운로드되었는지 확인하세요.")
        # 라벨 맵핑 정보도 저장
        import json
        with open(os.path.join(output_dir, 'label_map.json'), 'w') as f:
            json.dump({'label2id': label2id, 'id2label': id2label}, f)
        # 실제 저장된 파일 목록 출력
        print(f"\n전처리 완료! 모든 데이터가 '{output_dir}'에 저장되었습니다.")
        print("\n저장된 구성 요소:")
        for fname in os.listdir(output_dir):
            print(f"- {fname}")
        print("\n이제 이 데이터를 Huggingface Trainer에 전달하여 모델을 학습할 수 있습니다.")

    except (FileNotFoundError, ImportError) as e:
        if isinstance(e, FileNotFoundError):
            print(f"오류: '{data_dir}'에서 필요한 파일을 찾을 수 없습니다.")
        else:
            print("오류: 필수 라이브러리가 설치되지 않았을 수 있습니다.")
            print("pip install transformers datasets torch")
        print(f"상세 정보: {e}")
    except Exception as e:
        print(f"알 수 없는 오류가 발생했습니다: {e}")

if __name__ == '__main__':
    data_dir = 'data/processed'
    output_dir = 'data/tokenized'
    preprocess_for_kobert(data_dir, output_dir) 