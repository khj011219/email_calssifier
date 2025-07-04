import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from datasets import load_from_disk
import os
import json
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# 데이터 경로 및 설정
DATA_DIR = 'data/tokenized'
MODEL_NAME = 'monologg/kobert'
OUTPUT_DIR = 'models/kobert_trained_improved'
NUM_LABELS = 3  # 라벨 개수 (Work, Advertisement, Personal)
EPOCHS = 5  # 더 많은 에포크
BATCH_SIZE = 8  # GPU 메모리에 맞게 조정

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 디바이스: {device}")

# 라벨 맵 로드
def load_label_map(data_dir):
    try:
        with open(os.path.join(data_dir, 'label_map.json'), 'r', encoding='utf-8') as f:
            label_map = json.load(f)
        id2label = {int(k): v for k, v in label_map['id2label'].items()}
        return label_map['label2id'], id2label
    except Exception as e:
        print(f"라벨 맵 로드 실패: {e}")
        raise

# 데이터셋 로드
try:
    print("데이터셋을 로드합니다...")
    datasets = load_from_disk(DATA_DIR)
    label2id, id2label = load_label_map(DATA_DIR)
    print("데이터셋 로드 완료")
except Exception as e:
    print(f"데이터셋 로드 실패: {e}")
    raise

# 데이터 일부만 사용 (학습 10000개, 검증 2000개)
train_size = min(10000, len(datasets["train"]))
eval_size = min(2000, len(datasets["validation"]))
train_dataset = datasets["train"].select(range(train_size))
eval_dataset = datasets["validation"].select(range(eval_size))

print(f"학습 데이터: {len(train_dataset)}개")
print(f"검증 데이터: {len(eval_dataset)}개")

# 클래스 가중치 계산
try:
    labels = [int(l) for l in train_dataset['label']]
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=labels)
    class_weights = torch.FloatTensor(class_weights).to(device)  # 디바이스로 이동
    print(f"클래스 가중치: {class_weights}")
except Exception as e:
    print(f"클래스 가중치 계산 실패: {e}")
    raise

# 커스텀 Trainer
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # class_weights는 이미 올바른 디바이스에 있음
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# 모델 및 토크나이저 로드
try:
    print("모델을 로드합니다...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
        trust_remote_code=True
    )
    model.to(device)  # 모델을 디바이스로 이동
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print("모델 로드 완료")
except Exception as e:
    print(f"모델 로드 실패: {e}")
    raise

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {"accuracy": acc, "f1": f1}

# Trainer 설정 (최소한의 안전한 옵션만 사용)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    logging_dir=os.path.join(OUTPUT_DIR, 'logs'),
    logging_steps=50,
    save_strategy="no",  # 토크나이저 저장 문제 방지
    evaluation_strategy="steps",  # 평가는 실행하되
    eval_steps=500,  # 500 스텝마다 평가
    report_to=None,  # wandb 비활성화
    dataloader_num_workers=0,  # Windows에서 멀티프로세싱 문제 방지
    remove_unused_columns=False,
    push_to_hub=False,
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=None  # 평가 중 토크나이저 저장 문제 방지
)

if __name__ == "__main__":
    try:
        print("개선된 KoBERT 모델 학습을 시작합니다...")
        print("GPU 사용 여부:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("사용 중인 GPU:", torch.cuda.get_device_name(0))
            print("GPU 메모리:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
        
        # 학습 실행
        trainer.train()
        print("\n모델 학습이 완료되었습니다.")
        
        # 수동으로 모델만 저장 (토크나이저 제외)
        print("모델을 저장합니다...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 모델 가중치만 저장
        model_path = os.path.join(OUTPUT_DIR, "pytorch_model.bin")
        torch.save(model.state_dict(), model_path)
        
        # 설정 파일 저장
        model.config.save_pretrained(OUTPUT_DIR)
        
        print(f"모델이 '{OUTPUT_DIR}'에 저장되었습니다.")
        print("저장된 파일:")
        for file in os.listdir(OUTPUT_DIR):
            print(f"  - {file}")
        
    except Exception as e:
        print(f"학습 중 오류가 발생했습니다: {e}")
        print("오류 타입:", type(e).__name__)
        import traceback
        traceback.print_exc() 