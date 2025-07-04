import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import os

# 1. 데이터 로드
DATA_PATH = 'dataset/manual_labeled_300.csv'
df = pd.read_csv(DATA_PATH)
df = df[['body', 'label']].dropna()

# 2. 라벨 인코딩
label_list = sorted(df['label'].unique())
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}
df['label_id'] = df['label'].map(label2id)

# 3. train/validation 분리
train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label_id'])

# 4. Huggingface Dataset 변환
train_dataset = Dataset.from_pandas(train_df[['body', 'label_id']])
val_dataset = Dataset.from_pandas(val_df[['body', 'label_id']])

# 5. BERT 모델/토크나이저 준비
model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id
)

def preprocess(example):
    return tokenizer(example['body'], truncation=True, padding='max_length', max_length=256)

train_dataset = train_dataset.map(preprocess, batched=True)
val_dataset = val_dataset.map(preprocess, batched=True)

train_dataset = train_dataset.rename_column('label_id', 'labels')
val_dataset = val_dataset.rename_column('label_id', 'labels')

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# 6. Trainer 설정
output_dir = './bert_manual'
os.makedirs(output_dir, exist_ok=True)
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir=os.path.join(output_dir, 'logs'),
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 7. 학습
trainer.train()

# 8. 저장
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("학습 및 저장 완료! 결과 폴더:", output_dir) 