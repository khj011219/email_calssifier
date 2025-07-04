import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

# 1. 데이터 로드
df = pd.read_csv('dataset/emails_processed.csv')

# 2. 모델/토크나이저 로드
model_dir = './bert_manual'
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 3. 라벨 매핑
id2label = model.config.id2label

# 4. 예측 및 신뢰도 필터링
threshold = 0.9
results = []

for idx, body in tqdm(enumerate(df['body']), total=len(df), desc="라벨 예측 및 필터링"):
    text = str(body)[:1000]  # 너무 길면 1000자까지만
    inputs = tokenizer(text, truncation=True, padding='max_length', max_length=256, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        label = id2label[pred.item()]
        if conf.item() >= threshold:
            row = df.iloc[idx].copy()
            row['pred_label'] = label
            row['pred_confidence'] = conf.item()
            results.append(row)

# 5. 결과 저장
if results:
    confident_df = pd.DataFrame(results)
    confident_df.to_csv('dataset/emails_processed_labeled_confident.csv', index=False, encoding='utf-8-sig')
    print(f"신뢰도 {threshold} 이상 데이터만 남김: {len(confident_df)}개")
    print("결과: dataset/emails_processed_labeled_confident.csv")
else:
    print("신뢰도 기준을 만족하는 데이터가 없습니다.") 