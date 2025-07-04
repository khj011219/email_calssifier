import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json

class KoBERTEvaluator:
    def __init__(self, model_path, test_data_path):
        """
        KoBERT 모델 평가기 초기화
        
        Args:
            model_path: 학습된 모델 경로
            test_data_path: 테스트 데이터 경로
        """
        self.model_path = model_path
        self.test_data_path = test_data_path
        
        # GPU 사용 가능 여부 확인
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"사용 디바이스: {self.device}")
        
        # 모델과 토크나이저 로드
        print("모델과 토크나이저를 로드합니다...")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
        # 원본 KoBERT 토크나이저 사용 (저장된 토크나이저에 문제가 있음)
        self.tokenizer = AutoTokenizer.from_pretrained('monologg/kobert', trust_remote_code=True)
        
        # 모델을 평가 모드로 설정
        self.model.eval()
        self.model.to(self.device)
        
        # 라벨 매핑 로드
        self.label2id = {"Work": 0, "Advertisement": 1, "Personal": 2}
        self.id2label = {0: "Work", 1: "Advertisement", 2: "Personal"}
        
        print("모델 로드 완료!")
    
    def load_test_data(self, sample_size=None):
        """
        테스트 데이터 로드
        
        Args:
            sample_size: 샘플 크기 (None이면 전체 데이터)
        """
        print(f"테스트 데이터를 로드합니다... ({self.test_data_path})")
        
        # 데이터 로드
        df = pd.read_csv(self.test_data_path)
        
        # 샘플링 (메모리 효율성을 위해)
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            print(f"데이터 샘플링: {sample_size}개 사용")
        
        # 라벨을 숫자로 변환
        df['label_id'] = df['label'].map(self.label2id)
        
        # NaN 값 제거
        df = df.dropna(subset=['text', 'label_id'])
        
        print(f"로드된 데이터: {len(df)}개")
        print(f"라벨 분포:\n{df['label'].value_counts()}")
        
        return df
    
    def predict_single(self, text):
        """
        단일 텍스트에 대한 예측
        
        Args:
            text: 입력 텍스트
            
        Returns:
            predicted_label: 예측된 라벨
            confidence: 확신도
        """
        # 토크나이징
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        )
        
        # GPU로 이동
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 예측
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predicted_id = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_id].item()
        
        predicted_label = self.id2label[predicted_id]
        
        return predicted_label, confidence
    
    def evaluate_batch(self, df, batch_size=32):
        """
        배치 단위로 평가
        
        Args:
            df: 테스트 데이터프레임
            batch_size: 배치 크기
            
        Returns:
            predictions: 예측 결과 리스트
            confidences: 확신도 리스트
        """
        predictions = []
        confidences = []
        
        print("배치 단위로 예측을 수행합니다...")
        
        for i in tqdm(range(0, len(df), batch_size)):
            batch_df = df.iloc[i:i+batch_size]
            batch_texts = batch_df['text'].tolist()
            
            # 토크나이징
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # GPU로 이동
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 예측
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                predicted_ids = torch.argmax(probabilities, dim=-1)
                batch_confidences = torch.max(probabilities, dim=-1)[0]
            
            # 결과 변환
            batch_predictions = [self.id2label[id.item()] for id in predicted_ids]
            batch_confidences = batch_confidences.cpu().numpy()
            
            predictions.extend(batch_predictions)
            confidences.extend(batch_confidences)
        
        return predictions, confidences
    
    def calculate_metrics(self, true_labels, predicted_labels):
        """
        성능 메트릭 계산
        
        Args:
            true_labels: 실제 라벨
            predicted_labels: 예측 라벨
            
        Returns:
            metrics: 성능 메트릭 딕셔너리
        """
        # 기본 메트릭
        accuracy = accuracy_score(true_labels, predicted_labels)
        f1_macro = f1_score(true_labels, predicted_labels, average='macro')
        f1_weighted = f1_score(true_labels, predicted_labels, average='weighted')
        
        # 클래스별 F1-Score
        f1_per_class = f1_score(true_labels, predicted_labels, average=None)
        
        # 분류 리포트
        class_report = classification_report(
            true_labels, 
            predicted_labels, 
            target_names=list(self.label2id.keys()),
            output_dict=True
        )
        
        # 혼동 행렬
        cm = confusion_matrix(true_labels, predicted_labels)
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_per_class': f1_per_class,
            'classification_report': class_report,
            'confusion_matrix': cm
        }
        
        return metrics
    
    def plot_confusion_matrix(self, confusion_matrix, save_path=None):
        """
        혼동 행렬 시각화
        
        Args:
            confusion_matrix: 혼동 행렬
            save_path: 저장 경로 (None이면 표시만)
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=list(self.label2id.keys()),
            yticklabels=list(self.label2id.keys())
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"혼동 행렬이 {save_path}에 저장되었습니다.")
        
        plt.show()
    
    def print_results(self, metrics):
        """
        결과 출력
        
        Args:
            metrics: 성능 메트릭
        """
        print("\n" + "="*50)
        print("KoBERT 모델 성능 평가 결과")
        print("="*50)
        
        print(f"\n📊 전체 성능:")
        print(f"   정확도 (Accuracy): {metrics['accuracy']:.4f}")
        print(f"   F1-Score (Macro): {metrics['f1_macro']:.4f}")
        print(f"   F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
        
        print(f"\n📈 클래스별 F1-Score:")
        for i, label in enumerate(self.label2id.keys()):
            print(f"   {label}: {metrics['f1_per_class'][i]:.4f}")
        
        print(f"\n📋 상세 분류 리포트:")
        # classification_report 에러 수정
        try:
            print(classification_report(
                list(self.label2id.keys()),
                metrics['classification_report'],
                target_names=list(self.label2id.keys())
            ))
        except:
            # 에러 발생 시 간단한 리포트 출력
            print("상세 분류 리포트를 생성할 수 없습니다.")
            print("혼동 행렬을 확인해주세요.")
        
        # 혼동 행렬 출력
        print(f"\n🔍 혼동 행렬:")
        cm = metrics['confusion_matrix']
        print("실제\예측\tWork\tAdvertisement\tPersonal")
        for i, true_label in enumerate(self.label2id.keys()):
            print(f"{true_label}\t\t{cm[i][0]}\t{cm[i][1]}\t\t{cm[i][2]}")
        
        # 클래스별 예측 분포 분석
        print(f"\n📊 예측 분포 분석:")
        unique, counts = np.unique(list(self.label2id.keys()), return_counts=True)
        for label, count in zip(unique, counts):
            print(f"   {label}: {count}개 예측")
    
    def save_results(self, metrics, predictions, confidences, save_path):
        """
        결과를 파일로 저장
        
        Args:
            metrics: 성능 메트릭
            predictions: 예측 결과
            confidences: 확신도
            save_path: 저장 경로
        """
        results = {
            'metrics': {
                'accuracy': float(metrics['accuracy']),
                'f1_macro': float(metrics['f1_macro']),
                'f1_weighted': float(metrics['f1_weighted']),
                'f1_per_class': metrics['f1_per_class'].tolist()
            },
            'predictions': predictions,
            'confidences': [float(c) for c in confidences]
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n결과가 {save_path}에 저장되었습니다.")
    
    def run_evaluation(self, sample_size=1000, batch_size=32, save_results=True):
        """
        전체 평가 실행
        
        Args:
            sample_size: 테스트 데이터 샘플 크기
            batch_size: 배치 크기
            save_results: 결과 저장 여부
        """
        print("KoBERT 모델 평가를 시작합니다...")
        
        # 테스트 데이터 로드
        test_df = self.load_test_data(sample_size=sample_size)
        
        # 예측 수행
        predictions, confidences = self.evaluate_batch(test_df, batch_size=batch_size)
        
        # 메트릭 계산
        true_labels = test_df['label'].tolist()
        metrics = self.calculate_metrics(true_labels, predictions)
        
        # 결과 출력
        self.print_results(metrics)
        
        # 혼동 행렬 시각화
        self.plot_confusion_matrix(metrics['confusion_matrix'])
        
        # 결과 저장
        if save_results:
            results_path = os.path.join(os.path.dirname(self.model_path), 'evaluation_results.json')
            self.save_results(metrics, predictions, confidences, results_path)
        
        # 평균 확신도 계산
        avg_confidence = np.mean(confidences)
        print(f"\n📊 평균 확신도: {avg_confidence:.4f}")
        
        return metrics, predictions, confidences

def main():
    """메인 함수"""
    # 설정
    model_path = "models/kobert_trained/checkpoint-375"
    test_data_path = "data/processed/validation.csv"
    
    # 평가기 초기화
    evaluator = KoBERTEvaluator(model_path, test_data_path)
    
    # 평가 실행 (1000개 샘플로 테스트)
    metrics, predictions, confidences = evaluator.run_evaluation(
        sample_size=1000,  # 메모리 효율성을 위해 샘플링
        batch_size=16,     # GPU 메모리에 맞게 조정
        save_results=True
    )
    
    print("\n✅ 평가 완료!")

if __name__ == "__main__":
    main() 