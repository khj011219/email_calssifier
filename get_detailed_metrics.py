import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import os

def get_detailed_metrics():
    """모든 모델의 상세한 성능 지표를 계산합니다."""
    
    print("="*80)
    print("모든 모델 상세 성능 지표 (F1, Recall, Accuracy, Precision)")
    print("="*80)
    
    # 평가 결과 파일들
    evaluation_files = [
        "bert_manual_300_model/test_dataset_evaluation_checkpoint-264.json",
        "models/kobert_trained/test_dataset_evaluation_checkpoint-375.json",
        "models/kobert_trained_improved/test_dataset_evaluation_checkpoint-500.json"
    ]
    
    results_summary = []
    
    for file_path in evaluation_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                model_name = os.path.basename(data['model_path'])
                test_dataset = os.path.basename(data['test_dataset'])
                
                print(f"\n{'='*60}")
                print(f"모델: {model_name}")
                print(f"테스트 데이터셋: {test_dataset}")
                print(f"총 샘플 수: {data['total_samples']}")
                print(f"{'='*60}")
                
                # 기본 메트릭
                accuracy = data['metrics']['accuracy']
                f1_macro = data['metrics']['f1_macro']
                f1_weighted = data['metrics']['f1_weighted']
                f1_per_class = data['metrics']['f1_per_class']
                
                print(f"\n📊 전체 성능:")
                print(f"   정확도 (Accuracy): {accuracy:.4f}")
                print(f"   F1-Score (Macro): {f1_macro:.4f}")
                print(f"   F1-Score (Weighted): {f1_weighted:.4f}")
                
                # 클래스별 상세 지표 계산
                predictions = data['predictions']
                confidences = data['confidences']
                
                # 실제 라벨 추출 (오버샘플링된 데이터의 경우)
                if test_dataset == "test_labeled_dataset_oversampled.csv":
                    # 오버샘플링된 데이터에서 실제 라벨 추출
                    df = pd.read_csv('dataset/test_labeled_dataset_oversampled.csv')
                    true_labels = df['label'].tolist()
                else:
                    # 원본 데이터에서 실제 라벨 추출
                    df = pd.read_csv('dataset/test_labeled_dataset.csv')
                    true_labels = df['label'].tolist()
                
                # 클래스별 상세 지표 계산
                labels = ["Work", "Personal", "Advertisement"]
                precision, recall, f1, support = precision_recall_fscore_support(
                    true_labels, predictions, labels=labels, average=None
                )
                
                print(f"\n📈 클래스별 상세 지표:")
                print(f"{'클래스':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
                print("-" * 55)
                
                for i, label in enumerate(labels):
                    print(f"{label:<12} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {support[i]:<10.0f}")
                
                # 전체 평균 지표
                precision_macro, recall_macro, f1_macro_calc, _ = precision_recall_fscore_support(
                    true_labels, predictions, average='macro'
                )
                precision_weighted, recall_weighted, f1_weighted_calc, _ = precision_recall_fscore_support(
                    true_labels, predictions, average='weighted'
                )
                
                print(f"\n📊 전체 평균 지표:")
                print(f"   Precision (Macro): {precision_macro:.4f}")
                print(f"   Recall (Macro): {recall_macro:.4f}")
                print(f"   F1-Score (Macro): {f1_macro_calc:.4f}")
                print(f"   Precision (Weighted): {precision_weighted:.4f}")
                print(f"   Recall (Weighted): {recall_weighted:.4f}")
                print(f"   F1-Score (Weighted): {f1_weighted_calc:.4f}")
                
                # 평균 확신도
                avg_confidence = np.mean(confidences)
                print(f"   평균 확신도: {avg_confidence:.4f}")
                
                # 결과 요약 저장
                results_summary.append({
                    'model': model_name,
                    'dataset': test_dataset,
                    'accuracy': accuracy,
                    'precision_macro': precision_macro,
                    'recall_macro': recall_macro,
                    'f1_macro': f1_macro_calc,
                    'precision_weighted': precision_weighted,
                    'recall_weighted': recall_weighted,
                    'f1_weighted': f1_weighted_calc,
                    'avg_confidence': avg_confidence,
                    'total_samples': data['total_samples']
                })
                
                # 혼동 행렬 출력
                print(f"\n🔍 혼동 행렬:")
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(true_labels, predictions, labels=labels)
                print("실제\\예측\tWork\tPersonal\tAdvertisement")
                for i, true_label in enumerate(labels):
                    print(f"{true_label}\t\t{cm[i][0]}\t{cm[i][1]}\t\t{cm[i][2]}")
                
            except Exception as e:
                print(f"파일 {file_path} 처리 중 오류 발생: {e}")
        else:
            print(f"파일이 존재하지 않습니다: {file_path}")
    
    # 모든 모델 비교
    if results_summary:
        print(f"\n{'='*80}")
        print("모든 모델 성능 비교")
        print(f"{'='*80}")
        
        df_summary = pd.DataFrame(results_summary)
        
        # 정확도 기준 정렬
        df_summary_sorted = df_summary.sort_values('accuracy', ascending=False)
        
        print(f"\n📊 성능 순위 (정확도 기준):")
        print(f"{'순위':<4} {'모델':<25} {'데이터셋':<30} {'Accuracy':<10} {'F1-Macro':<10} {'Recall-Macro':<12} {'Precision-Macro':<15}")
        print("-" * 120)
        
        for i, (_, row) in enumerate(df_summary_sorted.iterrows(), 1):
            print(f"{i:<4} {row['model']:<25} {row['dataset']:<30} {row['accuracy']:<10.4f} {row['f1_macro']:<10.4f} {row['recall_macro']:<12.4f} {row['precision_macro']:<15.4f}")
        
        # 상세 비교표
        print(f"\n📋 상세 비교표:")
        print(df_summary_sorted[['model', 'dataset', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'avg_confidence']].to_string(index=False))
        
        # 결과를 CSV로 저장
        output_file = "model_performance_comparison.csv"
        df_summary_sorted.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n✅ 상세 비교 결과가 {output_file}에 저장되었습니다.")
        
        return df_summary_sorted
    
    return None

def analyze_best_model():
    """가장 성능이 좋은 모델의 상세 분석을 수행합니다."""
    
    print(f"\n{'='*80}")
    print("최고 성능 모델 상세 분석")
    print(f"{'='*80}")
    
    # 가장 성능이 좋은 모델 찾기
    evaluation_files = [
        "bert_manual_300_model/test_dataset_evaluation_checkpoint-264.json",
        "models/kobert_trained/test_dataset_evaluation_checkpoint-375.json",
        "models/kobert_trained_improved/test_dataset_evaluation_checkpoint-500.json"
    ]
    
    best_model = None
    best_accuracy = 0
    
    for file_path in evaluation_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            accuracy = data['metrics']['accuracy']
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = data
    
    if best_model:
        print(f"최고 성능 모델: {best_model['model_path']}")
        print(f"정확도: {best_model['metrics']['accuracy']:.4f}")
        print(f"F1-Macro: {best_model['metrics']['f1_macro']:.4f}")
        
        # 상세 분석
        predictions = best_model['predictions']
        confidences = best_model['confidences']
        
        # 실제 라벨 추출
        test_dataset = os.path.basename(best_model['test_dataset'])
        if test_dataset == "test_labeled_dataset_oversampled.csv":
            df = pd.read_csv('dataset/test_labeled_dataset_oversampled.csv')
        else:
            df = pd.read_csv('dataset/test_labeled_dataset.csv')
        
        true_labels = df['label'].tolist()
        
        # 분류 리포트
        print(f"\n📋 상세 분류 리포트:")
        print(classification_report(true_labels, predictions, target_names=["Work", "Personal", "Advertisement"]))
        
        # 확신도 분석
        print(f"\n📊 확신도 분석:")
        print(f"평균 확신도: {np.mean(confidences):.4f}")
        print(f"최소 확신도: {np.min(confidences):.4f}")
        print(f"최대 확신도: {np.max(confidences):.4f}")
        
        # 라벨별 확신도
        df_analysis = pd.DataFrame({
            'true_label': true_labels,
            'predicted_label': predictions,
            'confidence': confidences,
            'correct': [t == p for t, p in zip(true_labels, predictions)]
        })
        
        print(f"\n라벨별 평균 확신도:")
        for label in df_analysis['true_label'].unique():
            label_data = df_analysis[df_analysis['true_label'] == label]
            avg_conf = label_data['confidence'].mean()
            accuracy = label_data['correct'].mean()
            print(f"{label}: 확신도 {avg_conf:.4f}, 정확도 {accuracy:.4f}")

if __name__ == "__main__":
    # 상세 지표 계산
    results = get_detailed_metrics()
    
    # 최고 성능 모델 분석
    analyze_best_model() 