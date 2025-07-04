import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_predictions(predictions, true_labels, sample_texts=None):
    """
    예측 결과를 자세히 분석
    
    Args:
        predictions: 예측된 라벨 리스트
        true_labels: 실제 라벨 리스트
        sample_texts: 샘플 텍스트 (선택사항)
    """
    print("="*60)
    print("예측 결과 상세 분석")
    print("="*60)
    
    # 1. 예측 분포
    print("\n📊 예측 분포:")
    pred_counter = Counter(predictions)
    for label, count in pred_counter.items():
        print(f"   {label}: {count}개 ({count/len(predictions)*100:.1f}%)")
    
    # 2. 실제 분포
    print("\n📊 실제 분포:")
    true_counter = Counter(true_labels)
    for label, count in true_counter.items():
        print(f"   {label}: {count}개 ({count/len(true_labels)*100:.1f}%)")
    
    # 3. 클래스별 정확도
    print("\n📈 클래스별 정확도:")
    for label in set(true_labels):
        mask = np.array(true_labels) == label
        correct = sum(np.array(predictions)[mask] == label)
        total = sum(mask)
        accuracy = correct / total if total > 0 else 0
        print(f"   {label}: {correct}/{total} = {accuracy:.4f}")
    
    # 4. 오분류 패턴 분석
    print("\n❌ 오분류 패턴:")
    errors = []
    for i, (pred, true) in enumerate(zip(predictions, true_labels)):
        if pred != true:
            errors.append((true, pred, i))
    
    print(f"   총 오분류: {len(errors)}개")
    
    # 오분류 유형별 분석
    error_patterns = Counter([(true, pred) for true, pred, _ in errors])
    for (true, pred), count in error_patterns.most_common():
        print(f"   {true} → {pred}: {count}개")
    
    # 5. 샘플 오분류 사례 (텍스트가 있는 경우)
    if sample_texts and len(errors) > 0:
        print("\n📝 샘플 오분류 사례:")
        for i in range(min(5, len(errors))):
            true, pred, idx = errors[i]
            if idx < len(sample_texts):
                text = sample_texts[idx][:100] + "..." if len(sample_texts[idx]) > 100 else sample_texts[idx]
                print(f"   실제: {true} → 예측: {pred}")
                print(f"   텍스트: {text}")
                print()
    
    # 6. 시각화
    plot_prediction_analysis(predictions, true_labels)

def plot_prediction_analysis(predictions, true_labels):
    """예측 결과 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 예측 분포
    pred_counts = pd.Series(predictions).value_counts()
    axes[0].bar(pred_counts.index, pred_counts.values, color='skyblue')
    axes[0].set_title('예측 분포')
    axes[0].set_ylabel('개수')
    
    # 실제 분포
    true_counts = pd.Series(true_labels).value_counts()
    axes[1].bar(true_counts.index, true_counts.values, color='lightcoral')
    axes[1].set_title('실제 분포')
    axes[1].set_ylabel('개수')
    
    plt.tight_layout()
    plt.savefig('prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n📈 시각화가 'prediction_analysis.png'에 저장되었습니다.")

def suggest_improvements(predictions, true_labels):
    """모델 개선 제안"""
    print("\n💡 모델 개선 제안:")
    
    # 1. 클래스 불균형 문제
    true_counter = Counter(true_labels)
    max_count = max(true_counter.values())
    min_count = min(true_counter.values())
    imbalance_ratio = max_count / min_count
    
    print(f"   1. 클래스 불균형 문제:")
    print(f"      - 가장 많은 클래스: {max_count}개")
    print(f"      - 가장 적은 클래스: {min_count}개")
    print(f"      - 불균형 비율: {imbalance_ratio:.1f}:1")
    
    if imbalance_ratio > 5:
        print(f"      → 클래스 가중치 조정 또는 데이터 증강 필요")
    
    # 2. 모델 편향 문제
    pred_counter = Counter(predictions)
    most_predicted = pred_counter.most_common(1)[0]
    most_predicted_ratio = most_predicted[1] / len(predictions)
    
    print(f"\n   2. 모델 편향 문제:")
    print(f"      - 가장 많이 예측된 클래스: {most_predicted[0]} ({most_predicted_ratio:.1%})")
    
    if most_predicted_ratio > 0.8:
        print(f"      → 모델이 한 클래스에 편향됨")
        print(f"      → 학습 데이터 재검토 또는 모델 재학습 필요")
    
    # 3. 구체적 개선 방안
    print(f"\n   3. 구체적 개선 방안:")
    print(f"      - 더 많은 에포크로 학습")
    print(f"      - 학습률 조정")
    print(f"      - 데이터 증강 (텍스트 변형)")
    print(f"      - 앙상블 모델 사용")
    print(f"      - 하이퍼파라미터 튜닝")

if __name__ == "__main__":
    # 예시 데이터로 테스트
    print("예측 결과 분석 도구입니다.")
    print("evaluate_kobert.py 실행 후 결과를 이 스크립트로 분석하세요.") 