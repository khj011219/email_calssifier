import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def test_single_prediction():
    """단일 이메일 텍스트에 대한 예측 테스트"""
    
    # 모델과 토크나이저 로드
    model_path = "models/kobert_trained/checkpoint-375"
    print(f"모델을 로드합니다: {model_path}")
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
    # 원본 KoBERT 토크나이저 사용 (저장된 토크나이저에 문제가 있음)
    tokenizer = AutoTokenizer.from_pretrained('monologg/kobert', trust_remote_code=True)
    
    # GPU 사용 가능 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    model.eval()
    model.to(device)
    
    # 라벨 매핑
    id2label = {0: "Work", 1: "Advertisement", 2: "Personal"}
    
    # 테스트할 이메일 텍스트들
    test_emails = [
        "내일 오후 2시에 프로젝트 회의가 있습니다. 참석 부탁드립니다.",
        "주말에 같이 영화 보러 갈까요? 재미있는 영화가 개봉했대요.",
        "한정 시간 특가! 지금 구매하시면 50% 할인! 서두르세요!",
        "오늘 날씨가 정말 좋네요. 저녁에 같이 밥 먹을까요?",
        "다음 주 월요일까지 보고서 제출해주세요. 급합니다.",
        "새로운 제품 출시 안내입니다. 지금 구매하시면 사은품 증정!"
    ]
    
    print("\n" + "="*60)
    print("KoBERT 모델 단일 예측 테스트")
    print("="*60)
    
    for i, email_text in enumerate(test_emails, 1):
        print(f"\n📧 테스트 이메일 {i}:")
        print(f"   텍스트: {email_text}")
        
        # 토크나이징
        inputs = tokenizer(
            email_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        )
        
        # GPU로 이동
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 예측
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predicted_id = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_id].item()
        
        predicted_label = id2label[predicted_id]
        
        # 결과 출력
        print(f"   예측 결과: {predicted_label}")
        print(f"   확신도: {confidence:.4f}")
        
        # 모든 클래스별 확률
        print(f"   클래스별 확률:")
        for class_id, class_name in id2label.items():
            prob = probabilities[0][class_id].item()
            print(f"     {class_name}: {prob:.4f}")
        
        print("-" * 40)

if __name__ == "__main__":
    test_single_prediction() 