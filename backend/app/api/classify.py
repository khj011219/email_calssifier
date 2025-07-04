from fastapi import APIRouter, Body

router = APIRouter()

@router.post("/")
def classify(data: dict = Body(...)):
    # TODO: BERT 모델 inference 연동
    return {"input": data, "pred_label": "Work", "confidence": 0.95} 