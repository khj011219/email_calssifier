from fastapi import APIRouter

router = APIRouter()

@router.get("/summary")
def stats_summary():
    # TODO: 분류 통계 데이터 반환
    return {"work": 10, "personal": 5, "advertisement": 3} 