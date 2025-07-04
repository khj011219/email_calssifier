from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api import auth, gmail, classify, stats

app = FastAPI()

# CORS 설정 (배포 환경 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"  # 배포 시 모든 도메인 허용 (보안상 나중에 제한 필요)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth")
app.include_router(gmail.router, prefix="/gmail")
app.include_router(classify.router, prefix="/classify")
app.include_router(stats.router, prefix="/stats")

@app.get("/")
def root():
    return {"message": "Email Classifier FastAPI 백엔드 동작 중!"} 