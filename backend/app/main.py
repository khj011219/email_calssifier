from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from backend.app.api import auth, gmail, classify, stats
import os

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
    try:
        return FileResponse("index.html")
    except Exception as e:
        return {"error": f"index.html 파일을 찾을 수 없습니다: {e}"}

@app.get("/login")
def login_page():
    try:
        return FileResponse("login.html")
    except Exception as e:
        return {"error": f"login.html 파일을 찾을 수 없습니다: {e}"}

@app.get("/static/{filename}")
def static_files(filename: str):
    """정적 파일 서빙 (JS, CSS 등)"""
    try:
        if filename.endswith('.js'):
            return FileResponse(f"app.js", media_type="application/javascript")
        elif filename.endswith('.css'):
            return FileResponse(f"style.css", media_type="text/css")
        else:
            return {"error": "지원하지 않는 파일 형식입니다"}
    except Exception as e:
        return {"error": f"파일을 찾을 수 없습니다: {e}"} 