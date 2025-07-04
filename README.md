# 📧 이메일 분류 시스템

Gmail API를 활용한 이메일 자동 분류 웹 애플리케이션입니다.

## 🚀 주요 기능

- **Gmail API 연동**: 실제 Gmail 계정에서 이메일 가져오기
- **AI 분류**: 업무/개인/광고 카테고리로 자동 분류
- **웹 인터페이스**: 직관적인 웹 UI로 이메일 관리
- **재분류 기능**: AI 분류 결과 수정 및 저장

## 🛠️ 기술 스택

- **백엔드**: FastAPI (Python)
- **프론트엔드**: HTML, JavaScript, CSS
- **AI 모델**: BERT (한국어 이메일 분류)
- **인증**: Google OAuth 2.0

## 📋 설치 및 실행

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. Gmail API 설정
1. [Google Cloud Console](https://console.cloud.google.com/)에서 프로젝트 생성
2. Gmail API 활성화
3. OAuth 2.0 클라이언트 ID 생성
4. `credentials.json` 파일 다운로드하여 프로젝트 루트에 저장

### 3. 서버 실행
```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. 웹페이지 접속
브라우저에서 `index.html` 파일을 열거나, 정적 파일 서버를 사용

## 🌐 배포

### Render.com 배포
1. GitHub에 코드 업로드
2. [Render.com](https://render.com)에서 새 Web Service 생성
3. GitHub 저장소 연결
4. 환경 설정:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`

### 환경변수 설정
- `GOOGLE_CREDENTIALS`: Gmail API credentials.json 내용
- `SECRET_KEY`: 세션 암호화 키

## 📁 프로젝트 구조

```
email_classifier/
├── backend/
│   └── app/
│       ├── main.py          # FastAPI 메인 앱
│       └── api/
│           ├── auth.py      # 인증 API
│           ├── gmail.py     # Gmail API
│           ├── classify.py  # 분류 API
│           └── stats.py     # 통계 API
├── index.html               # 메인 페이지
├── login.html              # 로그인 페이지
├── app.js                  # 프론트엔드 로직
├── requirements.txt        # Python 의존성
└── Procfile               # 배포 설정
```

## 🔒 보안 고려사항

- Gmail API credentials는 환경변수로 관리
- CORS 설정은 실제 도메인으로 제한
- HTTPS 사용 권장

## 📞 지원

문제가 발생하거나 질문이 있으시면 이슈를 생성해주세요.

---

**참고**: 이 프로젝트는 교육 및 개발 목적으로 제작되었습니다. 