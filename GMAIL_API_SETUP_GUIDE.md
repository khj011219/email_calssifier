# 📧 Gmail API 설정 완전 가이드

이 가이드는 Gmail API를 사용하기 위한 Google Cloud Console 설정 과정을 단계별로 안내합니다.

## 🔧 1단계: Google Cloud Console 프로젝트 생성

### 1.1 Google Cloud Console 접속
1. [Google Cloud Console](https://console.cloud.google.com/)에 접속
2. Google 계정으로 로그인

### 1.2 새 프로젝트 생성
1. 상단의 프로젝트 선택 드롭다운 클릭
2. "새 프로젝트" 클릭
3. 프로젝트 이름 입력 (예: `email-classifier-app`)
4. "만들기" 클릭
5. 프로젝트가 생성될 때까지 대기

## 🔧 2단계: OAuth 동의 화면 구성

### 2.1 OAuth 동의 화면 접속
1. 왼쪽 메뉴에서 "API 및 서비스" > "OAuth 동의 화면" 클릭
2. 사용자 유형 선택:
   - **외부**: 모든 Google 계정 사용자 (권장)
   - **내부**: 조직 내 사용자만

### 2.2 앱 정보 입력
```
앱 이름: 이메일 분류 시스템
사용자 지원 이메일: [your-email@gmail.com]
앱 로고: (선택사항)
```

### 2.3 범위 추가
1. "범위 추가 또는 삭제" 클릭
2. 다음 범위들을 추가:
   - `https://www.googleapis.com/auth/gmail.readonly` (Gmail 읽기)
   - `https://www.googleapis.com/auth/userinfo.email` (이메일 주소)
   - `https://www.googleapis.com/auth/userinfo.profile` (기본 프로필 정보)

### 2.4 테스트 사용자 추가 (외부 앱인 경우)
1. "테스트 사용자" 섹션에서 "테스트 사용자 추가" 클릭
2. 본인의 Gmail 주소 추가
3. "저장 후 계속" 클릭

### 2.5 요약 및 완료
1. 모든 정보를 확인
2. "대시보드로 돌아가기" 클릭

## 🔧 3단계: API 활성화

### 3.1 API 라이브러리 접속
1. 왼쪽 메뉴에서 "API 및 서비스" > "라이브러리" 클릭

### 3.2 Gmail API 활성화
1. 검색창에 "Gmail API" 입력
2. Gmail API 선택
3. "사용" 버튼 클릭

### 3.3 People API 활성화
1. 검색창에 "People API" 입력
2. People API 선택
3. "사용" 버튼 클릭

## 🔧 4단계: OAuth 2.0 클라이언트 ID 생성

### 4.1 사용자 인증 정보 접속
1. 왼쪽 메뉴에서 "API 및 서비스" > "사용자 인증 정보" 클릭

### 4.2 OAuth 2.0 클라이언트 ID 생성
1. "사용자 인증 정보 만들기" 클릭
2. "OAuth 클라이언트 ID" 선택

### 4.3 애플리케이션 유형 선택
1. 애플리케이션 유형: **"데스크톱 앱"** 선택
2. 이름 입력: `이메일 분류 시스템 - 데스크톱`

### 4.4 클라이언트 ID 생성
1. "만들기" 클릭
2. 생성된 클라이언트 ID와 클라이언트 보안 비밀번호 확인

## 🔧 5단계: credentials.json 파일 다운로드

### 5.1 JSON 파일 다운로드
1. 생성된 OAuth 2.0 클라이언트 ID 옆의 다운로드 버튼 클릭
2. JSON 파일이 다운로드됨

### 5.2 파일 위치 이동
1. 다운로드된 JSON 파일을 프로젝트 루트 디렉토리로 이동
2. 파일명을 `credentials.json`으로 변경

## 🔧 6단계: 애플리케이션 테스트

### 6.1 Streamlit 앱 실행
```bash
streamlit run app.py
```

### 6.2 브라우저에서 접속
1. `http://localhost:8501` 접속
2. "🔗 Google 계정으로 로그인" 버튼 클릭
3. Google 계정 선택 및 권한 허용

## 🚨 주의사항

### 외부 앱의 경우
- **테스트 사용자 제한**: 최대 100명까지 테스트 사용자 추가 가능
- **검증 필요**: 100명 이상 사용하려면 Google 검증 과정 필요
- **개발 중**: 테스트 사용자만 사용 가능

### API 할당량
- **Gmail API**: 일일 1,000,000,000 할당량 단위
- **People API**: 일일 10,000 할당량 단위
- **모니터링**: Google Cloud Console에서 사용량 확인 가능

## 🔍 문제 해결

### "OAuth 동의 화면이 구성되지 않았습니다" 오류
- OAuth 동의 화면을 먼저 구성해야 함
- 위의 2단계 과정을 완료

### "이 앱이 확인되지 않았습니다" 경고
- 개발 중인 앱이므로 정상
- "고급" > "안전하지 않은 앱으로 이동" 클릭

### "권한이 거부되었습니다" 오류
- 테스트 사용자 목록에 본인 이메일 추가
- 올바른 Google 계정으로 로그인

### API 할당량 초과
- Google Cloud Console에서 할당량 확인
- 필요시 할당량 증가 요청

## 📞 추가 지원

문제가 발생하면 다음을 확인하세요:
1. [Google Cloud Console 문서](https://cloud.google.com/docs)
2. [Gmail API 문서](https://developers.google.com/gmail/api)
3. [OAuth 2.0 가이드](https://developers.google.com/identity/protocols/oauth2)

---

**참고**: 이 설정은 개발 및 테스트 목적입니다. 프로덕션 환경에서는 추가 보안 설정이 필요할 수 있습니다. 