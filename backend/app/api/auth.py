from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import json
from gmail_api import gmail_api
import os
from google.auth.transport.requests import Request as GoogleRequest

router = APIRouter()

# 간단한 세션 저장소 (실제 운영에서는 Redis나 DB 사용 권장)
sessions = {}

def get_credentials_from_env():
    """환경변수에서 Gmail API credentials 가져오기"""
    credentials_json = os.getenv('GOOGLE_CREDENTIALS')
    if credentials_json:
        try:
            return json.loads(credentials_json)
        except json.JSONDecodeError:
            return None
    return None

@router.get("/test")
def test_connection():
    """연결 테스트용 API"""
    return {"message": "Auth API 연결 성공!", "status": "ok"}

@router.get("/check-gmail-auth")
def check_gmail_auth():
    """Gmail 인증 상태 확인"""
    try:
        # 환경변수에서 credentials 확인
        credentials_data = get_credentials_from_env()
        if not credentials_data:
            return {
                "authenticated": False, 
                "message": "Gmail API credentials가 설정되지 않음",
                "setup_required": True
            }
        
        # Gmail API 인증 시도 (팝업 없이 토큰 파일만 확인)
        creds = None
        if os.path.exists('token.json'):
            from google.oauth2.credentials import Credentials
            creds = Credentials.from_authorized_user_file('token.json', gmail_api.SCOPES)
            
            if creds and creds.valid:
                return {"authenticated": True, "message": "이미 인증됨"}
            elif creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(GoogleRequest())
                    return {"authenticated": True, "message": "토큰 갱신됨"}
                except:
                    return {"authenticated": False, "message": "토큰 갱신 실패"}
        
        return {
            "authenticated": False, 
            "message": "인증 필요",
            "setup_required": True
        }
        
    except Exception as e:
        return {"authenticated": False, "message": f"인증 확인 오류: {str(e)}"}

@router.get("/status")
def check_auth_status(request: Request):
    """로그인 상태 확인"""
    session_id = request.cookies.get("session_id")
    print("DEBUG: session_id from cookie =", session_id)
    print("DEBUG: sessions dict keys =", list(sessions.keys()))
    if session_id and session_id in sessions:
        user_info = sessions[session_id]
        return {
            "authenticated": True,
            "user": user_info
        }
    return {
        "authenticated": False,
        "user": None
    }

@router.post("/google-login")
async def google_login(req: Request):
    """Gmail API 인증을 통한 로그인 처리"""
    try:
        # 환경변수에서 credentials 확인
        credentials_data = get_credentials_from_env()
        if not credentials_data:
            return JSONResponse({
                "success": False,
                "error": "Gmail API credentials가 설정되지 않음. 환경변수 GOOGLE_CREDENTIALS를 설정해주세요."
            }, status_code=500)
        
        # credentials를 파일로 저장 (필요한 경우)
        if not os.path.exists('credentials.json'):
            with open('credentials.json', 'w') as f:
                json.dump(credentials_data, f)
        
        # 이미 인증된 토큰이 있는지 확인
        creds = None
        if os.path.exists('token.json'):
            from google.oauth2.credentials import Credentials
            creds = Credentials.from_authorized_user_file('token.json', gmail_api.SCOPES)
            
            if creds and creds.valid:
                pass  # 이미 유효한 토큰이 있음
            elif creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(GoogleRequest())
                except Exception as e:
                    print(f"토큰 갱신 실패: {e}")
                    creds = None  # 토큰 갱신 실패
        
        # 토큰이 없거나 유효하지 않으면 새로 인증
        if not creds:
            # 배포 환경에서는 브라우저 팝업이 불가능하므로 오류 메시지 반환
            return JSONResponse({
                "success": False,
                "error": "Gmail 인증이 필요합니다. 로컬 환경에서 먼저 인증을 완료하거나, 환경변수에 토큰을 설정해주세요.",
                "setup_required": True
            }, status_code=401)
        
        # 사용자 정보 가져오기
        user_info = gmail_api.get_user_profile(creds)
        if user_info is None:
            return JSONResponse({
                "success": False,
                "error": "사용자 정보 가져오기 실패"
            }, status_code=401)
        
        # 세션 생성
        import uuid
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "email": user_info.get("email", ""),
            "name": user_info.get("name", ""),
            "gmail_id": user_info.get("gmail_id", "")
        }
        
        response = JSONResponse({
            "success": True,
            "user": sessions[session_id]
        })
        
        # 쿠키 설정
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            max_age=3600,  # 1시간
            samesite="lax"
        )
        
        return response
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": f"로그인 처리 중 오류: {str(e)}"
        }, status_code=500)

@router.post("/logout")
def logout(request: Request):
    """로그아웃"""
    session_id = request.cookies.get("session_id")
    
    if session_id and session_id in sessions:
        del sessions[session_id]
    
    response = JSONResponse({"success": True})
    response.delete_cookie("session_id")
    
    return response 