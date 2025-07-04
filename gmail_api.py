import os
import json
import base64
import email
from datetime import datetime
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import streamlit as st

class GmailAPI:
    def __init__(self):
        """Gmail API 초기화"""
        # Gmail API 스코프 (읽기 전용 + 사용자 정보)
        self.SCOPES = [
            'openid',
            'https://www.googleapis.com/auth/gmail.readonly',
            'https://www.googleapis.com/auth/userinfo.email',
            'https://www.googleapis.com/auth/userinfo.profile'
        ]
        
        # OAuth 2.0 클라이언트 설정 파일 경로
        self.CREDENTIALS_FILE = 'credentials.json'
        
        # 토큰 파일 경로
        self.TOKEN_FILE = 'token.json'
    
    def create_credentials_file(self):
        """OAuth 2.0 클라이언트 설정 파일 생성 가이드"""
        credentials_template = {
            "installed": {
                "client_id": "YOUR_CLIENT_ID.apps.googleusercontent.com",
                "project_id": "your-project-id",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_secret": "YOUR_CLIENT_SECRET",
                "redirect_uris": ["http://localhost"]
            }
        }
        
        if not os.path.exists(self.CREDENTIALS_FILE):
            with open(self.CREDENTIALS_FILE, 'w') as f:
                json.dump(credentials_template, f, indent=2)
            
            st.warning(f"""
            📋 Gmail API 설정이 필요합니다!
            
            1. Google Cloud Console에서 프로젝트를 생성하세요
            2. Gmail API와 People API를 활성화하세요
            3. OAuth 2.0 클라이언트 ID를 생성하세요
            4. 생성된 credentials.json 파일을 {self.CREDENTIALS_FILE}에 저장하세요
            
            자세한 설정 방법은 다음 링크를 참고하세요:
            https://developers.google.com/gmail/api/quickstart/python
            """)
            return False
        
        return True
    
    def authenticate_user(self):
        """사용자 인증 및 토큰 생성"""
        creds = None
        
        # 토큰 파일이 있으면 로드
        if os.path.exists(self.TOKEN_FILE):
            creds = Credentials.from_authorized_user_file(self.TOKEN_FILE, self.SCOPES)
        
        # 유효한 인증 정보가 없거나 만료된 경우
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    st.error(f"토큰 갱신 중 오류: {e}")
                    return None
            else:
                if not self.create_credentials_file():
                    return None
                
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.CREDENTIALS_FILE, self.SCOPES)
                    creds = flow.run_local_server(port=0)
                except Exception as e:
                    st.error(f"인증 중 오류: {e}")
                    return None
            
            # 토큰을 파일에 저장
            with open(self.TOKEN_FILE, 'w') as token:
                token.write(creds.to_json())
        
        return creds
    
    def get_gmail_service(self, creds):
        """Gmail 서비스 객체 생성"""
        try:
            service = build('gmail', 'v1', credentials=creds)
            return service
        except HttpError as error:
            st.error(f'Gmail 서비스 생성 중 오류: {error}')
            return None
    
    def get_user_profile(self, creds):
        """사용자 프로필 정보 가져오기"""
        try:
            service = build('gmail', 'v1', credentials=creds)
            profile = service.users().getProfile(userId='me').execute()
            
            # Google People API로 추가 정보 가져오기
            try:
                people_service = build('people', 'v1', credentials=creds)
                person = people_service.people().get(
                    resourceName='people/me',
                    personFields='names,emailAddresses,photos'
                ).execute()
                
                name = person.get('names', [{}])[0].get('displayName', 'Unknown')
                email = person.get('emailAddresses', [{}])[0].get('value', profile.get('emailAddress', ''))
                picture_url = person.get('photos', [{}])[0].get('url', '')
            except:
                # People API 실패시 Gmail 프로필 정보만 사용
                name = profile.get('emailAddress', '').split('@')[0]
                email = profile.get('emailAddress', '')
                picture_url = ''
            
            return {
                'gmail_id': profile.get('id', ''),
                'email': email,
                'name': name,
                'picture_url': picture_url
            }
            
        except HttpError as error:
            st.error(f'사용자 프로필 가져오기 중 오류: {error}')
            return None
    
    def list_messages(self, service, user_id='me', max_results=10, query=''):
        """이메일 목록 조회"""
        try:
            # 검색 쿼리 구성
            search_query = query if query else 'is:inbox'
            
            # 이메일 목록 요청
            response = service.users().messages().list(
                userId=user_id,
                q=search_query,
                maxResults=max_results
            ).execute()
            
            messages = response.get('messages', [])
            
            if not messages:
                st.info('이메일이 없습니다.')
                return []
            
            return messages
            
        except HttpError as error:
            st.error(f'이메일 목록 조회 중 오류: {error}')
            return []
    
    def get_message_details(self, service, message_id, user_id='me'):
        """이메일 상세 정보 조회"""
        try:
            # 이메일 상세 정보 요청
            message = service.users().messages().get(
                userId=user_id, 
                id=message_id,
                format='full'
            ).execute()
            
            # 헤더 정보 추출
            headers = message['payload']['headers']
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')
            date = next((h['value'] for h in headers if h['name'] == 'Date'), 'Unknown')
            
            # 본문 추출
            body = self.extract_message_body(message['payload'])
            
            return {
                'id': message_id,
                'subject': subject,
                'sender': sender,
                'date': date,
                'body': body,
                'snippet': message.get('snippet', '')
            }
            
        except HttpError as error:
            st.error(f'이메일 상세 정보 조회 중 오류: {error}')
            return None
    
    def extract_message_body(self, payload):
        """이메일 본문 추출"""
        body = ""
        
        if 'parts' in payload:
            # 멀티파트 메시지
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    if 'data' in part['body']:
                        body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                        break
                elif part['mimeType'] == 'text/html':
                    if 'data' in part['body']:
                        body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                        break
        else:
            # 단일 파트 메시지
            if 'data' in payload['body']:
                body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
        
        return body
    
    def search_emails(self, service, query='', max_results=10):
        """이메일 검색"""
        try:
            # 검색 쿼리 구성
            search_query = query  # 쿼리가 비어 있으면 전체 메일함
            
            # 검색 실행
            response = service.users().messages().list(
                userId='me',
                q=search_query,
                maxResults=max_results
            ).execute()
            
            messages = response.get('messages', [])
            
            if not messages:
                st.info('검색 결과가 없습니다.')
                return []
            
            # 각 이메일의 상세 정보 조회
            email_details = []
            with st.spinner('이메일 정보를 가져오는 중...'):
                for message in messages:
                    details = self.get_message_details(service, message['id'])
                    if details:
                        email_details.append(details)
            
            return email_details
            
        except HttpError as error:
            st.error(f'이메일 검색 중 오류: {error}')
            return []
    
    def get_recent_emails(self, service, max_results=10, query=None):
        """최근 이메일 조회 (전체 메일함, 쿼리 지원)"""
        if query is None:
            query = ''
        return self.search_emails(service, query, max_results)
    
    def get_unread_emails(self, service, max_results=10):
        """읽지 않은 이메일 조회"""
        return self.search_emails(service, 'is:unread', max_results)
    
    def mark_as_read(self, service, message_id, user_id='me'):
        """이메일을 읽음으로 표시"""
        try:
            service.users().messages().modify(
                userId=user_id,
                id=message_id,
                body={'removeLabelIds': ['UNREAD']}
            ).execute()
            return True
        except HttpError as error:
            st.error(f'이메일 상태 변경 중 오류: {error}')
            return False
    
    def get_email_statistics(self, service, days=7):
        """이메일 통계 가져오기"""
        try:
            # 최근 N일간의 이메일 조회
            from datetime import datetime, timedelta
            date_query = f"after:{(datetime.now() - timedelta(days=days)).strftime('%Y/%m/%d')}"
            
            response = service.users().messages().list(
                userId='me',
                q=date_query,
                maxResults=1000
            ).execute()
            
            messages = response.get('messages', [])
            
            # 라벨별 통계
            label_stats = {}
            for message in messages:
                msg_details = service.users().messages().get(
                    userId='me', 
                    id=message['id'],
                    format='metadata',
                    metadataHeaders=['From', 'Subject']
                ).execute()
                
                # 라벨 확인 (간단한 분류)
                labels = msg_details.get('labelIds', [])
                if 'UNREAD' in labels:
                    label_stats['unread'] = label_stats.get('unread', 0) + 1
                else:
                    label_stats['read'] = label_stats.get('read', 0) + 1
            
            return {
                'total': len(messages),
                'read': label_stats.get('read', 0),
                'unread': label_stats.get('unread', 0),
                'period_days': days
            }
            
        except HttpError as error:
            st.error(f'이메일 통계 가져오기 중 오류: {error}')
            return None

# 전역 Gmail API 인스턴스
gmail_api = GmailAPI() 