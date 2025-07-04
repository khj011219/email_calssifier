import sqlite3
import bcrypt
import os
import json
from datetime import datetime, timedelta
import streamlit as st
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

class AuthSystem:
    def __init__(self, db_path="users.db"):
        """인증 시스템 초기화"""
        self.db_path = db_path
        self.SCOPES = [
            'openid',
            'https://www.googleapis.com/auth/gmail.readonly',
            'https://www.googleapis.com/auth/userinfo.email',
            'https://www.googleapis.com/auth/userinfo.profile'
        ]
        self.CREDENTIALS_FILE = 'credentials.json'
        self.TOKEN_FILE = 'token.json'
        self.init_database()
    
    def init_database(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 사용자 테이블 생성 (Gmail 기반)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gmail_id TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                name TEXT,
                picture_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # 이메일 분류 데이터 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS email_classifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                email_id TEXT,
                subject TEXT,
                sender TEXT,
                classification TEXT,
                confidence REAL,
                classified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def authenticate_with_gmail(self):
        """Gmail OAuth를 통한 사용자 인증"""
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
                if not os.path.exists(self.CREDENTIALS_FILE):
                    st.error("Gmail API 설정 파일(credentials.json)이 필요합니다.")
                    return None
                
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.CREDENTIALS_FILE, self.SCOPES)
                    creds = flow.run_local_server(port=0)
                except Exception as e:
                    st.error(f"Gmail 인증 중 오류: {e}")
                    return None
            
            # 토큰을 파일에 저장
            with open(self.TOKEN_FILE, 'w') as token:
                token.write(creds.to_json())
        
        return creds
    
    def get_user_info_from_gmail(self, creds):
        """Gmail API를 통해 사용자 정보 가져오기"""
        try:
            # Gmail API 서비스 생성
            service = build('gmail', 'v1', credentials=creds)
            
            # 사용자 프로필 정보 가져오기
            profile = service.users().getProfile(userId='me').execute()
            
            # Google People API로 추가 정보 가져오기 (선택사항)
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
            st.error(f'사용자 정보 가져오기 중 오류: {error}')
            return None
    
    def login_or_register_user(self, user_info):
        """사용자 로그인 또는 등록"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 기존 사용자 확인
            cursor.execute("SELECT id, name, email, picture_url FROM users WHERE gmail_id = ? OR email = ?", 
                         (user_info['gmail_id'], user_info['email']))
            existing_user = cursor.fetchone()
            
            if existing_user:
                # 기존 사용자 로그인
                user_id, name, email, picture_url = existing_user
                
                # 정보 업데이트 (이름이나 프로필 사진이 변경되었을 수 있음)
                cursor.execute(
                    "UPDATE users SET name = ?, picture_url = ?, last_login = CURRENT_TIMESTAMP WHERE id = ?",
                    (user_info['name'], user_info['picture_url'], user_id)
                )
                
                user_data = {
                    'id': user_id,
                    'gmail_id': user_info['gmail_id'],
                    'name': user_info['name'],
                    'email': email,
                    'picture_url': user_info['picture_url']
                }
            else:
                # 새 사용자 등록
                cursor.execute(
                    "INSERT INTO users (gmail_id, email, name, picture_url) VALUES (?, ?, ?, ?)",
                    (user_info['gmail_id'], user_info['email'], user_info['name'], user_info['picture_url'])
                )
                user_id = cursor.lastrowid
                
                user_data = {
                    'id': user_id,
                    'gmail_id': user_info['gmail_id'],
                    'name': user_info['name'],
                    'email': user_info['email'],
                    'picture_url': user_info['picture_url']
                }
            
            conn.commit()
            conn.close()
            return True, user_data
            
        except Exception as e:
            return False, f"사용자 로그인/등록 중 오류: {str(e)}"
    
    def get_gmail_service(self, creds):
        """Gmail 서비스 객체 생성"""
        try:
            service = build('gmail', 'v1', credentials=creds)
            return service
        except HttpError as error:
            st.error(f'Gmail 서비스 생성 중 오류: {error}')
            return None
    
    def save_email_classification(self, user_id, email_id, subject, sender, classification, confidence=0.0):
        """이메일 분류 결과 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO email_classifications (user_id, email_id, subject, sender, classification, confidence) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, email_id, subject, sender, classification, confidence)
            )
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            st.error(f"분류 결과 저장 중 오류: {e}")
            return False
    
    def get_user_classifications(self, user_id, limit=50):
        """사용자의 이메일 분류 기록 조회"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT email_id, subject, sender, classification, confidence, classified_at FROM email_classifications WHERE user_id = ? ORDER BY classified_at DESC LIMIT ?",
                (user_id, limit)
            )
            
            results = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'email_id': row[0],
                    'subject': row[1],
                    'sender': row[2],
                    'classification': row[3],
                    'confidence': row[4],
                    'classified_at': row[5]
                }
                for row in results
            ]
            
        except Exception as e:
            st.error(f"분류 기록 조회 중 오류: {e}")
            return []
    
    def get_user_by_id(self, user_id):
        """사용자 정보 조회"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT id, gmail_id, name, email, picture_url, created_at, last_login FROM users WHERE id = ?", (user_id,))
            user = cursor.fetchone()
            conn.close()
            
            if user:
                return {
                    'id': user[0],
                    'gmail_id': user[1],
                    'name': user[2],
                    'email': user[3],
                    'picture_url': user[4],
                    'created_at': user[5],
                    'last_login': user[6]
                }
            return None
            
        except Exception as e:
            return None

# 전역 인증 시스템 인스턴스
auth_system = AuthSystem() 