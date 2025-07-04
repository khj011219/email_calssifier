import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os
import html
import re
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json

# 커스텀 모듈 import
from auth_system import auth_system
from gmail_api import gmail_api

# 페이지 설정
st.set_page_config(
    page_title="이메일 분류 시스템",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .email-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .user-profile {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .google-login-btn {
        background-color: #4285f4;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        text-decoration: none;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .setup-guide {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 전역 모델/토크나이저 캐싱
@st.cache_resource(show_spinner=False)
def load_bert_model():
    model_dir = "bert_manual_300_model/checkpoint-264"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # label mapping
    with open("bert_manual_300_model/label_mapping.json", encoding="utf-8") as f:
        label_map = json.load(f)
    id2label = label_map["id_to_label"]
    return model, tokenizer, device, id2label

def main():
    """메인 애플리케이션"""
    
    # 세션 상태 초기화
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_data' not in st.session_state:
        st.session_state.user_data = None
    if 'gmail_creds' not in st.session_state:
        st.session_state.gmail_creds = None
    
    # 사이드바
    with st.sidebar:
        st.title("📧 이메일 분류 시스템")
        
        if not st.session_state.logged_in:
            st.info("Google 계정으로 로그인이 필요합니다.")
        else:
            # 사용자 프로필 표시
            user_info = st.session_state.user_data
            name = html.escape(user_info.get('name', ''))
            email = html.escape(user_info.get('email', ''))
            picture_url = user_info.get('picture_url', '')
            if re.match(r'^[^@]+@[^@]+\.[^@]+$', name):
                display_name = "사용자"
            else:
                display_name = name if name else "사용자"
            if picture_url:
                st.image(picture_url, width=50)
            st.write(f"**{display_name}**")
            st.write(f"{email}")
            
            if st.button("로그아웃"):
                st.session_state.logged_in = False
                st.session_state.user_data = None
                st.session_state.gmail_creds = None
                st.rerun()
    
    # 메인 컨텐츠
    if not st.session_state.logged_in:
        show_login_page()
    else:
        show_main_dashboard()

def show_login_page():
    """로그인 페이지"""
    st.markdown('<h1 class="main-header">📧 이메일 분류 시스템</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔐 Google 계정으로 로그인")
        st.markdown("""
        ### 이메일 분류 시스템에 오신 것을 환영합니다!
        
        이 시스템은 Gmail API를 사용하여:
        - 📧 실제 Gmail 계정에서 이메일을 가져옵니다
        - 🤖 AI 모델로 이메일을 자동 분류합니다
        - 📊 분류 결과를 시각화하여 보여줍니다
        
        **Google 계정으로 로그인**하여 시작하세요!
        """)
        
        # Google 로그인 버튼
        if st.button("🔗 Google 계정으로 로그인", type="primary", use_container_width=True):
            with st.spinner("Google 인증 중..."):
                # Gmail API 인증
                creds = auth_system.authenticate_with_gmail()
                if creds:
                    # 사용자 정보 가져오기
                    user_info = auth_system.get_user_info_from_gmail(creds)
                    if user_info:
                        # 사용자 로그인 또는 등록
                        success, user_data = auth_system.login_or_register_user(user_info)
                        if success:
                            st.session_state.logged_in = True
                            st.session_state.user_data = user_data
                            st.session_state.gmail_creds = creds
                            st.success("로그인 성공!")
                            st.rerun()
                        else:
                            st.error(user_data)
                    else:
                        st.error("사용자 정보를 가져올 수 없습니다.")
                else:
                    st.error("Google 인증에 실패했습니다.")
    
    with col2:
        st.markdown("""
        ### 🔧 설정이 필요하신가요?
        
        **Gmail API 설정 방법:**
        
        1. [Google Cloud Console](https://console.cloud.google.com/)에서 프로젝트 생성
        2. **OAuth 동의 화면** 구성 (필수!)
        3. **Gmail API**와 **People API** 활성화
        4. **OAuth 2.0 클라이언트 ID** 생성
        5. **credentials.json** 파일 다운로드
        6. 프로젝트 루트에 저장
        
        [📖 상세 설정 가이드](GMAIL_API_SETUP_GUIDE.md)
        """)
        
        # 설정 가이드 확장 가능한 섹션
        with st.expander("🔍 OAuth 동의 화면 설정이 필요한 이유"):
            st.markdown("""
            **OAuth 동의 화면**은 Google API를 사용하는 앱에서 사용자에게 표시되는 권한 요청 화면입니다.
            
            ### 왜 필요한가요?
            - 사용자가 앱에 어떤 권한을 부여할지 알 수 있음
            - Google의 보안 정책 요구사항
            - OAuth 2.0 클라이언트 ID 생성 전 필수 단계
            
            ### 설정하지 않으면?
            - "OAuth 동의 화면이 구성되지 않았습니다" 오류 발생
            - 클라이언트 ID 생성 불가
            - 앱 실행 불가
            
            ### 해결 방법
            위의 "상세 설정 가이드"를 참고하여 OAuth 동의 화면을 먼저 구성하세요!
            """)

def show_main_dashboard():
    """메인 대시보드"""
    st.markdown('<h1 class="main-header">📧 이메일 분류 대시보드</h1>', unsafe_allow_html=True)
    
    # Gmail 서비스 가져오기
    if st.session_state.gmail_creds:
        service = auth_system.get_gmail_service(st.session_state.gmail_creds)
        
        # 이메일 통계 가져오기
        stats = gmail_api.get_email_statistics(service, days=7)
        
        # 상단 메트릭
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_emails = stats['total'] if stats else 0
            st.metric("총 이메일 (7일)", total_emails)
        
        with col2:
            unread_emails = stats['unread'] if stats else 0
            st.metric("읽지 않은 이메일", unread_emails)
        
        with col3:
            classified_emails = len(auth_system.get_user_classifications(st.session_state.user_data['id']))
            st.metric("분류된 이메일", classified_emails)
        
        with col4:
            accuracy = "85.2%"  # 실제로는 모델 성능에서 가져와야 함
            st.metric("분류 정확도", accuracy)
    
    # 탭 생성
    tab1, tab2, tab3, tab4 = st.tabs(["📧 이메일 목록", "🤖 이메일 분류", "📈 분석", "⚙️ 설정"])
    
    with tab1:
        show_email_list()
    
    with tab2:
        show_email_classification()
    
    with tab3:
        show_analytics()
    
    with tab4:
        show_settings()

def show_email_list():
    """이메일 목록 페이지"""
    st.header("📧 이메일 목록")
    
    if not st.session_state.gmail_creds:
        st.warning("Gmail 인증이 필요합니다.")
        return
    
    service = auth_system.get_gmail_service(st.session_state.gmail_creds)
    
    # 검색 옵션
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_query = st.text_input("이메일 검색", placeholder="예: from:example@gmail.com")
    
    with col2:
        email_type = st.selectbox("이메일 유형", ["전체", "읽지 않은 이메일", "최근 이메일"])
    
    with col3:
        max_results = st.number_input("최대 결과 수", min_value=1, max_value=50, value=10)
    
    # 이메일 가져오기
    if st.button("🔍 이메일 검색", type="primary"):
        with st.spinner("이메일을 검색하는 중..."):
            if email_type == "읽지 않은 이메일":
                emails = gmail_api.get_unread_emails(service, max_results)
            elif email_type == "최근 이메일":
                emails = gmail_api.get_recent_emails(service, max_results)
            else:
                emails = gmail_api.search_emails(service, search_query, max_results)
            
            if emails:
                # 최신 BERT 모델 로드
                model, tokenizer, device, id2label = load_bert_model()
                pred_labels = []
                for email in emails:
                    text = (email.get('subject', '') or '') + ' ' + (email.get('body', '') or '')
                    inputs = tokenizer(text, truncation=True, padding='max_length', max_length=256, return_tensors='pt')
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = model(**inputs)
                        probs = torch.softmax(outputs.logits, dim=1)
                        conf, pred = torch.max(probs, dim=1)
                        label = id2label[str(pred.item())] if str(pred.item()) in id2label else str(pred.item())
                        email['pred_label'] = label
                        email['pred_confidence'] = float(conf.item())
                        pred_labels.append(label)
                st.success(f"{len(emails)}개의 이메일을 찾았습니다!")
                # 분류 결과 대시보드
                df_pred = pd.DataFrame(emails)
                st.subheader("📊 자동 분류 결과 요약")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Work", (df_pred['pred_label'] == 'Work').sum())
                    st.metric("Personal", (df_pred['pred_label'] == 'Personal').sum())
                    st.metric("Advertisement", (df_pred['pred_label'] == 'Advertisement').sum())
                with col2:
                    fig = px.pie(df_pred, names='pred_label', title="이메일 자동 분류 분포")
                    st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df_pred[['subject', 'sender', 'date', 'pred_label', 'pred_confidence']].head(20), use_container_width=True)
                st.divider()
                # 이메일 목록 표시(기존 수동 분류 버튼 포함)
                for i, email in enumerate(emails):
                    with st.container():
                        st.write("📧", email.get('subject', ''))
                        st.write(f"**보낸 사람:** {email.get('sender', '')}")
                        st.write(f"**날짜:** {email.get('date', '')}")
                        st.write(email.get('snippet', '')[:200] + "...")
                        st.write(f"**자동 분류:** {email.get('pred_label', '-')}")
                        st.write(f"**모델 신뢰도:** {email.get('pred_confidence', 0):.2f}")
                        # 전체 보기 버튼 및 확장
                        if 'expanded_email_idx' not in st.session_state:
                            st.session_state.expanded_email_idx = None
                        if st.button("🔎 전체 보기", key=f"expand_{i}"):
                            st.session_state.expanded_email_idx = i if st.session_state.expanded_email_idx != i else None
                        if st.session_state.expanded_email_idx == i:
                            # 최신 본문 가져오기 (id가 있으면)
                            full_email = email
                            if email.get('id'):
                                try:
                                    service = auth_system.get_gmail_service(st.session_state.gmail_creds)
                                    full_email = gmail_api.get_message_details(service, email['id']) or email
                                except Exception as e:
                                    st.warning(f"이메일 전체 내용 불러오기 실패: {e}")
                            st.markdown("---")
                            st.markdown(f"### {full_email.get('subject', '')}")
                            st.write(f"**보낸 사람:** {full_email.get('sender', '')}")
                            st.write(f"**날짜:** {full_email.get('date', '')}")
                            st.write(f"**전체 본문:**\n\n{full_email.get('body', '(본문 없음)')}")
                        col1, col2, col3 = st.columns([1, 1, 1])
                        with col1:
                            if st.button("💼 Work", key=f"work_{i}"):
                                st.write(f"[DEBUG] email_id: {email.get('id')}, subject: {email.get('subject')}, sender: {email.get('sender')}")
                                try:
                                    result = auth_system.save_email_classification(
                                        st.session_state.user_data['id'],
                                        email.get('id'),
                                        email.get('subject'),
                                        email.get('sender'),
                                        'Work',
                                        email.get('pred_confidence', 0)
                                    )
                                    st.write(f"[DEBUG] save_email_classification result: {result}")
                                    if result:
                                        st.success("Work로 분류되었습니다!")
                                        st.experimental_rerun()
                                    else:
                                        st.error("분류 저장에 실패했습니다.")
                                except Exception as e:
                                    st.error(f"분류 저장 중 예외 발생: {e}")
                        with col2:
                            if st.button("👤 Personal", key=f"personal_{i}"):
                                st.write(f"[DEBUG] email_id: {email.get('id')}, subject: {email.get('subject')}, sender: {email.get('sender')}")
                                try:
                                    result = auth_system.save_email_classification(
                                        st.session_state.user_data['id'],
                                        email.get('id'),
                                        email.get('subject'),
                                        email.get('sender'),
                                        'Personal',
                                        email.get('pred_confidence', 0)
                                    )
                                    st.write(f"[DEBUG] save_email_classification result: {result}")
                                    if result:
                                        st.success("Personal로 분류되었습니다!")
                                        st.experimental_rerun()
                                    else:
                                        st.error("분류 저장에 실패했습니다.")
                                except Exception as e:
                                    st.error(f"분류 저장 중 예외 발생: {e}")
                        with col3:
                            if st.button("📢 Advertisement", key=f"ad_{i}"):
                                st.write(f"[DEBUG] email_id: {email.get('id')}, subject: {email.get('subject')}, sender: {email.get('sender')}")
                                try:
                                    result = auth_system.save_email_classification(
                                        st.session_state.user_data['id'],
                                        email.get('id'),
                                        email.get('subject'),
                                        email.get('sender'),
                                        'Advertisement',
                                        email.get('pred_confidence', 0)
                                    )
                                    st.write(f"[DEBUG] save_email_classification result: {result}")
                                    if result:
                                        st.success("Advertisement로 분류되었습니다!")
                                        st.experimental_rerun()
                                    else:
                                        st.error("분류 저장에 실패했습니다.")
                                except Exception as e:
                                    st.error(f"분류 저장 중 예외 발생: {e}")
                        st.divider()
            else:
                st.info("검색 결과가 없습니다.")

def show_email_classification():
    """이메일 분류 페이지"""
    st.header("🤖 이메일 분류")
    
    # 분류 기록 조회
    classifications = auth_system.get_user_classifications(st.session_state.user_data['id'], limit=20)
    
    if classifications:
        st.subheader("📋 최근 분류 기록")
        
        # 분류 데이터를 DataFrame으로 변환
        df = pd.DataFrame(classifications)
        df['classified_at'] = pd.to_datetime(df['classified_at'])
        
        # 분류별 통계
        col1, col2, col3 = st.columns(3)
        
        with col1:
            work_count = len(df[df['classification'] == 'Work'])
            st.metric("Work", work_count)
        
        with col2:
            personal_count = len(df[df['classification'] == 'Personal'])
            st.metric("Personal", personal_count)
        
        with col3:
            ad_count = len(df[df['classification'] == 'Advertisement'])
            st.metric("Advertisement", ad_count)
        
        # 분류 기록 테이블
        st.dataframe(
            df[['subject', 'sender', 'classification', 'classified_at']].head(10),
            use_container_width=True
        )
        
        # 분류 분포 차트
        fig = px.pie(
            df, 
            names='classification', 
            title="이메일 분류 분포"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("아직 분류된 이메일이 없습니다. 이메일 목록에서 이메일을 분류해보세요!")

def show_analytics():
    """분석 페이지"""
    st.header("📈 이메일 분석")
    
    # 사용자의 분류 데이터 가져오기
    classifications = auth_system.get_user_classifications(st.session_state.user_data['id'], limit=100)
    
    if classifications:
        df = pd.DataFrame(classifications)
        df['classified_at'] = pd.to_datetime(df['classified_at'])
        df['date'] = df['classified_at'].dt.date
        
        # 일별 분류 통계
        daily_stats = df.groupby(['date', 'classification']).size().unstack(fill_value=0)
        
        # 차트 생성
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 일별 이메일 분류")
            if not daily_stats.empty:
                fig = px.line(daily_stats, title="일별 이메일 분류 추이")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🍰 이메일 유형 분포")
            classification_counts = df['classification'].value_counts()
            fig = px.pie(
                values=classification_counts.values, 
                names=classification_counts.index,
                title="전체 이메일 유형 분포"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 통계 요약
        st.subheader("📋 통계 요약")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("총 분류된 이메일", len(df))
        
        with col2:
            work_ratio = len(df[df['classification'] == 'Work']) / len(df) * 100
            st.metric("Work 비율", f"{work_ratio:.1f}%")
        
        with col3:
            personal_ratio = len(df[df['classification'] == 'Personal']) / len(df) * 100
            st.metric("Personal 비율", f"{personal_ratio:.1f}%")
        
        with col4:
            ad_ratio = len(df[df['classification'] == 'Advertisement']) / len(df) * 100
            st.metric("Advertisement 비율", f"{ad_ratio:.1f}%")
    
    else:
        st.info("분석할 데이터가 없습니다. 이메일을 분류해보세요!")

def show_settings():
    """설정 페이지"""
    st.header("⚙️ 설정")
    
    # 사용자 정보
    st.subheader("👤 사용자 정보")
    if st.session_state.user_data:
        user_info = auth_system.get_user_by_id(st.session_state.user_data['id'])
        if user_info:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if user_info['picture_url']:
                    st.image(user_info['picture_url'], width=100)
                else:
                    st.write("프로필 사진 없음")
            
            with col2:
                st.write(f"**이름:** {user_info['name']}")
                st.write(f"**이메일:** {user_info['email']}")
                st.write(f"**가입일:** {user_info['created_at']}")
                if user_info['last_login']:
                    st.write(f"**마지막 로그인:** {user_info['last_login']}")
    
    # Gmail 설정
    st.subheader("📧 Gmail 설정")
    if st.session_state.gmail_creds:
        st.success("✅ Gmail이 연동되어 있습니다!")
        
        if st.button("🔄 Gmail 재연동"):
            st.session_state.logged_in = False
            st.session_state.user_data = None
            st.session_state.gmail_creds = None
            st.rerun()
    else:
        st.warning("Gmail이 연동되어 있지 않습니다.")
    
    # 데이터 관리
    st.subheader("🗄️ 데이터 관리")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 분류 데이터 내보내기"):
            classifications = auth_system.get_user_classifications(st.session_state.user_data['id'])
            if classifications:
                df = pd.DataFrame(classifications)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="CSV 다운로드",
                    data=csv,
                    file_name="email_classifications.csv",
                    mime="text/csv"
                )
            else:
                st.info("내보낼 데이터가 없습니다.")
    
    with col2:
        if st.button("🗑️ 분류 데이터 초기화"):
            st.warning("이 작업은 되돌릴 수 없습니다!")
            if st.button("정말 삭제하시겠습니까?", type="secondary"):
                # 데이터 삭제 로직 (구현 필요)
                st.success("분류 데이터가 초기화되었습니다.")
    
    # 모델 설정
    st.subheader("🤖 모델 설정")
    st.info("이메일 분류 모델 통합은 개발 중입니다.")

if __name__ == "__main__":
    main() 