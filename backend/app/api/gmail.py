from fastapi import APIRouter, Query, Body
from gmail_api import gmail_api
from models.predict_email_category import predict_category
import os, json, csv
from datetime import datetime

router = APIRouter()

@router.get("/emails")
def list_emails(q: str = Query("", description="검색 쿼리"), max_results: int = 10):
    # TODO: Gmail API 연동 및 이메일 목록 반환
    return {"emails": [], "query": q, "max_results": max_results}

@router.get("/emails/{email_id}")
def email_detail(email_id: str):
    # TODO: Gmail API 연동 및 이메일 상세 반환
    return {"email_id": email_id, "detail": "(이메일 상세 내용)"}

@router.post("/classify_all")
def classify_all_emails(
    max_results: int = Query(10, description="가져올 메일 개수"),
    unread: bool = Query(False, description="읽지 않은 메일만"),
    query: str = Query("", description="검색어(제목, 본문, 보낸사람 등)")
):
    # 1. Gmail 인증 및 서비스 객체 생성
    creds = gmail_api.authenticate_user()
    if creds is None:
        return {"error": "Gmail 인증 실패"}
    service = gmail_api.get_gmail_service(creds)
    if service is None:
        return {"error": "Gmail 서비스 생성 실패"}

    # 2. 쿼리 조합
    gmail_query = query.strip()
    if unread:
        gmail_query = ("is:unread " + gmail_query).strip()

    # 3. 메일 가져오기
    emails = gmail_api.get_recent_emails(service, max_results=max_results, query=gmail_query)

    # 4. 분류 및 카테고리별 리스트업
    categories = {"Work": [], "Personal": [], "Ad": []}
    for email in emails:
        label, confidence = predict_category(email["subject"], email["body"])
        email_info = {
            "subject": email["subject"],
            "body": email["body"],
            "sender": email.get("sender", ""),
            "date": email.get("date", ""),
            "confidence": confidence
        }
        if label in categories:
            categories[label].append(email_info)
        else:
            categories["Personal"].append(email_info)  # Unknown은 Personal로
    return categories

@router.post("/save_relabels")
def save_relabels(data: dict):
    relabels = data.get("relabels", [])
    if not relabels:
        return {"status": "no_data"}
    # 저장 폴더 및 파일명
    save_dir = "relabels"
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "relabels.csv")
    is_new = not os.path.exists(csv_path)
    # 저장할 필드명
    fieldnames = ["subject", "body", "sender", "date", "old_category", "new_category"]
    # 파일에 append
    with open(csv_path, "a", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        for row in relabels:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return {"status": "ok", "saved": len(relabels), "file": "relabels.csv"} 