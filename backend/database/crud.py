# backend/database/crud.py

from sqlalchemy.orm import Session
from .models import Report, ConfirmedCase
from datetime import datetime

# 草稿报告
def create_draft(db: Session, *, session_id: str | None, mode: str,
                transcript: str | None, context: str | None, output: str) -> int:
    obj = Report(
        session_id=session_id,
        mode=mode,
        transcript=transcript,
        context=context,
        output=output,
        status="draft",
    )
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj.id

# 最终确认报告
def confirm_report(db: Session, *, report_id: int) -> bool:
    obj = db.get(Report, report_id)
    if not obj or obj.status != "draft":
        return False
    obj.status = "final"
    db.commit()
    return True

# 是否确认草稿
def discard_draft(db: Session, *, report_id: int) -> bool:
    obj = db.get(Report, report_id)
    if not obj or obj.status != "draft":
        return False
    db.delete(obj)
    db.commit()
    return True

def get_drafts_by_session(db: Session, *, session_id: str):
    return (
        db.query(Report)
        .filter(Report.session_id == session_id, Report.status == "draft")
        .order_by(Report.created_at.desc())
        .all()
    )

def move_report_to_confirmed_case(db: Session, *, report_id: int,
                                patient_id: int, doctor_id: int) -> int | None:
    """
    将 'final' 状态的报告写入 ConfirmedCase，返回新 case_id。
    """
    rpt = db.get(Report, report_id)
    if not rpt or rpt.status != "final":
        return None
    case = ConfirmedCase(
        patient_id=patient_id,
        doctor_id=doctor_id,
        consultation_date=rpt.created_at or datetime.utcnow(),
        ai_summary={"mode": rpt.mode, "output": rpt.output},  # 也可在此做结构化解析
        doctor_notes=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(case)
    db.commit()
    db.refresh(case)
    return case.case_id
