# backend/main.py
import os, sys
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 让 Python 找到 medical_agent/src
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from medical_agent.src.agent.graph import graph
from backend.database.session import SessionLocal
from backend.database.init_db import init_db
from backend.database import crud

init_db()
app = FastAPI(title="Medical Agent (Local)")

class InferenceRequest(BaseModel):
    texts: List[str]
    mode: Optional[str] = "realtime_advice"
    transcript: Optional[str] = None
    context: Optional[str] = None
    session_id: Optional[str] = None

@app.post("/realtime")
def realtime(req: InferenceRequest):
    try:
        payload = req.model_dump()
        out = graph.invoke(payload, entry_point="embedding_node")
        return {"ok": True, "data": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summary")
def summary(req: InferenceRequest):
    try:
        payload = req.model_dump()
        out = graph.invoke(payload, entry_point="summary_start")
        # 生成草稿：先暂存，医生确认后再入 ConfirmedCase
        db = SessionLocal()
        try:
            draft_id = crud.create_draft(
                db,
                session_id=payload.get("session_id"),
                mode=payload.get("mode", "final_report"),
                transcript=payload.get("transcript") or (payload.get("texts") or [""])[0],
                context=out.get("context") if isinstance(out, dict) else None,
                output=(out.get("llm_output") if isinstance(out, dict) else str(out)),
            )
        finally:
            db.close()
        return {"ok": True, "draft_id": draft_id, "data": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from pydantic import BaseModel

class IdReq(BaseModel):
    report_id: int

@app.post("/report/comfirm")
def confirm(req: IdReq):
    db = SessionLocal()
    try:
        ok = crud.confirm_report(db, report_id=req.report_id)
        if not ok:
            raise HTTPException(400, "草稿不存在或状态不对")
        return {"ok": True}
    finally:
        db.close()

@app.post("/report/discard")
def discard(req: IdReq):
    db = SessionLocal()
    try:
        ok = crud.discard_draft(db, report_id=req.report_id)
        if not ok:
            raise HTTPException(400, "草稿不存在或已处理")
        return {"ok": True}
    finally:
        db.close()

class MoveReq(BaseModel):
    report_id: int
    patient_id: int
    doctor_id: int

@app.post("/report/move_to_case")
def move_to_case(req: MoveReq):
    db = SessionLocal()
    try:
        case_id = crud.move_report_to_confirmed_case(
            db,
            report_id=req.report_id,
            patient_id=req.patient_id,
            doctor_id=req.doctor_id,
        )
        if case_id is None:
            raise HTTPException(400, "报告不存在或未确认(final)")
        return {"ok": True, "case_id": case_id}
    finally:
        db.close()