# tests/test_db.py

from datetime import date, datetime
from backend.database.session import SessionLocal
from backend.database.models import Patient, ConfirmedCase

def test_database_write_and_read():
    """测试数据库的写入和读取功能"""
    db = SessionLocal()

    try:
        new_patient = Patient(
            name="张三",
            birth_date=date(1985, 5, 20),
            gender="男",
            contact_info="12345678910",
        )
        db.add(new_patient)
        db.commit()
        db.refresh(new_patient)

        print(f"新患者ID: {new_patient.patient_id}")

        new_case = ConfirmedCase(
            patient_id=new_patient.patient_id,
            doctor_id=101,
            consultation_date=datetime.now(),
            ai_summary={"summary": "患者有轻微发烧，建议多喝水。"},
            doctor_notes="患者症状较轻，建议居家休息。",
        )

        db.add(new_case)
        db.commit()
        db.refresh(new_case)

        print(f"新病例ID: {new_case.case_id}")
        print("-" * 20)
        print("病例已添加到数据库:")

        # 断言1: 检查资料库 Patient 表是否大于 0
        patient_count = db.query(Patient).count()
        assert patient_count > 0
        
        # 断言2: 检查新添加的患者是否存在
        quired_patient = db.query(Patient).filter(Patient.name == "张三").first()
        assert quired_patient is not None
        assert quired_patient.name == "张三"
        
        # 断言3: 检查新添加的病例是否存在
        assert len(quired_patient.cases) == 1
        
        print("断言均通过，数据库写入和读取功能正常。")

    finally:
        db.close()
        print("-" * 20)
        print("数据库测试完成。")
