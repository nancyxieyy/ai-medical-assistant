# backend/database/init_db.py

from sqlalchemy import create_engine
from .base import Base
from .models import Patient, ConfirmedCase, MedicalKnowledge, Report  # 确保模型已导入

# 使用 SQLite 数据库
DATABASE_URL = "sqlite:///./ai_medical_assistant.db"

engine = create_engine(DATABASE_URL, echo=True)

def init_db():
    """初始化数据库，创建所有表"""
    print("Initializing database...")
    Base.metadata.create_all(bind=engine)
    print("Database initialized.")

if __name__ == "__main__":
    init_db()