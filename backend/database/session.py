# backend/database/session.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 和 init_db.py 中相同的資料庫 URL
DATABASE_URL = "sqlite:///./ai_medical_assistant.db"

engine = create_engine(
    DATABASE_URL,
    # SQLite 特有參數
    connect_args={"check_same_thread": False}
)

# 建立一個新的 SessionLocal 類別
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)