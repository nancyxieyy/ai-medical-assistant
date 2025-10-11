# tests/tets_multi_db.py

import os
import redis
import json
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# 加载环境变量
load_dotenv(dotenv_path='backend/.env')

# 连接到 SQLite 数据库

from backend.database.session import SessionLocal
from backend.database.models import Patient, ConfirmedCase

def test_app_sqlite_connection():
    """测试应用程序是否能连接到 SQLite 数据库"""

    print("--- 测试应用程序连接到 SQLite 数据库 ---")
    db = SessionLocal()
    try:
        patient_count = db.query(Patient).count()
        assert patient_count >= 0
        print(f"成功连接到 SQLite 数据库，患者数量: {patient_count}")
    finally:
        db.close()
    


# 连接到 Redis 数据库
def test_app_redis_connection():
    """测试应用程序是否能连接到 Redis 数据库"""
    print("--- 测试应用程序连接到 Redis 数据库 ---")
    
    redis_url = os.getenv("REDIS_URL")
    app_prefix = os.getenv("APP_REDIS_PREFIX", "app:")
    
    r = redis.from_url(redis_url, decode_responses=True)
    
    # 断言连接成功
    assert r.ping() is True
    
    test_key = f"{app_prefix}multi_db_test"
    test_value = "success"
    
    # 测试写入和读取
    r.set(test_key, test_value, ex=10)
    retrieved_value = r.get(test_key)
    
    # 断言读写成功
    assert retrieved_value == test_value
    print("成功连接到 Redis 数据库，读写测试通过")
        

# 验证 LangGraph 的 PostgreSQL 资料库
def  test_langgraph_postgres_connection():
    """测试是否能连接到 LangGraph 的 PostgreSQL 资料库"""
    print("--- 测试连接到 LangGraph 的 PostgreSQL 资料库 ---")
    postgres_url = os.getenv("POSTGRES_URL")
    
    # 断言环境变量存在
    assert postgres_url is not None, "POSTGRES_URL 环境变量未设置"
    
    print(f"DEBUG: 正在嘗試連接到 PostgreSQL，URL 為: '{postgres_url}'")
    
    pg_engine = create_engine(postgres_url)
    
    try:
        with pg_engine.connect() as connection:
            result = connection.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))
            all_tables = [row[0] for row in result]
            
            print(f"在 LangGraph 的 PostgreSQL 中找到的资料表: {all_tables}")
            
            # 斷言：檢查我們之前在 TablePlus 中看到的 `checkpoints` 表是否存在
            assert "checkpoints" in all_tables
            print("PostgreSQL 连接与查询成功！找到了 LangGraph 的资料表。")
    except Exception as e:
        assert False, f"连接到 PostgreSQL 失敗: {e}"