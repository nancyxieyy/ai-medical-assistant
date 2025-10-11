# tests/test_redis.py

import redis
import json
import time
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def test_redis_write_and_read():
    """测试 Redis 的写入和读取功能"""
    # 连接到 Redis
    r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

    # 测试连接
    try:
        assert r.ping() is True, "Redis 连接失败，请确保 Redis 服务器正在运行。"
        print("Redis 连接成功！")
    except redis.exceptions.ConnectionError as e:
        assert False, f"Redis 连接失败: {e}"


    # 准备测试数据
    APP_PREFIX = os.getenv("APP_REDIS_PREFIX", "app:")
    case_id = f"{APP_PREFIX}test_case:redis_001"
    expected_transcript = "患者有轻微发烧，建议多喝水。"
    expected_ai_summary_dict = {"主诉": "轻微发烧", "建议": "多喝水"}
    
    case_data = {
        "transcript": expected_transcript,
        "ai_summary": json.dumps(expected_ai_summary_dict)
    }

    expire_time = 60  # 过期时间，单位为秒
    
    # 使用 Pipeline 进行原子性写入
    try:
        with r.pipeline() as pipe:
            pipe.hset(case_id, mapping=case_data)
            pipe.expire(case_id, 60)
            pipe.execute()
        print(f"已将测试数据写入 Redis (Key: {case_id})")
    except redis.exceptions.RedisError as e:
        print(f"写入 Redis 失败: {e}")
        exit(1)

    # 读取数据并验证
    retrieved_data = r.hgetall(case_id)
    print("从 Redis 读取的数据:")
    print(retrieved_data)

    # 确保数据被读取且饱含关键字段
    assert retrieved_data is not None, "未能从 Redis 读取到数据"
    assert "transcript" in retrieved_data, "读取的数据中缺少 'transcript' 字段"
    assert "ai_summary" in retrieved_data, "读取的数据中缺少 'ai_summary' 字段"
    
    # 验证 transcript 内容
    assert retrieved_data["transcript"] == expected_transcript, "transcript 内容不匹配"
    
    # 验证 ai_summary 内容
    actual_ai_summary_dict = json.loads(retrieved_data["ai_summary"])
    assert actual_ai_summary_dict == expected_ai_summary_dict, "ai_summary 内容不匹配"
    
    print("数据内容验证通过！")
    
    # 验证过期时间
    remaining_time_initial = r.ttl(case_id)
    print(f"数据剩余过期时间: {remaining_time_initial} 秒")
    assert 0 < remaining_time_initial <= expire_time, "过期时间不在预期范围内"
    
    # 等待过期时间后再次检查
    print(f"等待 {expire_time + 1} 秒以验证过期...")
    time.sleep(expire_time + 1)
    remaining_time_after_wait = r.ttl(case_id)
    print(f"等待后数据剩余过期时间: {remaining_time_after_wait} 秒")
    assert remaining_time_after_wait == -2, "数据未按预期过期"
    
    print("测试成功！数据读写与过期时间均符合预期。")