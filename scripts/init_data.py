# scripts/init_data.py

import sys
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# 将项目根目录添加到 sys.path 中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from backend.database.session import SessionLocal
from backend.database.models import MedicalKnowledge

from datasets import load_from_disk
from datetime import datetime

import logging

def clean_data(dataset, src):
    """清洗数据集，去除空问答对，映射字段"""
    # 去除空的问答对
    dataset = [x for x in dataset if x.get('questions') and x.get('answers')]

    # 结构映射成 title/content/source_url
    mapped_data = []
    for x in dataset:
        q = x.get('question') or x.get('questions') or ""
        a = x.get('answer') or x.get('answers') or ""

        # 如果是 list（或 list of list），取最内层元素
        if isinstance(q, list):
            q = q[0] if q and isinstance(q[0], str) else (q[0][0] if q and isinstance(q[0], list) else "")
        if isinstance(a, list):
            a = a[0] if a and isinstance(a[0], str) else (a[0][0] if a and isinstance(a[0], list) else "")

        question = str(q).replace("\n", " ").strip()
        answer = str(a).replace("\n", " ").strip()
        if not question or not answer:
            continue
        mapped_data.append({
            "title": question,
            "content_text": f"问题:{question}\n回答:{answer}",
            "source_url": f"huatuo://{src}",
            "created_at": datetime.now(),
            "source_type": src
        }) 
    return mapped_data


# KNOWLEDGE_DATA = [
#     {
#         "title": "关于普通感冒的症状",
#         "content_text": "普通感冒是一种上呼吸道病毒感染。常见症状包括流鼻涕、鼻塞、喉咙痛、咳嗽、打喷嚏和轻微的身体疼痛或头痛。发烧通常不常见或仅为低热。症状通常在接触病毒后一到三天出现。",
#         "source_url": "https://example.com/common-cold"
#     },
#     {
#         "title": "高血压的定义与标准",
#         "content_text": "高血压，即动脉血压持续升高。通常定义为收缩压持续等于或高于140毫米汞柱（mmHg），或舒张压持续等于或高于90毫米汞柱。长期高血压是心脏病、中风和肾脏疾病等多种严重健康问题的主要风险因素。",
#         "source_url": "https://example.com/hypertension"
#     }
# ]


def main():
    # 加载数据集
    ds_ency = load_from_disk("data/huatuo/huatuo_encyclopedia")
    ds_kg = load_from_disk("data/huatuo/huatuo_knowledge_graph")

    # 清洗空字段/映射字段
    data_ency = clean_data(ds_ency['train'].select(range(2000)), "ency")
    data_kg = clean_data(ds_kg['train'].select(range(2000)), "kg")

    # 合并两个数据集
    KNOWLEDGE_DATA = data_ency + data_kg

    print(f"准备写入 {len(KNOWLEDGE_DATA)} 条知识数据...")

    db = SessionLocal()
    try:
        # 批量插入数据，每1000条提交一次
        batch = []
        for i, data in enumerate(KNOWLEDGE_DATA):
            batch.append(MedicalKnowledge(**data))
            if len(batch) >= 1000:
                db.add_all(batch)
                db.commit()
                print(f"已写入 {i+1} 条数据...")
                batch = []
        if batch:
            db.add_all(batch)
            db.commit()

            # 统计总数
            total = db.query(MedicalKnowledge).count()
            print(f"已写入 {len(KNOWLEDGE_DATA)} 条数据...")
        
        # 提交所有更改到数据库
        db.commit()
        print("数据初始化完成。")
    except Exception as e:
        print(f"数据初始化时出错: {e}")
        db.rollback()
    finally:
        # 关闭数据库连接
        db.close()
        print("数据库连接已关闭。")
        
if __name__ == "__main__":
    main()