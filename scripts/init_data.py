# scripts/init_data.py

import sys
import os


# 将项目根目录添加到 sys.path 中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from backend.database.session import SessionLocal
from backend.database.models import MedicalKnowledge

print("脚本已启动，正在初始化数据...")

KNOWLEDGE_DATA = [
    {
        "title": "关于普通感冒的症状",
        "content_text": "普通感冒是一种上呼吸道病毒感染。常见症状包括流鼻涕、鼻塞、喉咙痛、咳嗽、打喷嚏和轻微的身体疼痛或头痛。发烧通常不常见或仅为低热。症状通常在接触病毒后一到三天出现。",
        "source_url": "https://example.com/common-cold"
    },
    {
        "title": "高血压的定义与标准",
        "content_text": "高血压，即动脉血压持续升高。通常定义为收缩压持续等于或高于140毫米汞柱（mmHg），或舒张压持续等于或高于90毫米汞柱。长期高血压是心脏病、中风和肾脏疾病等多种严重健康问题的主要风险因素。",
        "source_url": "https://example.com/hypertension"
    }
]

def main():
    db = SessionLocal()
    try:
        print(f"准备写入 {len(KNOWLEDGE_DATA)} 条知识数据...")
        for data in KNOWLEDGE_DATA:
            # 检查是否已存在相同标题的条目，避免重复插入
            exists = db.query(MedicalKnowledge).filter(MedicalKnowledge.title == data["title"]).first()
            if not exists:
                # 如果不存在，则插入新条目
                db_knowledge = MedicalKnowledge(
                    title=data["title"],
                    content_text=data["content_text"],
                    source_url=data["source_url"]
                )
                # 加入到「臨時病歷夾」(Session)
                db.add(db_knowledge)
                print(f"已添加知识条目: {data['title']}")
            else:
                print(f"知识条目已存在，跳过: {data['title']}")
        
        # 提交所有更改到数据库
        db.commit()
        print("数据初始化完成。")
    finally:
        # 关闭数据库连接
        db.close()
        print("数据库连接已关闭。")
        
if __name__ == "__main__":
    main()