# scripts/build_rag_index.py

import sys
import os

# 将项目根目录添加到 sys.path 中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

print(os.getenv("CHROMA_PERSIST"))
print(os.getenv("HUGGINGFACEHUB_HUB_CACHE"))

from backend.database.session import SessionLocal
from backend.database.models import MedicalKnowledge

# 从数据库加载医学知识数据
# 先尝试2000条，后续可调整
def load_knowledge_from_db():
    """从数据库加载医学知识数据"""
    db = SessionLocal()
    records = db.query(MedicalKnowledge).limit(2000).all()
    data = []
    for record in records:
        data.append({
            "title": record.title,
            "content_text": record.content_text,
            "source_url": record.source_url,
            "created_at": record.created_at,
            "source_type": record.source_type
        })
    db.close()
    return data

import re

# 对文本进行 chunking
def chunk_text(text, chunk_size=500, overlap=80):
    """将文本分块，默认每块500字符，重叠80字符"""
    
    # 简单按标点符号分句
    sents = re.split(r'(?<=[。！？\n])', text)
    chunks, current = [], ""
    for sent in sents:
        if len(current) + len(sent) > chunk_size:
            chunks.append(current)
            current = current[-overlap:] + sent  # 保留重叠部分
        else:
            current += sent
    if current:
        chunks.append(current)
    return chunks

from sentence_transformers import SentenceTransformer

# Embedding 生成
def generate_embeddings(texts, model_name="sentence-transformers/all-mpnet-base-v2"):
    """使用 SentenceTransformer 生成文本嵌入"""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    print(embeddings.shape)
    return embeddings

# 构建 Chroma 向量数据库
