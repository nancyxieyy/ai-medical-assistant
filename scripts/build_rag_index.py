# scripts/build_rag_index.py

import sys
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# 将项目根目录添加到 sys.path 中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

print(os.getenv("CHROMA_PERSIST"))
print(os.getenv("HUGGINGFACE_HUB_CACHE"))

from backend.database.session import SessionLocal
from backend.database.models import MedicalKnowledge
import random

# 从数据库加载医学知识数据
# 先尝试2000条，后续可调整
def load_knowledge_from_db():
    """从数据库加载医学知识数据"""
    db = SessionLocal()
    try:
        records = db.query(MedicalKnowledge).limit(4000).all()
        data = []
        if not records:
            print("数据库返回空集，请检查数据库是否为空")
            return []
        else:
            for record in records:
                data.append({
                    "title": record.title,
                    "content_text": record.content_text,
                    "source_url": record.source_url,
                    "created_at": record.created_at,
                    "source_type": record.source_type
                })
    finally:
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
            chunks.append(current.strip())
            current = current[-overlap:].strip() + sent  # 保留重叠部分
        else:
            current += sent
    if current:
        chunks.append(current.strip())
    # 丢掉太短的块
    return [(i, c) for i, c in enumerate(chunks) if len(c) > 10]

from backend.nodes.embedding_node import JinaEmbeddingNode
import chromadb
from tqdm import tqdm
import gc

import hashlib
def make_id(title, i, source_type):
    """为每个chunk生成唯一id"""
    h = hashlib.md5(f"{title}_{source_type}".encode("utf-8")).hexdigest()[:8]
    return f"{source_type}-{h}-{i}"


if __name__ == "__main__": 
    # 初始化 Node 和 Chroma
    embedder = JinaEmbeddingNode()
    chroma_dir = os.getenv("CHROMA_PERSIST", "rag_store")
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection("medical_knowledge")

    # 加载数据
    data = load_knowledge_from_db()
    if not data:
        print("数据库返回空集，请检查数据或 offset")
        sys.exit(0)
    print(f"已載入 {len(data)} 条医学知识，写入目录：{chroma_dir}")

    # 逐条构建向量
    batch_size = 100
    buffer = []
    for row in tqdm(data, desc="Processing documents"):
        chunks = chunk_text(row["content_text"])
        if not chunks:
            continue

        texts = [c for _, c in chunks]

        embeddings = []
        for text_chunk in texts:
            res = embedder.run({"texts": [text_chunk]})
            # 检查返回结果是否有效
            if res and res.get("embeddings") and res["embeddings"]:
                embeddings.extend(res["embeddings"])
            else:
                embeddings.append(None) # 添加一個佔位符，保持长度一致

        for i, (chunk, emb) in enumerate(zip(texts, embeddings)):
            # 跳过嵌入失敗或为空的块
            if not emb or len(emb) < 100:
                continue
            buffer.append({
                "id": make_id(row["title"], i, row["source_type"]),
                "doc": chunk,
                "emb": emb,
                "meta": {
                    "title": row["title"],
                    "source_type": row["source_type"],
                    "source_url": row["source_url"],
                    "chunk_index": i
                }
            })

        # 写入 ChromaDB
        if len(buffer) >= batch_size:
            collection.add(
                ids=[b["id"] for b in buffer],
                documents=[b["doc"] for b in buffer],
                embeddings=[b["emb"] for b in buffer],
                metadatas=[b["meta"] for b in buffer]
            )
            print(f"\n写入 {len(buffer)} 条 chunk（当前总计: {collection.count()}）")
            buffer.clear()

        # 在处理完一篇大文章的所有 chunks 后，呼叫垃圾回收
        gc.collect()

    # 处理剩余的 buffer
    if buffer:
        collection.add(
            ids=[b["id"] for b in buffer],
            documents=[b["doc"] for b in buffer],
            embeddings=[b["emb"] for b in buffer],
            metadatas=[b["meta"] for b in buffer],
        )
        print(f"\n最后一批写入 {len(buffer)} 条 chunk（总计: {collection.count()}）")

    # 测试查询
    query = "发烧两天 咳嗽 是否需要用抗生素"
    q_vec = embedder.run({"texts": [query]})["embeddings"]
    res = collection.query(query_embeddings=q_vec, n_results=3)
    for m, d in zip(res["metadatas"][0], res["documents"][0]):
        print(f"[{m['source_type']}] {m['title']} → {d[:80]}...")

    print("当前向量条数：", collection.count())
