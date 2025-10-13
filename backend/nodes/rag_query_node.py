# backend/nodes/rag_query_node.py

import chromadb
from typing import Dict, Any
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

class RAGQueryNode:
    """
    RAG 检索节点
    根据 embedding 从 Chroma 查询医学知识
    """
    
    def __init__(self, chroma_dir="rag_store"):
        # 初始化 Chroma 数据库
        chroma_dir = os.getenv("CHROMA_PERSIST", "rag_store")
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.client.get_or_create_collection("medical_knowledge")
        print(f"RAGQueryNode 已连接向量库(路径：{chroma_dir})")

    def run(self, state: Dict[str, Any]):
        """
        输入: state["embeddings"](来自上游 Embedding Node)
        输出: state["context"] (供 LLM 节点使用的文本上下文)
        """
        query_emb = state.get("embeddings")
        if not query_emb:
            print("RAG Query Node: 没收到 embedding 输入")
            return {"context": ""}
        
        # 向量检索
        results = self.collection.query(
            query_embeddings=query_emb, 
            n_results=3
        )
        
        # 格式化结果为可读文本
        docs = []
        for meta, doc in zip(results["metadatas"][0], results["documents"][0]):
            docs.append(f"[{meta['source_type']}] {meta['title']} → {doc[:500]}...")
            # print(f"[{meta['source_type']}] {meta['title']} → {doc[:300]}...")
        
        context_text = "\n".join(docs)
        return {"context": context_text}
    

if __name__ == "__main__":
    from embedding_node import JinaEmbeddingNode

    # 初始化两个节点
    embed_node = JinaEmbeddingNode()
    rag_node = RAGQueryNode()

    # 模拟医生输入
    query_text = "患者发烧两天并有轻微咳嗽，应如何处理？"
    emb = embed_node.run({"texts": [query_text]})["embeddings"]
    
    # 传入 RAG 节点检索
    result = rag_node.run({"embeddings": emb})
    print("\n检索结果:\n", result["context"])