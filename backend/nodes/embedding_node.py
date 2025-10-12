# backend/nodes/embedding_node.py

import os
os.environ["HUGGINGFACE_HUB_CACHE"] = ".cache/hf"

from sentence_transformers import SentenceTransformer

# 初始化模型
# model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)

# texts = ["发烧两天咳嗽是否需要用抗生素"]

# embeddings = model.encode(texts, normalize_embeddings=True)

# print(f"向量维度：{len(embeddings[0])}")
# print(f"前10个数值示例：{embeddings[0][:10]}")


class JinaEmbeddingNode:
    """Embedding Node框架"""
    
    def __init__(self, model_name="jinaai/jina-embeddings-v3"):
        """初始化模型"""
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
    
    def run(self, state):
        """
        state 是 LangGraph 的上下文字典
        state["texts"] 是要转成向量的文本列表
        """
        texts = state["texts"]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return {"embeddings": embeddings.tolist()}

if __name__ == "__main__":
    node = JinaEmbeddingNode()
    sample = {"texts": ["发烧两天，咳嗽，是否需要使用抗生素？"]}
    result = node.run(sample)
    print(f"生成向量维度：{len(result['embeddings'][0])}")
    print(f"前10个数值示例：{result['embeddings'][0][:10]}")