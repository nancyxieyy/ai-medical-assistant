# backend/nodes/embedding_node.py

import os
from sentence_transformers import SentenceTransformer

class JinaEmbeddingNode:
    """Embedding Node框架"""
    
    def __init__(self, model_name="jinaai/jina-embeddings-v3"):
        """初始化模型"""
        # self.model = SentenceTransformer(model_name, trust_remote_code=True)
        cache_dir = os.getenv("HUGGINGFACE_HUB_CACHE", os.path.expanduser("~/.cache/huggingface"))
        os.makedirs(cache_dir, exist_ok=True)

        print(f"正在加载嵌入模型：{model_name}")
        print(f"模型缓存目录：{cache_dir}")

        self.model = SentenceTransformer(
            model_name,
            cache_folder=cache_dir,
            trust_remote_code=True
        )
        print("JinaEmbeddingNode 模型加载完成。")
    
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
    print(f"生成向量维度:{len(result['embeddings'][0])}")
    print(f"前10个数值示例:{result['embeddings'][0][:10]}")