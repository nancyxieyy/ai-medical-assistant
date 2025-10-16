# backend/nodes/embedding_node.py

import os
from sentence_transformers import SentenceTransformer

os.environ["HUGGINGFACE_HUB_CACHE"] = ".cache/hf"

class JinaEmbeddingNode:
    """Embedding Node框架"""
    
    # 全局缓存
    _model = None
    
    def __init__(self, model_name="jinaai/jina-embeddings-v3"):
        """初始化模型"""
        cache_dir = ".cache/hf"
        os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir

        if JinaEmbeddingNode._model is None:
            print(f"正在加载嵌入模型：{model_name}")
            JinaEmbeddingNode._model = SentenceTransformer(model_name, trust_remote_code=True)
            print("嵌入模型加载完成（仅首次）")
        self.model = JinaEmbeddingNode._model
    
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