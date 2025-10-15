# backend/ASR/medical_graphs.py

from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, Any, List

# 导入三个节点
from backend.nodes.embedding_node import JinaEmbeddingNode
from backend.nodes.rag_query_node import RAGQueryNode
from backend.nodes.llm_doctor_node import LLMDoctorAdviceNode

class MedicalState(TypedDict, total=False):
    """定义图的共享状态"""
    texts: List[str]
    embeddings: List[List[float]]
    context: str
    mode: str
    llm_output: str

def make_nodes():
    """通用节点"""
    embed = JinaEmbeddingNode()
    rag = RAGQueryNode()
    llm = LLMDoctorAdviceNode()
    
    def embedding_node_fn(state: MedicalState) -> Dict[str, Any]:
        return embed.run({"texts": state["texts"]})
    
    def rag_query_node_fn(state: MedicalState) -> Dict[str, Any]:
        return rag.run({"embeddings": state["embeddings"]})
    
    def llm_node_fn(state: MedicalState) -> Dict[str, Any]:
        return llm.run({
            "mode": state.get("mode","realtime_advice"),
            "transcript": state["texts"][0],
            "context":state.get("context", "")
            })
    
    return embedding_node_fn, rag_query_node_fn, llm_node_fn

def build_realtime_agent():
    """建立实时建议 Agent"""
    builder = StateGraph(MedicalState)
    embedding_node_fn, rag_query_node_fn, llm_node_fn = make_nodes()
    
    builder.add_node("embedding", embedding_node_fn)
    builder.add_node("rea_query", rag_query_node_fn)
    builder.add_node("llm_doctor", llm_node_fn)
    
    builder.set_entry_point("embedding")
    builder.add_edge("embedding", "rea_query")
    builder.add_edge("rea_query", "llm_doctor")
    builder.add_edge("llm_doctor", END)
    
    return builder.compile()

def build_summary_agent():
    """问诊总结 Agent"""
    builder = StateGraph(MedicalState)
    embedding_node_fn, rag_query_node_fn, llm_node_fn = make_nodes()
    
    builder.add_node("embedding", embedding_node_fn)
    builder.add_node("rea_query", rag_query_node_fn)
    builder.add_node("llm_doctor", llm_node_fn)
    
    builder.set_entry_point("embedding")
    builder.add_edge("embedding", "rea_query")
    builder.add_edge("rea_query", "llm_doctor")
    builder.add_edge("llm_doctor", END)
    
    return builder.compile()