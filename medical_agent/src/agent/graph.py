from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from backend.nodes.embedding_node import JinaEmbeddingNode
from backend.nodes.rag_query_node import RAGQueryNode
from backend.nodes.llm_doctor_node import LLMDoctorAdviceNode

# ---------------- 定义状态 ----------------
class State(TypedDict, total=False):
    transcript: str
    texts: List[str]
    embeddings: List[List[float]]
    context: str
    mode: str
    llm_output: str

# ---------------- 初始化节点 ----------------
embed = JinaEmbeddingNode()
rag = RAGQueryNode()
llm = LLMDoctorAdviceNode()

def embedding_node_fn(state: State) -> Dict[str, Any]:
    return embed.run(state)

def rag_node_fn(state: State) -> Dict[str, Any]:
    return rag.run(state)

def llm_node_fn(state: State) -> Dict[str, Any]:
    return llm.run(state)

# ---------------- 定义 Graph ----------------
builder = StateGraph(State)

builder.add_node("embedding_node", embedding_node_fn)
builder.add_node("rag_query_node", rag_node_fn)
builder.add_node("llm_doctor_node", llm_node_fn)

# 时回答 Agent（entry point）
builder.set_entry_point("embedding_node")
builder.add_edge("embedding_node", "rag_query_node")
builder.add_edge("rag_query_node", "llm_doctor_node")
builder.add_edge("llm_doctor_node", END)

# 问诊总结 Agent（第二入口）
builder.add_entry_point("summary_start")
builder.add_edge("summary_start", "embedding_node")

graph = builder.compile()