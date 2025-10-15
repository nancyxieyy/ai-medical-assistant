# medical_agent/run_local.py

from .src.agent.graph import graph

# 实时模式
res1 = graph.invoke({
    "texts": ["我这两天咳嗽发烧38度"],
    "mode": "realtime_advice"
}, entry_point="embedding_node")

print("\n实时建议:")
print(res1)

# 总结模式
res2 = graph.invoke({
    "texts": ["患者主诉咳嗽3天，伴有发烧和乏力"],
    "mode": "final_report"
}, entry_point="summary_start")

print("\n问诊总结:")
print(res2)