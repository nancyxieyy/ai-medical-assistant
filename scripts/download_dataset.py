# scripts/download_dataset.py

from datasets import load_dataset
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# 下载数据集
huatuo_encyclopedia = load_dataset("FreedomIntelligence/huatuo_encyclopedia_qa")
huatuo_knowledge_graph = load_dataset("FreedomIntelligence/huatuo_knowledge_graph_qa")

# 创建本地目录并保存数据集
local_dir = "data/huatuo"

try:
    os.makedirs(os.path.join(local_dir, "huatuo_encyclopedia"), exist_ok=True)
    os.makedirs(os.path.join(local_dir, "huatuo_knowledge_graph"), exist_ok=True)
    huatuo_encyclopedia.save_to_disk(os.path.join(local_dir, "huatuo_encyclopedia"))
    huatuo_knowledge_graph.save_to_disk(os.path.join(local_dir, "huatuo_knowledge_graph"))
    print(f"数据集已成功下载并保存到 {local_dir} 目录下。")
except Exception as e:
    print(f"保存数据集时出错: {e}")