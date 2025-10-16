# AI 医疗问诊助手（AI Medical Assistant）

本项目基于 **LangGraph + Sherpa-ONNX + Qwen + Jina Embedding + ChromaDB**，  
实现了一个可在本地运行的 **智能语音问诊与报告生成系统**。  
系统能在医生与患者问诊过程中进行实时语音识别、生成建议、并自动生成结构化病历报告。

---

## 功能概览

- **实时语音识别 (ASR)**  
  使用 [Sherpa-ONNX](https://modelscope.cn/models/pengzhendong/sherpa-onnx-streaming-paraformer-bilingual-zh-en/files) 进行中英文双语流式识别。

- **LangGraph 智能图**  
  管理两个独立的智能 Agent：  
  - **Realtime Agent**：针对实时对话生成简短医疗建议；  
  - **Summary Agent**：问诊结束后生成完整病历报告。

- **RAG 医学知识检索**  
  通过 [Jina Embedding](https://huggingface.co/jinaai/jina-embeddings-v3) + ChromaDB 从医学知识库中检索相关内容辅助回答。

- **Qwen 模型文本生成**  
  使用 [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) 模型生成自然、结构化、符合医疗逻辑的报告文本。

---

## 项目结构

```Bash
backend/
│── ASR/
│ ├── Agent_ASR_Integration.py # 主入口：语音识别 + LangGraph 集成
│ └── medical_graphs.py # LangGraph 双 Agent 定义
│
│── nodes/
│ ├── embedding_node.py # 向量化节点（Jina Embedding）
│ ├── rag_query_node.py # RAG 检索节点（ChromaDB）
│ └── llm_doctor_node.py # 大语言模型节点（Qwen3-1.7B）
│
│── database/
│ ├── models.py / session.py # 数据模型与数据库连接
│
└── rag_store/ # 向量数据库（自动生成）
```

---

## 环境配置

### 1. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate
```

### 2. 安装依赖
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 下载语音识别模型
模型地址（从 ModelScope 下载）：
`https://modelscope.cn/models/pengzhendong/sherpa-onnx-streaming-paraformer-bilingual-zh-en`

解压后放到项目根目录下：
`./sherpa-onnx-streaming-paraformer-bilingual-zh-en/`

### 4. 运行项目
```bash
python -m backend.ASR.Agent_ASR_Integration
```

运行后系统会：
- 启动麦克风监听；
- 识别语音内容；
- 生成实时医疗建议；
- 问诊结束后输出一份病历报告草稿（report_draft.txt）。


## 模型说明
| 模型类型 | 使用模型                                         | 说明                   |
| -------- | ------------------------------------------------ | ---------------------- |
| 语音识别 | sherpa-onnx-streaming-paraformer-bilingual-zh-en | 实时语音转文本         |
| 向量模型 | jinaai/jina-embeddings-v3                        | 文本向量化用于检索     |
| 医疗生成 | Qwen/Qwen3-1.7B                                  | 医疗建议与病历报告生成 |

### 注意事项
- 运行前请确保麦克风正常工作；
- 模型较大，首次加载时间较长；
- 本系统仅供研究与演示，不可作为正式医疗诊断工具。

---

## 作者

[Mike Miao](https://github.com/mikey-miao)
[Nancy Xie](https://github.com/nancyxieyy)
