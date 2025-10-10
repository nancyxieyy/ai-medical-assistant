# AI 医疗语音助理系统（AI Medical Consultation Assistant）

基于 **Overhearing LLM Agent（旁听型智能体）** 理念的开源医疗语音辅助系统。  
系统可监听医生与患者的问诊语音，对语音进行转写与语义摘要，  
并由医生确认后生成正式病历或导出PDF报告。  
在提高医生记录效率的同时，保证数据安全与合规。

---

## 项目简介

**项目目标：**  
打造一个低成本、可本地运行的医疗语音辅助系统，  
实现以下核心功能：
- 医生问诊语音识别（FunASR）
- AI 自动生成结构化摘要（Qwen2）
- 医生确认与编辑（Human-in-the-loop）
- 一键导出病历报告（PDF）
- 医学知识检索问答（LangChain + Chroma）

系统设计遵循 **隐私优先、医生主导** 原则，  
不存储原始音频，仅保留脱敏文本结果，  
符合《个人信息保护法（PIPL）》与医疗信息安全要求。

---

## 技术栈

| 模块 | 使用技术 |
|------|------------|
| 后端 | FastAPI + Python 3.10 |
| 前端 | Vue3 + TailwindCSS |
| 语音识别 | Whisper / FasterWhisper |
| 摘要生成 | Ollama + Mistral / Llama3 / Qwen2 |
| 检索问答 | LangChain + ChromaDB |
| 数据存储 | SQLite + SQLAlchemy |
| 报告导出 | ReportLab / pdfkit |
| 部署 | Docker + Render / Vercel |

---

## 项目结构（计划中）

``` bash
ai-medical-assistant/
│
├── backend/
│ ├── api/
│ │ ├── audio.py            # 语音识别接口
│ │ ├── summary.py          # 摘要生成接口
│ │ ├── confirm.py          # 医生确认接口
│ │ ├── chat.py             # 医学问答接口（RAG）
│ │ ├── patient.py          # 病人数据管理
│ │ └── report.py           # 报告导出接口
│ ├── utils/
│ │ ├── whisper_utils.py    # Whisper 转写
│ │ ├── summarizer.py       # LLM 摘要逻辑
│ │ ├── pdf_tools.py        # PDF 报告生成
│ │ └── security.py         # 数据安全与脱敏
│ ├── database/
│ │ ├── __init__.py         # (空檔案)
│ │ ├── crud.py             # (空檔案) - 未來放資料庫增刪改查邏輯
│ │ ├── init_db.py          # (空檔案) - 未來放初始化資料庫的腳本
│ │ ├── models.py           # (空檔案) - 未來放 SQLAlchemy 的 ORM 模型
│ │ └── schemas.py
│ ├── rag/
│ │ ├── knowledge_base/
│ │ ├── build_index.py
│ │ └── retriever.py
│ ├── main.py
│ ├── .env                  # (空檔案) - 未來放設定和密碼
│ ├── requirements.txt
│ └── venv/
│
├── frontend/
│ ├── src/
│ │ ├── components/
│ │ └── pages/
│ ├── package.json
│ └── Dockerfile
│
├── docker-compose.yml
└── README.md
```

---

## 本地运行指南

1. 克隆项目：
```bash
   git clone git@github.com:nancyxieyy/ai-medical-assistant.git
   cd ai-medical-assistant
```

2. 启动后端：
``` bash
    cd backend
    pip install -r requirements.txt
    uvicorn main:app --reload
```

3. 启动前端：
``` bash
    cd frontend
    npm install
    npm run dev
```

4. 浏览器访问：
http://localhost:3000

---

## 当前功能（MVP）
- Whisper 语音转写模块
- 本地 LLM 摘要生成（JSON结构）
- 医生确认 / 编辑 / 忽略流程
- PDF 报告导出
- RAG 医学问答模块
- Docker 一键部署
- 数据与隐私设计

---

## 不保存原始音频，仅保存转写文本；
- 医生确认后才写入数据库；
- 所有处理可本地化运行；
- 遵循 人机共管（Human-in-the-loop） 设计原则。

---

## 团队成员
姓名	角色	主要职责
谢越莹	技术负责人 / 全栈工程师	系统架构、模型集成、后端与部署
苗嘉峻	产品负责人 / 前端与体验	产品设计、交互原型、前端开发与文档

---

## 开源协议
本项目基于 MIT License 开源。

---

## 后续计划
- 实时语音输入与流式摘要

- 医疗影像或检验报告扩展模块

- 病历智能搜索与分析

- 多语言支持

- 云端演示版本上线（Render + Vercel）
