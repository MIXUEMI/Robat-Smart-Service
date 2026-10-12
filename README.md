# 小扫 · 智能客服系统

基于 **LangGraph ReAct Agent** + **RAG** + **多工具调用** 的扫地机器人智能客服系统。

通过 ReAct 范式，集成 8 个原子化工具函数，实现从被动问答到主动服务的智能化升级。

## 功能特性

- 🧠 **ReAct Agent 引擎** — 基于 LangGraph `create_react_agent`，Thought → Action → Observation 推理循环
- 📚 **RAG 知识库检索** — ChromaDB 向量库 + BGE 本地 Embedding，语义检索产品手册/故障排查/维护指南
- 🔧 **8 个工具函数** — 知识检索、天气查询、位置定位、设备状态、网页抓取、产品规格、故障排查、工单提交
- 🔗 **中间件链** — 日志审计（JSONL）、全链路性能监控、动态 Prompt 场景切换
- ⚡ **SSE 流式响应** — FastAPI + SSE，首字延迟 < 800ms
- 🎨 **Streamlit 聊天 UI** — 流式渲染、工具调用可视化、会话管理

## 技术栈

| 组件 | 技术 |
|------|------|
| LLM | DeepSeek (deepseek-chat) |
| Agent 框架 | LangGraph + LangChain |
| Embedding | BAAI/bge-small-zh-v1.5 (本地，512维) |
| 向量库 | ChromaDB |
| 后端 | FastAPI + SSE |
| 前端 | Streamlit |
| 环境 | Python 3.10+ |

## 项目结构

```
rag_agent_customer_service/
├── app.py                     # Streamlit 前端入口
├── server.py                  # FastAPI 服务入口
├── main.py                    # CLI 对话入口（保留）
├── requirements.txt           # Python 依赖
├── .env / .env.example        # 环境变量配置
├── .gitignore
│
├── configs/
│   └── settings.py            # 全局配置（LLM/Embedding/Agent/检索/日志）
│
├── src/
│   ├── llm/                   # LLM 封装
│   │   └── chat_model.py      # DeepSeek OpenAI 兼容接口
│   │
│   ├── embedding/             # 本地 Embedding
│   │   └── embeddings.py      # BGE-small-zh 模型加载（单例）
│   │
│   ├── vectordb/              # 向量库层
│   │   ├── loader.py          # 文档加载 & 递归分块管道
│   │   ├── store.py           # ChromaDB 存储 & 语义检索
│   │   └── retriever.py       # 高层检索 API + 结果格式化
│   │
│   ├── tools/                 # 工具函数层（8 个 @tool）
│   │   ├── __init__.py        # 工具注册 & 场景子集
│   │   ├── knowledge_search.py # RAG 知识库检索
│   │   ├── weather.py         # 天气查询（Mock）
│   │   ├── location.py        # 用户定位（Mock）
│   │   ├── device_status.py   # 设备状态（Mock）
│   │   ├── web_scraper.py     # 网页抓取（真实 HTTP）
│   │   ├── product_spec.py    # 产品规格查询（Mock）
│   │   ├── troubleshooting.py # 故障排查指南（Mock）
│   │   └── ticket.py          # 工单提交（Mock）
│   │
│   ├── agent/                 # ReAct Agent 引擎
│   │   ├── agent.py           # Agent 创建 & 同步/异步执行 & 流式输出
│   │   ├── prompt.py          # System Prompt 模板（4 个场景变体）
│   │   └── memory.py          # 对话历史管理（多会话隔离）
│   │
│   ├── middleware/            # 中间件链（Callback Handlers）
│   │   ├── logging_mw.py      # 日志审计（JSONL）
│   │   ├── monitor_mw.py      # 性能监控（延迟/token/工具调用）
│   │   └── prompt_switch.py   # 关键词场景检测 & 动态 Prompt 切换
│   │
│   └── api/                   # FastAPI 路由
│       ├── routes.py          # /chat, /chat/stream, /health
│       └── schemas.py         # Pydantic 请求/响应模型
│
├── data/
│   ├── kb/                    # 知识库文档（Mock）
│   │   ├── product_manual.md  # 产品使用手册
│   │   ├── troubleshooting.md # 故障排查指南
│   │   └── maintenance.md     # 日常维护指南
│   ├── mock/
│   │   └── mock_data.json     # 设备/产品/天气 Mock 数据集
│   └── chroma_db/             # 向量库持久化目录
│
├── logs/                      # 运行日志 & 审计日志
└── docs/
    └── superpowers/           # 设计文档 & 实施计划
```

## 架构概览

```
┌──────────────────────────────────────────┐
│           🖥️ 前端层 (Streamlit)            │
│     聊天 UI / 流式渲染 / 工具可见          │
├──────────────────────────────────────────┤
│            🌐 API 层 (FastAPI)            │
│   POST /chat  |  POST /chat/stream (SSE) │
├──────────────────────────────────────────┤
│          🔗 中间件层 (Callbacks)          │
│   日志审计 / 性能监控 / 动态Prompt切换     │
├──────────────────────────────────────────┤
│        🧠 Agent 层 (LangGraph ReAct)      │
│   Thought → Action → Observation → Answer│
├──────────────────────────────────────────┤
│            🔧 工具层 (8 @tool)            │
│  RAG / 天气 / 定位 / 设备 / 抓取 /        │
│  规格 / 故障排查 / 工单                   │
├──────────────────────────────────────────┤
│      💾 数据层 (ChromaDB + Embedding)     │
│  文档加载 → 分块 → 向量化 → 语义检索      │
└──────────────────────────────────────────┘
```

请求链路：`Streamlit → HTTP → FastAPI → 中间件 → ReAct Agent → Tools → SSE Stream → 逐token渲染`

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境（需要 Python 3.10+）
python -m venv .venv

# 激活（Windows）
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 复制模板
copy .env.example .env

# 编辑 .env，填入你的 DeepSeek API Key
# DEEPSEEK_API_KEY=sk-your_key_here
```

### 3. 初始化知识库

```bash
# 将 Mock 文档写入向量库（首次运行自动执行）
python -c "from src.vectordb.retriever import KnowledgeRetriever; r = KnowledgeRetriever(); print(f'Ingested {r.ingest_directory(\"data/kb\", force=True)} chunks')"
```

### 4. 启动服务

**方式一：CLI 对话（简单测试）**
```bash
python main.py
```

**方式二：完整前后端（推荐）**
```bash
# Terminal 1 — 后端
.venv\Scripts\activate
uvicorn server:app --reload --port 8000

# Terminal 2 — 前端
.venv\Scripts\activate
streamlit run app.py
```

然后打开浏览器访问 http://localhost:8501

### 5. API 文档

后端启动后访问 Swagger UI：http://localhost:8000/docs

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/chat` | POST | 同步对话，返回完整回答 |
| `/api/v1/chat/stream` | POST | SSE 流式对话，逐 token 推送 |
| `/api/v1/health` | GET | 健康检查（模型、向量库状态） |

### SSE 事件协议

```
event: token       ← LLM 输出 token
event: tool_start  ← Agent 开始调用工具
event: tool_end    ← 工具调用完成
event: done        ← 对话结束
```

## 对话示例

```
用户：我的扫地机报E05错误怎么办？

小扫：🔧 正在调用 get_troubleshooting_guide
      🔧 正在调用 search_knowledge_base

      ## 🔧 E05 故障：边刷缠绕

      您的机器人显示 E05 代码，意思是边刷被异物缠住了。

      ### 排查步骤：
      1. 将机器翻转，找到底部边刷
      2. 取下边刷，检查是否有毛发、线头缠绕
      3. 用清洁工具刀片割断并清除缠绕物
      4. 检查边刷卡槽，清理残留物
      5. 重新安装边刷，确认旋转顺畅

      ### 💡 预防建议
      建议每周定期清理边刷，尤其家里有宠物或长发成员时。
```

```
用户：L5和L3有什么区别？

小扫：🔧 正在调用 query_product_spec

      ## L5 vs L3 对比

      | 参数 | L3 标准版 | L5 旗舰版 |
      |------|----------|----------|
      | 吸力 | 3500Pa | 5000Pa |
      | 续航 | 150min | 180min |
      | 导航 | LDS激光 | LDS+AI视觉 |
      | 价格 | ¥2,499 | ¥3,999 |

      L5 适合大户型（≤200㎡）和对智能避障有更高要求的用户。
```

## 工具清单

| # | 工具 | 分类 | 数据来源 |
|---|------|------|---------|
| 1 | `search_knowledge_base` | RAG 检索 | ChromaDB |
| 2 | `get_weather` | 环境感知 | Mock（预留 API） |
| 3 | `get_user_location` | 环境感知 | Mock（预留 API） |
| 4 | `get_device_status` | 设备数据 | Mock JSON |
| 5 | `fetch_web_content` | 外部数据 | 真实 HTTP 抓取 |
| 6 | `query_product_spec` | 设备数据 | Mock 结构化数据 |
| 7 | `get_troubleshooting_guide` | 设备数据 | Mock 故障库 |
| 8 | `submit_support_ticket` | 服务闭环 | Mock 工单系统 |
