# 扫地机器人智能客服系统 — 设计文档

**日期：** 2026-06-05
**版本：** 1.0.0
**状态：** 设计确认，待实施

---

## 1. 项目概述

### 1.1 目标

针对扫地机器人用户在售前咨询、售后故障排查、日常维护中的痛点，构建基于大语言模型的智能客服系统。通过 ReAct 范式，结合 RAG 与多工具调用能力，实现从被动问答到主动服务的智能化升级。

### 1.2 技术选型

| 组件 | 选型 | 说明 |
|---|---|---|
| LLM | DeepSeek (deepseek-chat) | OpenAI 兼容接口，中文能力强，性价比高 |
| Agent 框架 | LangGraph `create_react_agent` | 比旧版 langchain.agents 更稳定，原生 streaming |
| Embedding | BAAI/bge-small-zh-v1.5 | 本地运行，512 维，~100MB |
| 向量库 | ChromaDB | 轻量持久化，LangChain 深度集成 |
| 后端 | FastAPI + SSE | 异步流式响应，解耦前后端 |
| 前端 | Streamlit | 快速聊天 UI，原生支持流式渲染 |
| 工具封装 | LangChain `@tool` 装饰器 | 统一接口，可组合可测试 |

### 1.3 当前状态

**已完成：**
- 项目骨架（目录结构、配置管理）
- LLM 封装（`src/llm/chat_model.py` — DeepSeek OpenAI 兼容接口）
- CLI 对话循环（`main.py`）
- 依赖管理（`requirements.txt` — langchain, streamlit, chromadb 等）
- 环境变量配置（`.env` — LLM/Embedding/Retrieval 参数）

**待实施：** 以下设计文档覆盖的全部模块

---

## 2. 整体架构

### 2.1 分层架构

```
┌──────────────────────────────────────────────────────┐
│                   🖥️ 前端层 (Streamlit)                 │
│         聊天 UI / 流式渲染 / 会话管理 / 工具可见          │
├──────────────────────────────────────────────────────┤
│                   🌐 API 层 (FastAPI)                  │
│        POST /chat (同步) / POST /chat/stream (SSE)     │
├──────────────────────────────────────────────────────┤
│                  🔗 中间件层 (Callbacks)               │
│        日志审计 / 性能监控 / 动态 Prompt 切换            │
├──────────────────────────────────────────────────────┤
│               🧠 Agent 层 (LangGraph ReAct)            │
│       推理循环 / 工具调度 / 对话记忆 / 流式输出          │
├──────────────────────────────────────────────────────┤
│                  🔧 工具层 (8 @tool)                   │
│    RAG检索 / 天气 / 定位 / 设备状态 / 抓取 / 规格 /      │
│                 故障排查 / 工单                         │
├──────────────────────────────────────────────────────┤
│               💾 数据层 (ChromaDB + Embedding)         │
│       文档加载 / 分块 / 向量化 / 语义检索 / Mock数据      │
└──────────────────────────────────────────────────────┘
```

### 2.2 目录结构

```
rag_agent_customer_service/
├── main.py                   # CLI 入口（已有，保留）
├── app.py                    # 🔴 Streamlit 前端
├── server.py                 # 🔴 FastAPI 服务入口
├── requirements.txt          # 已有
├── .env                      # 已有（需更新）
├── .env.example              # 环境变量模板
│
├── configs/                  # ✅ 已有
│   ├── __init__.py
│   └── settings.py           # 扩展：Embedding/Retrieval/Prompt 配置项
│
├── src/
│   ├── __init__.py
│   ├── llm/                  # ✅ 已有
│   │   ├── __init__.py
│   │   └── chat_model.py
│   │
│   ├── embedding/            # 🔴 本地 Embedding
│   │   ├── __init__.py
│   │   └── embeddings.py
│   │
│   ├── vectordb/             # 🔴 向量库层
│   │   ├── __init__.py
│   │   ├── loader.py         # 文档加载 & 分块
│   │   ├── store.py          # ChromaDB 写入 & 检索
│   │   └── retriever.py      # 语义检索接口
│   │
│   ├── tools/                # 🔴 8 个工具函数
│   │   ├── __init__.py
│   │   ├── knowledge_search.py
│   │   ├── weather.py
│   │   ├── location.py
│   │   ├── device_status.py
│   │   ├── web_scraper.py
│   │   ├── product_spec.py
│   │   ├── troubleshooting.py
│   │   └── ticket.py
│   │
│   ├── agent/                # 🔴 ReAct Agent 引擎
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── prompt.py
│   │   └── memory.py
│   │
│   ├── middleware/           # 🔴 中间件链
│   │   ├── __init__.py
│   │   ├── logging_mw.py
│   │   ├── monitor_mw.py
│   │   └── prompt_switch.py
│   │
│   ├── api/                  # 🔴 FastAPI 路由
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── schemas.py
│   │
│   └── streamlit/            # 🔴 Streamlit UI 组件
│       ├── __init__.py
│       ├── chat_ui.py
│       └── utils.py
│
├── data/
│   ├── kb/                   # 知识库原始文档（Mock）
│   │   ├── product_manual.md
│   │   ├── troubleshooting.md
│   │   └── maintenance.md
│   ├── mock/                 # Mock 数据
│   └── chroma_db/            # 向量持久化目录
│
├── logs/                     # 运行日志
└── docs/
    └── superpowers/specs/    # 设计文档
```

### 2.3 请求处理链路

```
Streamlit UI → HTTP POST → FastAPI /chat/stream
  → Middleware: AuditLogging (记请求)
  → Middleware: PromptSwitch (场景检测)
  → Agent: ReAct Loop
       ├── Thought: 分析下一步
       ├── Action: 调用工具
       ├── Observation: 获取结果
       └── Final Answer: 整合回答
  → Middleware: PerformanceMonitor (记耗时/token)
  → SSE Stream → Streamlit 逐 token 渲染
```

---

## 3. 数据服务层

### 3.1 Embedding 模块 (`src/embedding/embeddings.py`)

- **模型**：`BAAI/bge-small-zh-v1.5`
- **维度**：512
- **大小**：~100MB，首次自动下载缓存
- **接口**：兼容 LangChain `Embeddings` 基类
- **加载**：模块级单例，避免重复初始化

```python
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

def get_embedding_model():
    return HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
```

### 3.2 文档处理管道 (`src/vectordb/loader.py`)

```
data/kb/*.md, *.txt → TextLoader → RecursiveCharacterTextSplitter → Documents
```

- **chunk_size**: 512（从 .env `CHUNK_SIZE` 读取）
- **chunk_overlap**: 64（从 .env `CHUNK_OVERLAP` 读取）
- **策略**：递归分割，优先在段落/句子边界切分

### 3.3 向量存储 (`src/vectordb/store.py`)

```python
class VectorStore:
    def __init__(self):
        self.embedding = get_embedding_model()
        self.client = chromadb.PersistentClient(path="data/chroma_db")

    def create_collection(name: str) → Chroma Collection
    def add_documents(docs: list, collection: str) → None
    def similarity_search(query: str, k: int = 4) → List[Document]
    def delete_collection(name: str) → None
```

### 3.4 检索接口 (`src/vectordb/retriever.py`)

```python
class KnowledgeRetriever:
    def retrieve(query: str, top_k: int = 4) → List[Document]
    def retrieve_with_score(query: str, k: int = 4) → List[tuple]
    def ingest_directory(path: str) → int  # 返回入库文档数
```

### 3.5 Mock 知识库数据

3 个中文 Markdown 文档，每个 ~100-200 行：

| 文件 | 内容范围 |
|---|---|
| `data/kb/product_manual.md` | 产品型号、参数表、操作指南、WiFi 配网、APP 使用 |
| `data/kb/troubleshooting.md` | 错误代码（E01-E15）、常见故障、排查步骤 |
| `data/kb/maintenance.md` | 滤网更换周期、刷头清洁、传感器擦拭、固件升级 |

---

## 4. 工具函数层（8 个工具）

### 4.1 工具清单

| # | 工具名 | 分类 | 输入 → 输出 | 数据来源 |
|---|---|---|---|---|
| 1 | `search_knowledge_base` | RAG | query: str → 文档片段+来源 | ChromaDB（真实） |
| 2 | `get_weather` | 环境感知 | city: str → 天气信息 | Mock（预留API） |
| 3 | `get_user_location` | 环境感知 | ip: str → 城市/省份 | Mock（预留API） |
| 4 | `get_device_status` | 设备数据 | device_id: str → 状态JSON | Mock JSON |
| 5 | `fetch_web_content` | 外部数据 | url: str → 网页摘要 | 真实HTTP抓取 |
| 6 | `query_product_spec` | 设备数据 | model: str → 规格参数 | Mock 结构化数据 |
| 7 | `get_troubleshooting_guide` | 设备数据 | error_code: str → 排查步骤 | Mock + RAG |
| 8 | `submit_support_ticket` | 服务闭环 | user+issue+device → 工单ID | Mock 工单 |

### 4.2 Mock 策略

- **真实数据**：`search_knowledge_base` (ChromaDB)、`fetch_web_content` (HTTP)
- **结构化 Mock**：`get_device_status`、`query_product_spec`、`get_troubleshooting_guide` — JSON/文本 Mock，接口预留真实数据源切换
- **API 预留**：`get_weather`、`get_user_location` — Mock 数据先行，接口设计支持后续接入真实 API

### 4.3 工具注册

```python
# src/tools/__init__.py
ALL_TOOLS = [
    search_knowledge_base,
    get_weather,
    get_user_location,
    get_device_status,
    fetch_web_content,
    query_product_spec,
    get_troubleshooting_guide,
    submit_support_ticket,
]

def get_tools_for_scenario(scenario: str = "default") -> list:
    """按场景返回不同工具集"""
    ...
```

---

## 5. ReAct Agent 引擎

### 5.1 核心实现 (`src/agent/agent.py`)

- **框架**：`langgraph.prebuilt.create_react_agent`
- **工具绑定**：`llm.bind_tools(tools)` 利用 DeepSeek 原生 function calling
- **执行模式**：
  - `invoke(messages)` — 同步返回完整结果
  - `astream(messages)` — 异步流式，逐 token 返回
- **安全边界**：最大 10 轮 Thought-Action 迭代
- **流式事件**：`on_chat_model_stream` (token)、`on_tool_start`、`on_tool_end`

### 5.2 System Prompt (`src/agent/prompt.py`)

- **角色**：「小扫」，专业扫地机器人智能客服助手
- **原则**：先理解意图 → 需要事实时调用工具 → 整合信息友好回答 → 信息不足时诚实告知
- **场景变体**：
  - `SYSTEM_PROMPT_BASE` — 默认通用 Prompt
  - `PRESALE_PROMPT` — 售前咨询（产品推荐、参数对比）
  - `AFTERSALE_PROMPT` — 售后支持（故障排查、工单提交）
  - `MAINTENANCE_PROMPT` — 日常维护（耗材、保养、技巧）

### 5.3 对话记忆 (`src/agent/memory.py`)

- **存储**：`InMemoryChatMessageHistory` — 会话级内存存储
- **能力**：多会话隔离、最近 N 轮保留、手动清除
- **扩展**：预留 `RunnableWithMessageHistory` 接口，后续可切换到 Redis/DB

---

## 6. 中间件链

### 6.1 实现机制

基于 LangChain `BaseCallbackHandler`，利用框架原生钩子，无需侵入 Agent 代码。

### 6.2 中间件清单

**AuditLoggingHandler** (`middleware/logging_mw.py`)：
- `on_llm_start` — 记录 LLM 调用：时间、prompt 摘要、session_id
- `on_tool_start` — 记录工具调用：工具名、输入参数
- `on_tool_end` — 记录工具结果：输出摘要
- 输出格式：`logs/audit.jsonl`

**PerformanceMonitor** (`middleware/monitor_mw.py`)：
- 全链路耗时（ms）
- Token 总用量（prompt + completion）
- 工具调用次数
- 首字延迟（第一个 streaming token 的时间）

**PromptSwitch** (`middleware/prompt_switch.py`)：
- 关键词匹配检测用户意图场景（售前/售后/维护）
- 自动选择对应的 Prompt 变体
- 预留 LLM 意图分类升级路径

### 6.3 中间件链顺序

```
请求 → AuditLogging → PromptSwitch → Agent 执行 → PerformanceMonitor → 响应
```

---

## 7. FastAPI 服务 & SSE 流式

### 7.1 端点

| 端点 | 方法 | 说明 |
|---|---|---|
| `/api/v1/chat` | POST | 同步对话，返回完整答案 |
| `/api/v1/chat/stream` | POST | SSE 流式对话，逐 token 推送 |
| `/api/v1/health` | GET | 健康检查（模型、向量库状态） |

### 7.2 SSE 事件协议

```
event: token      ← LLM 输出 token
event: tool_start ← Agent 开始调用工具
event: tool_end   ← 工具调用完成
event: done       ← 对话结束
```

### 7.3 应用生命周期 (`server.py`)

- **startup**：初始化 Embedding 模型 → 向量库连接 → Agent 实例化 → 日志输出
- **CORS**：允许 Streamlit 跨域访问
- **启动命令**：`uvicorn server:app --reload --port 8000`

### 7.4 数据模型 (`src/api/schemas.py`)

```python
class ChatRequest(BaseModel):
    session_id: str
    query: str (1-2000 chars)
    user_id: str | None
    device_id: str | None

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    tool_calls: list[str]
```

---

## 8. Streamlit 前端

### 8.1 界面布局

- **侧边栏**：会话管理（新对话、历史列表、清除）+ 系统状态面板（模型、向量库）
- **主聊天区**：消息气泡（用户/AI）+ 工具调用过程展示（`st.status` 可折叠）
- **输入区**：`st.chat_input` 聊天输入框

### 8.2 核心特性

- **流式渲染**：`st.empty()` placeholder + SSE `iter_lines()` 逐 token 更新
- **工具可见**：`st.status()` 展示 Agent 工具调用过程
- **会话管理**：`st.session_state` 维护 session_id + messages
- **健康检查**：启动时检查 FastAPI 连接状态
- **错误处理**：API 超时/断连优雅降级

### 8.3 启动方式

```bash
# 1. 启动后端
uvicorn server:app --reload --port 8000

# 2. 启动前端
streamlit run app.py
```

---

## 9. 需要更新的现有文件

### 9.1 `configs/settings.py`

新增配置项：
- `EMBEDDING_MODEL` — 本地 Embedding 模型名
- `EMBEDDING_DEVICE` — cpu / cuda
- `RETRIEVAL_TOP_K` — 检索返回数量
- `CHUNK_SIZE` / `CHUNK_OVERLAP` — 文档分块参数
- `AGENT_MAX_ITERATIONS` — Agent 最大迭代轮数
- `LOG_LEVEL` / `LOG_DIR` — 日志配置

### 9.2 `requirements.txt`

新增依赖：
- `sentence-transformers` — 本地 Embedding
- `langgraph` — ReAct Agent
- `fastapi` / `uvicorn` — Web 服务
- `sse-starlette` — SSE 流式
- `beautifulsoup4` / `lxml` — 网页解析
- `chromadb` — 已存在
- `langchain-chroma` — LangChain Chroma 集成


---

## 10. 实施阶段划分

| Phase | 内容 | 依赖 |
|---|---|---|
| Phase 1 | 数据层 — Embedding + ChromaDB + Mock 文档 + 检索 | 无 |
| Phase 2 | 工具层 — 8 个 @tool 函数 | Phase 1 (RAG 工具依赖检索) |
| Phase 3 | Agent 层 — ReAct 引擎 + Prompt + Memory | Phase 1, 2 |
| Phase 4 | 中间件 + API — FastAPI + SSE + Callbacks | Phase 3 |
| Phase 5 | 前端 — Streamlit UI + 流式消费 | Phase 4 |
| Phase 6 | 集成测试 + 优化 — 端到端测试 + 首字延迟优化 | Phase 5 |
