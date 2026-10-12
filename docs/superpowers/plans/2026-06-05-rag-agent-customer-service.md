# 扫地机器人智能客服系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建基于 LangGraph ReAct Agent 的扫地机器人智能客服系统，集成 RAG + 8 工具调用 + FastAPI SSE 流式 + Streamlit 前端

**Architecture:** 6 层分层架构 — 前端(Streamlit) → API(FastAPI) → 中间件(Callbacks) → Agent(LangGraph ReAct) → 工具(8 @tool) → 数据(ChromaDB + 本地 Embedding)

**Tech Stack:** Python 3.10+, LangChain 0.2.x, LangGraph, FastAPI, Streamlit, ChromaDB, sentence-transformers (BAAI/bge-small-zh-v1.5), DeepSeek API

---

## File Structure Map

```
rag_agent_customer_service/
├── .env                              # [MODIFY] 对齐 settings.py 的 key 命名
├── .env.example                      # [CREATE] 环境变量模板
├── requirements.txt                  # [MODIFY] 新增 langgraph, sentence-transformers, fastapi 等
├── server.py                         # [CREATE] FastAPI 入口
├── app.py                            # [CREATE] Streamlit 前端入口
│
├── configs/
│   └── settings.py                   # [MODIFY] 扩展 Embedding/Retrieval/Logging 配置
│
├── src/
│   ├── embedding/
│   │   ├── __init__.py               # [CREATE]
│   │   └── embeddings.py            # [CREATE] 本地 Embedding 模型加载
│   │
│   ├── vectordb/
│   │   ├── __init__.py               # [CREATE]
│   │   ├── loader.py                # [CREATE] 文档加载 & 分块管道
│   │   ├── store.py                 # [CREATE] ChromaDB 向量存储
│   │   └── retriever.py            # [CREATE] 语义检索封装
│   │
│   ├── tools/
│   │   ├── __init__.py               # [CREATE] 工具注册 ALL_TOOLS
│   │   ├── knowledge_search.py       # [CREATE] RAG 检索工具
│   │   ├── weather.py               # [CREATE] 天气查询 (Mock)
│   │   ├── location.py              # [CREATE] 用户定位 (Mock)
│   │   ├── device_status.py         # [CREATE] 设备状态 (Mock)
│   │   ├── web_scraper.py           # [CREATE] 网页抓取 (真实HTTP)
│   │   ├── product_spec.py          # [CREATE] 产品规格 (Mock)
│   │   ├── troubleshooting.py       # [CREATE] 故障排查 (Mock)
│   │   └── ticket.py                # [CREATE] 工单提交 (Mock)
│   │
│   ├── agent/
│   │   ├── __init__.py               # [CREATE]
│   │   ├── agent.py                 # [CREATE] ReAct Agent 引擎
│   │   ├── prompt.py                # [CREATE] System Prompt 模板
│   │   └── memory.py               # [CREATE] 对话历史管理
│   │
│   ├── middleware/
│   │   ├── __init__.py               # [CREATE] 中间件链注册
│   │   ├── logging_mw.py            # [CREATE] 日志审计 Callback
│   │   ├── monitor_mw.py            # [CREATE] 性能监控 Callback
│   │   └── prompt_switch.py         # [CREATE] 动态 Prompt 切换
│   │
│   └── api/
│       ├── __init__.py               # [CREATE]
│       ├── routes.py                # [CREATE] FastAPI 端点
│       └── schemas.py               # [CREATE] Pydantic 模型
│
├── data/
│   ├── kb/
│   │   ├── product_manual.md         # [CREATE] Mock 产品手册
│   │   ├── troubleshooting.md        # [CREATE] Mock 故障排查文档
│   │   └── maintenance.md           # [CREATE] Mock 维护文档
│   └── mock/
│       └── mock_data.json            # [CREATE] 设备/产品 Mock 数据集
```

---

## Phase 1: 数据层 — Embedding + ChromaDB + Mock 文档 + 检索

### Task 1.1: 更新项目配置和依赖

**Files:**
- Modify: `configs/settings.py`
- Modify: `.env`
- Create: `.env.example`
- Modify: `requirements.txt`

- [ ] **Step 1: 更新 requirements.txt 添加新依赖**

Replace the entire file content:

```txt
langchain==0.2.17
langchain-openai==0.1.25
langchain-community==0.2.19
python-dotenv==1.0.1
openai==1.55.3
streamlit==1.38.0
chromadb==0.5.5
pandas==2.2.2
numpy==1.26.4
pydantic==2.8.2
tiktoken==0.7.0
sentence-transformers==3.0.1
langgraph==0.2.0
langchain-chroma==0.1.4
fastapi==0.112.0
uvicorn==0.30.6
sse-starlette==2.1.3
beautifulsoup4==4.12.3
lxml==5.3.0
requests==2.32.3
torch==2.3.1
```

- [ ] **Step 2: 更新 .env 对齐 settings.py 命名规范 + 新增配置项**

Replace the entire file content:

```env
# ── LLM ──
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=YOUR_API_KEY_HERE
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

# ── Embedding ──
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
EMBEDDING_DEVICE=cpu

# ── Retrieval ──
RETRIEVAL_TOP_K=4
CHUNK_SIZE=512
CHUNK_OVERLAP=64

# ── Agent ──
AGENT_MAX_ITERATIONS=10

# ── Logging ──
LOG_LEVEL=INFO
LOG_DIR=logs
```

- [ ] **Step 3: 创建 .env.example**

```env
# ── LLM ──
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

# ── Embedding ──
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
EMBEDDING_DEVICE=cpu

# ── Retrieval ──
RETRIEVAL_TOP_K=4
CHUNK_SIZE=512
CHUNK_OVERLAP=64

# ── Agent ──
AGENT_MAX_ITERATIONS=10

# ── Logging ──
LOG_LEVEL=INFO
LOG_DIR=logs
```

- [ ] **Step 4: 扩展 configs/settings.py**

Replace the entire file content:

```python
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ── App ──
    APP_NAME: str = os.getenv("APP_NAME", "RAG Agent Customer Service")
    APP_ENV: str = os.getenv("APP_ENV", "dev")

    # ── LLM ──
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "deepseek")

    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # ── Embedding ──
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")

    # ── Retrieval ──
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "4"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))

    # ── Agent ──
    AGENT_MAX_ITERATIONS: int = int(os.getenv("AGENT_MAX_ITERATIONS", "10"))

    # ── Logging ──
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR: str = os.getenv("LOG_DIR", "logs")

    # ── ChromaDB ──
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "robot_vacuum_kb")


settings = Settings()
```

- [ ] **Step 5: 安装依赖验证**

Run: `pip install -r requirements.txt`
Expected: All packages install successfully without errors.

- [ ] **Step 6: Commit**

```bash
git add configs/settings.py .env .env.example requirements.txt
git commit -m "chore: update config and dependencies for Phase 1-6"
```

---

### Task 1.2: 创建 Mock 知识库文档

**Files:**
- Create: `data/kb/product_manual.md`
- Create: `data/kb/troubleshooting.md`
- Create: `data/kb/maintenance.md`
- Create: `data/mock/mock_data.json`

- [ ] **Step 1: 创建 data/kb/product_manual.md**

```markdown
# 灵悦扫地机器人 — 产品使用手册

## 产品型号概览

| 型号 | 吸力 (Pa) | 电池容量 (mAh) | 续航 (min) | 尘盒容量 (ml) | 水箱容量 (ml) | 导航方式 | 参考价格 (元) |
|------|----------|---------------|-----------|-------------|-------------|---------|-------------|
| L1 基础版 | 2500 | 3200 | 120 | 400 | 200 | 陀螺仪 | 1499 |
| L3 标准版 | 3500 | 5200 | 150 | 500 | 300 | LDS激光 | 2499 |
| L5 旗舰版 | 5000 | 6500 | 180 | 600 | 400 | LDS+AI视觉 | 3999 |
| L5 Pro 顶配版 | 6000 | 6500 | 200 | 600 | 400 | LDS+AI+3D结构光 | 4999 |

## 快速入门

### 开箱检查
打开包装后，请检查以下物品是否齐全：
1. 扫地机器人主机 ×1
2. 充电底座 ×1
3. 电源适配器 ×1
4. 边刷 ×2（备用）
5. HEPA滤网 ×1（备用）
6. 拖布 ×2
7. 清洁工具 ×1
8. 用户手册 ×1

### 首次使用步骤
1. 将充电底座靠墙放置，左右各留 0.5m 空间，前方留 1.5m
2. 安装边刷：将边刷对准卡槽按下，听到"咔哒"声即安装到位
3. 长按电源键 3 秒开机
4. 下载"灵悦智家"APP，按照指引完成 WiFi 配网
5. 首次使用建议充满电（约4小时）后再开始清扫

### WiFi 配网指南
1. 确保家中 WiFi 为 2.4GHz（不支持 5GHz）
2. 打开"灵悦智家"APP → 添加设备 → 选择对应型号
3. 同时按住机器"回充"+"局部清扫"键 3 秒，听到"等待配网"语音提示
4. 在 APP 中输入 WiFi 密码，等待连接完成（约 30 秒）
5. 配网成功后，机器人会语音提示"配网成功"

### 配网失败排查
- 确认 WiFi 是 2.4GHz 而非 5GHz
- 确认密码正确
- 将机器靠近路由器重新尝试
- 长按复位键 10 秒恢复出厂设置后重试
- L5 系列支持蓝牙辅助配网，可以尝试蓝牙模式

## 操作指南

### 清扫模式
- **自动清扫**：全屋弓字形规划清扫，适合日常使用
- **安静模式**：降低吸力噪音至 55dB，适合夜间或家有婴儿
- **强力模式**：最大吸力，适合地毯或重污区域
- **定点清扫**：以机器为中心 1.5m×1.5m 区域深度清扫
- **沿边清扫**：沿墙边和家具边缘清扫

### APP 功能
- **实时地图**：查看清扫路径和已扫区域
- **禁区设置**：在地图上画出不想让机器人进入的区域
- **分区清扫**：指定某个房间单独清扫
- **预约清扫**：设置每天定时开始清扫
- **耗材状态**：查看滤网、边刷、主刷的剩余寿命

### 语音控制
L3 及以上型号支持语音控制：
- "小灵小灵，开始清扫"
- "小灵小灵，回去充电"
- "小灵小灵，安静模式"

## 技术规格（L5 旗舰版）
- **导航技术**：LDS激光雷达 + AI视觉识别
- **避障能力**：可识别 100+ 种常见物体（拖鞋、数据线、宠物粪便等）
- **越障能力**：≤ 2cm 门槛
- **爬坡角度**：≤ 15°
- **噪音**：标准模式 62dB，安静模式 55dB，强力模式 68dB
- **尘盒**：600ml，带尘满检测
- **水箱**：400ml 智能电控水箱，3 档水量调节
- **滤网**：HEPA 高效过滤，可水洗
- **适用面积**：≤ 200㎡
```

- [ ] **Step 2: 创建 data/kb/troubleshooting.md**

```markdown
# 灵悦扫地机器人 — 故障排查指南

## 错误代码速查表

| 错误代码 | 故障现象 | 可能原因 | 排查步骤 |
|----------|---------|---------|---------|
| E01 | 无法开机 | 电池耗尽 / 电源键故障 | 1.连接充电底座充电30分钟 2.长按电源键10秒强制重启 3.若仍无效联系售后 |
| E02 | 无法充电 | 充电触点脏污 / 适配器故障 | 1.用干布清洁机器底部和充电底座的充电触点 2.检查适配器指示灯是否亮 3.更换插座测试 |
| E03 | 边刷不转 | 毛发缠绕 / 边刷磨损 | 1.取下边刷清理缠绕毛发 2.检查边刷卡扣是否磨损 3.更换备用边刷 |
| E04 | 主刷不转 | 异物卡住 / 皮带断裂 | 1.翻转机器取出主刷 2.清理刷头两端缠绕物 3.检查滚刷仓内是否有异物 4.重新安装主刷 |
| E05 | 边刷缠绕 | 毛发/线头缠绕过紧 | 1.翻转机器 2.取下边刷 3.用清洁工具上的刀片割断缠绕物 4.清理后重新安装 |
| E06 | 主刷缠绕 | 长发/宠物毛发缠绕 | 1.取出主刷 2.用清洁工具沿刷体沟槽切割毛发 3.两端轴承也需清理 4.清理完毕后装回 |
| E07 | 无法回充 | 基站信号异常 / 红外传感器脏污 | 1.清洁机器和基站的红外传感器窗口 2.检查基站电源和指示灯 3.将机器手动放到基站前1米处重新测试 4.重新配对基站 |
| E08 | 悬崖传感器异常 | 传感器脏污 / 底部有强反光 | 1.用软布清洁底部4个悬崖传感器 2.避免在深色高反光地板上使用 3.检查传感器窗口是否有遮挡 |
| E09 | 激光头异常 | LDS激光雷达被遮挡或卡住 | 1.检查顶部激光塔是否被物体压住 2.用手轻轻转动激光头确认是否顺畅 3.清理激光塔周围的灰尘 4.重新开机 |
| E10 | 风机异常 | 风机进风口堵塞 / 风机故障 | 1.检查并清理尘盒进风口 2.清洗或更换 HEPA 滤网 3.确认尘盒安装到位 4.依然异常联系售后 |
| E11 | 水箱异常 | 水箱未安装到位 / 水泵故障 | 1.取出水箱重新安装 2.检查水箱底部出水孔是否堵塞 3.清洁水箱触点 4.更换水箱密封圈 |
| E12 | 轮子悬空 | 机器人被卡住或抬起 | 1.检查机器人是否卡在门槛或家具下 2.将机器移到平地上 3.检查驱动轮是否能正常转动 |
| E13 | 电池过热 | 环境温度过高 / 电池散热不良 | 1.将机器移到阴凉处等待冷却 2.检查电池仓是否有灰尘堆积 3.避免在超过 40°C 环境使用 |
| E14 | 传感器校准失败 | 传感器被干扰 / 需要重新校准 | 1.将机器放在平整地面上 2.关机重启 3.通过 APP 执行传感器校准 |
| E15 | 固件更新失败 | 网络中断 / 电量不足 | 1.确保 WiFi 连接正常 2.确保电量 > 30% 3.放在基站上重新升级 4.通过 APP 手动触发更新 |

## 常见问题 Q&A

### 清扫效果不佳
- **漏扫严重**：检查传感器是否脏污，清理后重新建图
- **吸力不足**：检查尘盒是否已满，HEPA 滤网是否需要更换（建议每 3 个月更换）
- **拖地不干净**：检查拖布是否需要更换，调整 APP 中的水量档位
- **边角残留**：边刷是易耗品，建议每 6 个月更换

### 噪音问题
- **异常噪音**：检查主刷/边刷/轮子是否有异物卡住
- **噪音增大**：可能是主刷轴承磨损或风机进风口堵塞
- **异响**：关机后检查底盘是否有硬物刮擦痕迹

### 续航问题
- **续航明显缩短**：电池老化，建议联系售后更换电池包（保修期内免费）
- **频繁回充**：检查是否设置了过于频繁的回充策略，或清扫面积超出产品适用范围

### 建图问题
- **地图偏移**：在 APP 中删除旧地图，重新建图
- **建图不完整**：确保所有房门打开，光线充足，首次建图建议跟随观察
- **禁区不生效**：检查 APP 版本是否为最新，重新设置禁区
```

- [ ] **Step 3: 创建 data/kb/maintenance.md**

```markdown
# 灵悦扫地机器人 — 日常维护指南

## 定期维护时间表

| 维护项目 | 频率 | 更换周期 | 操作说明 |
|----------|------|---------|---------|
| 清理尘盒 | 每次清扫后 | — | 取出尘盒，打开盖子倒掉垃圾，可用水冲洗后晾干 |
| 清理边刷 | 每周 | 6个月 | 取下边刷，用清洁工具割断缠绕毛发，检查刷毛磨损 |
| 清理主刷 | 每周 | 12个月 | 取出主刷，沿沟槽清理缠绕毛发，两端轴承上油 |
| 清洗 HEPA 滤网 | 每2周 | 3个月 | 取出滤网，用清水冲洗（不用肥皂），彻底晾干后装回 |
| 清洗拖布 | 每次拖地后 | 3个月 | 拆下拖布，手洗或机洗，晾干后使用 |
| 清洁传感器 | 每月 | — | 用柔软干布擦拭激光头、悬崖传感器、红外传感器窗口 |
| 清洁充电触点 | 每月 | — | 用干布擦拭机器底部和充电底座的金属触点 |
| 清洁万向轮 | 每月 | — | 取下万向轮，清理缠绕毛发和灰尘 |
| 检查驱动轮 | 每3月 | — | 检查轮子转动是否顺畅，清理轮轴缠绕物 |
| 更换边刷 | — | 6个月 | 刷毛明显磨损变形时更换 |
| 更换 HEPA 滤网 | — | 3个月 | 清洗后仍有异味或吸力下降时更换 |
| 更换主刷 | — | 12个月 | 刷毛磨损严重或刷体变形时更换 |
| 更换拖布 | — | 3个月 | 拖布变硬或清洁效果明显下降时更换 |

## 详细清洁步骤

### 如何清理主刷
1. 翻转机器人，找到主刷仓盖板（两个卡扣）
2. 同时按压卡扣，取下盖板
3. 取出主刷组件（含刷体和两端轴承）
4. 用清洁工具上的小刀沿刷体沟槽割断缠绕的毛发
5. 取下两端轴承，清理轴上的缠绕物
6. 用干布擦拭滚刷仓内部
7. 重新安装轴承 → 放入主刷 → 扣紧盖板（听到"咔哒"声）

### 如何清理传感器
1. **激光雷达**：用柔软干布轻轻擦拭顶部激光塔的透明窗口
2. **悬崖传感器**：机器底部有4个圆形窗口（前后各2），逐一擦拭
3. **红外回充传感器**：机器前方和充电底座各有红外窗口，用干布清洁
4. **沿墙传感器**：机器右侧面的小窗口
5. **注意**：切勿使用湿布、酒精或清洁剂擦拭传感器，可能损坏涂层

### 如何清洗滤网
1. 打开机器上盖，取出尘盒
2. 打开尘盒盖子，取出 HEPA 滤网
3. 在垃圾桶上方轻轻敲打滤网，去除表面灰尘
4. 用清水从滤网外侧向内侧冲洗（不要反向）
5. 不可使用刷子、肥皂或清洁剂
6. 将滤网放在通风处彻底晾干（至少 24 小时）
7. 确保滤网完全干燥后再装回使用，否则会发霉并损坏风机

## 季度深度保养

### 春季保养（花粉季节）
- 重点清洁 HEPA 滤网，花粉季建议每 2 周清洗一次
- 检查并清理所有传感器，花粉可能影响传感器精度
- 拖地时可在水箱中加少量专用清洁液（不可使用消毒液）

### 夏季保养（高温季节）
- 注意电池散热，避免在阳光直射的房间充电
- 清理尘盒和滤网频率加倍
- 检查驱动轮是否有热胀冷缩导致的松动

### 秋季保养（落叶季节）
- 检查并清理主刷和边刷，秋季毛发脱落增多
- 清理万向轮，落叶碎屑容易缠绕
- 建议做一次深度传感器清洁

### 冬季保养（干燥季节）
- 干燥天气静电多，尘盒和滤网易吸附细尘，清洁频率提升
- 拖地频率可适当减少，防止地板过湿
- 电池在低温下续航可能下降 10-15%，属正常现象

## 固件升级指南
1. 确保机器人电量 > 30% 并放在充电底座上
2. 打开"灵悦智家"APP → 设备设置 → 固件升级
3. 点击"检查更新"，如有新版本会显示更新内容
4. 点击"立即升级"，升级过程约 5-10 分钟
5. 升级期间机器人会重启，请勿断电
6. 升级完成后会语音提示"升级成功"
7. 建议开启 APP 中的"自动更新"开关，获取最新功能和优化

## 耗材购买渠道
- "灵悦智家"APP 商城（推荐，保证正品）
- 灵悦官方天猫/京东旗舰店
- 授权线下门店（可在 APP 中查询最近门店）
- 耗材套装（滤网×2 + 边刷×2 + 拖布×2）：¥149
```

- [ ] **Step 4: 创建 data/mock/mock_data.json**

```json
{
  "devices": {
    "RB-001": {
      "device_id": "RB-001",
      "model": "L5",
      "nickname": "客厅小灵",
      "firmware_version": "4.2.1",
      "battery": 85,
      "mode": "自动清扫",
      "status": "清扫中",
      "filter_life": 72,
      "side_brush_life": 45,
      "main_brush_life": 68,
      "error_code": null,
      "last_clean": "2026-06-05 08:30",
      "total_area": 128.5,
      "total_hours": 342,
      "warranty_expire": "2027-03-15"
    },
    "RB-002": {
      "device_id": "RB-002",
      "model": "L3",
      "nickname": "卧室助手",
      "firmware_version": "4.1.8",
      "battery": 42,
      "mode": "回充中",
      "status": "返回充电",
      "filter_life": 30,
      "side_brush_life": 15,
      "main_brush_life": 55,
      "error_code": "E05",
      "last_clean": "2026-06-04 19:00",
      "total_area": 95.2,
      "total_hours": 210,
      "warranty_expire": "2026-08-20"
    },
    "RB-003": {
      "device_id": "RB-003",
      "model": "L1",
      "nickname": "书房小扫",
      "firmware_version": "3.9.2",
      "battery": 100,
      "mode": "待机",
      "status": "充电完成",
      "filter_life": 90,
      "side_brush_life": 80,
      "main_brush_life": 88,
      "error_code": null,
      "last_clean": "2026-06-04 22:00",
      "total_area": 45.8,
      "total_hours": 120,
      "warranty_expire": "2027-11-01"
    }
  },
  "products": {
    "L1": {
      "model": "L1 基础版",
      "suction_power_pa": 2500,
      "battery_capacity_mah": 3200,
      "runtime_min": 120,
      "dust_bin_ml": 400,
      "water_tank_ml": 200,
      "navigation": "陀螺仪",
      "obstacle_avoidance": "机械碰撞",
      "noise_db": 65,
      "max_area_sqm": 100,
      "wifi_band": "2.4GHz",
      "app_support": true,
      "voice_control": false,
      "price_cny": 1499,
      "weight_kg": 3.2,
      "dimensions_mm": "330×330×78"
    },
    "L3": {
      "model": "L3 标准版",
      "suction_power_pa": 3500,
      "battery_capacity_mah": 5200,
      "runtime_min": 150,
      "dust_bin_ml": 500,
      "water_tank_ml": 300,
      "navigation": "LDS激光",
      "obstacle_avoidance": "LDS避障",
      "noise_db": 62,
      "max_area_sqm": 150,
      "wifi_band": "2.4GHz",
      "app_support": true,
      "voice_control": true,
      "price_cny": 2499,
      "weight_kg": 3.5,
      "dimensions_mm": "330×330×80"
    },
    "L5": {
      "model": "L5 旗舰版",
      "suction_power_pa": 5000,
      "battery_capacity_mah": 6500,
      "runtime_min": 180,
      "dust_bin_ml": 600,
      "water_tank_ml": 400,
      "navigation": "LDS+AI视觉",
      "obstacle_avoidance": "AI视觉+结构光",
      "noise_db": 62,
      "max_area_sqm": 200,
      "wifi_band": "2.4GHz/5GHz",
      "app_support": true,
      "voice_control": true,
      "price_cny": 3999,
      "weight_kg": 3.8,
      "dimensions_mm": "340×340×82"
    },
    "L5 Pro": {
      "model": "L5 Pro 顶配版",
      "suction_power_pa": 6000,
      "battery_capacity_mah": 6500,
      "runtime_min": 200,
      "dust_bin_ml": 600,
      "water_tank_ml": 400,
      "navigation": "LDS+AI+3D结构光",
      "obstacle_avoidance": "3D结构光+AI视觉",
      "noise_db": 60,
      "max_area_sqm": 250,
      "wifi_band": "2.4GHz/5GHz",
      "app_support": true,
      "voice_control": true,
      "price_cny": 4999,
      "weight_kg": 3.9,
      "dimensions_mm": "340×340×84"
    }
  },
  "weather": {
    "beijing": {"city": "北京", "temp_c": 28, "humidity": 55, "condition": "晴", "suggestion": "天气晴好，适合开窗通风后让机器人全屋清扫"},
    "shanghai": {"city": "上海", "temp_c": 32, "humidity": 78, "condition": "多云", "suggestion": "湿度较高，建议开启强力模式，注意拖地后地板干燥时间较长"},
    "guangzhou": {"city": "广州", "temp_c": 33, "humidity": 85, "condition": "雷阵雨", "suggestion": "雨天湿度大，如已拖地建议调低水量，防止地板过湿"},
    "shenzhen": {"city": "深圳", "temp_c": 31, "humidity": 80, "condition": "阴", "suggestion": "湿度偏高，HEPA滤网清洁频率建议加倍"},
    "chengdu": {"city": "成都", "temp_c": 26, "humidity": 70, "condition": "阴", "suggestion": "阴天光线不足，机器人视觉导航可能受影响，建议开灯清扫"},
    "hangzhou": {"city": "杭州", "temp_c": 30, "humidity": 72, "condition": "晴", "suggestion": "天气不错，适合全屋深度清扫"}
  },
  "tickets": []
}
```

- [ ] **Step 5: Commit**

```bash
git add data/kb/product_manual.md data/kb/troubleshooting.md data/kb/maintenance.md data/mock/mock_data.json
git commit -m "feat: add mock knowledge base documents and mock data"
```

---

### Task 1.3: 实现本地 Embedding 模块

**Files:**
- Create: `src/embedding/__init__.py`
- Create: `src/embedding/embeddings.py`

- [ ] **Step 1: 创建 src/embedding/__init__.py**

```python
from src.embedding.embeddings import get_embedding_model

__all__ = ["get_embedding_model"]
```

- [ ] **Step 2: 创建 src/embedding/embeddings.py**

```python
"""本地 Embedding 模型加载模块。

使用 BAAI/bge-small-zh-v1.5，512维中文向量模型。
模型首次加载后缓存在模块级变量中，避免重复初始化。
"""

from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from configs.settings import settings

_embedding_model = None


def get_embedding_model() -> HuggingFaceBgeEmbeddings:
    """获取本地 Embedding 模型（单例模式）。

    Returns:
        HuggingFaceBgeEmbeddings: 兼容 LangChain Embeddings 接口的实例
    """
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = HuggingFaceBgeEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": settings.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )

    return _embedding_model
```

- [ ] **Step 3: 验证 Embedding 模型可以加载**

Run the following Python snippet (Ctrl+C to exit after seeing success):

```bash
python -c "from src.embedding.embeddings import get_embedding_model; m = get_embedding_model(); print('Embedding dim:', len(m.embed_query('测试')))"
```

Expected output: `Embedding dim: 512`

Note: First run will download the model (~100MB), this is normal.

- [ ] **Step 4: Commit**

```bash
git add src/embedding/__init__.py src/embedding/embeddings.py
git commit -m "feat: add local embedding module with BGE-small-zh"
```

---

### Task 1.4: 实现文档加载和分块管道

**Files:**
- Create: `src/vectordb/__init__.py`
- Create: `src/vectordb/loader.py`

- [ ] **Step 1: 创建 src/vectordb/__init__.py**

```python
from src.vectordb.loader import DocumentLoader
from src.vectordb.store import VectorStore
from src.vectordb.retriever import KnowledgeRetriever

__all__ = ["DocumentLoader", "VectorStore", "KnowledgeRetriever"]
```

- [ ] **Step 2: 创建 src/vectordb/loader.py**

```python
"""文档加载和分块管道。

支持从目录加载 .md / .txt 文件，使用 RecursiveCharacterTextSplitter
进行递归分块，优先在段落和句子边界切分。
"""

import os
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from configs.settings import settings


class DocumentLoader:
    """从指定目录加载文档并执行分块。

    Usage:
        loader = DocumentLoader()
        docs = loader.load_and_split("data/kb")
        print(f"Loaded {len(docs)} chunks")
    """

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    def load_directory(self, directory_path: str) -> list:
        """从目录中加载所有支持的文档文件。

        Args:
            directory_path: 包含 .md / .txt 文件的目录路径

        Returns:
            list[Document]: LangChain Document 对象列表（未分块）
        """
        if not os.path.isdir(directory_path):
            raise FileNotFoundError(f"目录不存在: {directory_path}")

        loader = DirectoryLoader(
            path=directory_path,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True},
            show_progress=False,
        )

        md_docs = loader.load()

        txt_loader = DirectoryLoader(
            path=directory_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True},
            show_progress=False,
        )

        txt_docs = txt_loader.load()

        all_docs = md_docs + txt_docs

        if not all_docs:
            raise ValueError(f"目录 {directory_path} 中未找到任何 .md 或 .txt 文件")

        return all_docs

    def split_documents(self, documents: list) -> list:
        """将文档列表分块。

        Args:
            documents: LangChain Document 对象列表

        Returns:
            list[Document]: 分块后的 Document 对象列表，每个包含 chunk_id 元数据
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", "。", "；", "，", " ", ""],
        )

        chunks = splitter.split_documents(documents)

        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "unknown")
            chunk.metadata["chunk_id"] = f"{Path(source).stem}_{i}"

        return chunks

    def load_and_split(self, directory_path: str) -> list:
        """加载目录中的文档并分块（一步完成）。

        Args:
            directory_path: 文档目录路径

        Returns:
            list[Document]: 分块并附加元数据后的 Document 列表
        """
        docs = self.load_directory(directory_path)
        chunks = self.split_documents(docs)
        return chunks
```

- [ ] **Step 3: Commit**

```bash
git add src/vectordb/__init__.py src/vectordb/loader.py
git commit -m "feat: add document loader and chunking pipeline"
```

---

### Task 1.5: 实现 ChromaDB 向量存储

**Files:**
- Create: `src/vectordb/store.py`

- [ ] **Step 1: 创建 src/vectordb/store.py**

```python
"""ChromaDB 向量存储模块。

封装 ChromaDB PersistentClient，提供 collection 管理、
文档入库和语义检索功能。
"""

import os

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma

from configs.settings import settings
from src.embedding.embeddings import get_embedding_model


class VectorStore:
    """ChromaDB 向量存储封装。

    Usage:
        store = VectorStore()
        store.ingest_documents(chunks)
        results = store.similarity_search("如何清理滤网", k=4)
    """

    def __init__(self, persist_dir: str | None = None, collection_name: str | None = None):
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self.embedding = get_embedding_model()

        os.makedirs(self.persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self._vectorstore = None

    def _get_or_create_vectorstore(self) -> Chroma:
        """获取或创建 Chroma vectorstore 实例（懒加载）。"""
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding,
                persist_directory=self.persist_dir,
            )
        return self._vectorstore

    def collection_exists(self) -> bool:
        """检查 collection 是否已存在且非空。"""
        try:
            collection = self.client.get_collection(self.collection_name)
            return collection.count() > 0
        except Exception:
            return False

    def ingest_documents(self, documents: list) -> int:
        """将文档列表写入向量库。

        Args:
            documents: LangChain Document 对象列表

        Returns:
            int: 入库的文档数
        """
        if not documents:
            return 0

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding,
            collection_name=self.collection_name,
            persist_directory=self.persist_dir,
        )

        self._vectorstore = vectorstore
        return len(documents)

    def similarity_search(self, query: str, k: int | None = None) -> list:
        """语义相似度检索。

        Args:
            query: 搜索查询文本
            k: 返回的文档数量，默认使用 settings.RETRIEVAL_TOP_K

        Returns:
            list[Document]: 最相关的文档列表
        """
        if k is None:
            k = settings.RETRIEVAL_TOP_K

        vectorstore = self._get_or_create_vectorstore()
        return vectorstore.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int | None = None) -> list:
        """语义检索，附带相似度分数。

        Args:
            query: 搜索查询文本
            k: 返回数量

        Returns:
            list[tuple[Document, float]]: (文档, 相似度分数) 列表
        """
        if k is None:
            k = settings.RETRIEVAL_TOP_K

        vectorstore = self._get_or_create_vectorstore()
        return vectorstore.similarity_search_with_relevance_scores(query, k=k)

    def delete_collection(self) -> None:
        """删除当前 collection。"""
        try:
            self.client.delete_collection(self.collection_name)
            self._vectorstore = None
        except Exception:
            pass

    def get_collection_stats(self) -> dict:
        """获取 collection 统计信息。"""
        try:
            collection = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "count": collection.count(),
                "exists": True,
            }
        except Exception:
            return {"name": self.collection_name, "count": 0, "exists": False}
```

- [ ] **Step 2: Commit**

```bash
git add src/vectordb/store.py
git commit -m "feat: add ChromaDB vector store module"
```

---

### Task 1.6: 实现检索接口 + 端到端验证

**Files:**
- Create: `src/vectordb/retriever.py`

- [ ] **Step 1: 创建 src/vectordb/retriever.py**

```python
"""知识库检索接口。

封装 VectorStore，提供高层检索 API：
- 批量文档入库
- 语义检索（带/不带分数）
- 检索结果格式化
"""

from src.vectordb.loader import DocumentLoader
from src.vectordb.store import VectorStore


class KnowledgeRetriever:
    """知识库高层检索接口。

    Usage:
        retriever = KnowledgeRetriever()
        retriever.ingest_directory("data/kb")
        results = retriever.retrieve("如何清理滤网")
    """

    def __init__(self):
        self.store = VectorStore()
        self.loader = DocumentLoader()

    def ingest_directory(self, directory_path: str = "data/kb", force: bool = False) -> int:
        """从目录加载文档并写入向量库。

        Args:
            directory_path: 文档目录路径
            force: 是否强制重新入库（即使 collection 已存在）

        Returns:
            int: 入库的文档数量
        """
        if self.store.collection_exists() and not force:
            stats = self.store.get_collection_stats()
            return stats["count"]

        chunks = self.loader.load_and_split(directory_path)
        count = self.store.ingest_documents(chunks)
        return count

    def retrieve(self, query: str, top_k: int | None = None) -> list:
        """语义检索相关文档。

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            list[Document]: 相关文档列表
        """
        return self.store.similarity_search(query, k=top_k)

    def retrieve_with_scores(self, query: str, top_k: int | None = None) -> list:
        """语义检索，带相似度分数。

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            list[tuple[Document, float]]: (文档, 分数) 列表
        """
        return self.store.similarity_search_with_score(query, k=top_k)

    def format_results(self, docs: list) -> str:
        """将检索结果格式化为可读文本。

        Args:
            docs: Document 对象列表

        Returns:
            str: 格式化的文本，每条含来源和内容
        """
        if not docs:
            return ""

        parts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "未知来源")
            parts.append(f"[来源{i + 1}: {source}]\n{doc.page_content}")

        return "\n\n---\n\n".join(parts)
```

- [ ] **Step 2: 运行端到端验证**

Run: (this will load docs, chunk, embed, and insert into ChromaDB)

```bash
python -c "
from src.vectordb.retriever import KnowledgeRetriever

# 入库
r = KnowledgeRetriever()
count = r.ingest_directory('data/kb', force=True)
print(f'Ingested {count} chunks')

# 检索测试
results = r.retrieve('E05错误怎么处理')
print(f'\n--- Query: E05错误怎么处理 ---')
for i, doc in enumerate(results):
    print(f'Result {i+1}: {doc.page_content[:150]}...')
    print(f'  Source: {doc.metadata.get(\"source\", \"?\")}')
    print()
"
```

Expected: 成功入库若干 chunks，检索返回 E05 相关的文档内容。

- [ ] **Step 3: Commit**

```bash
git add src/vectordb/retriever.py
git commit -m "feat: add knowledge retriever with end-to-end verification"
```

---

## Phase 2: 工具层 — 8 个 @tool 函数

### Task 2.1: 创建 RAG 知识库检索工具

**Files:**
- Create: `src/tools/__init__.py`
- Create: `src/tools/knowledge_search.py`

- [ ] **Step 1: 创建 src/tools/__init__.py（占位，后续逐步填充）**

```python
"""工具函数层 — 统一注册所有 @tool 工具。"""

# 工具将在各子模块创建后逐个注册到此列表
ALL_TOOLS = []
```

- [ ] **Step 2: 创建 src/tools/knowledge_search.py**

```python
"""知识库语义检索工具。

Agent 调用此工具搜索产品知识库，获取相关文档片段。
"""

from langchain_core.tools import tool

from src.vectordb.retriever import KnowledgeRetriever


@tool
def search_knowledge_base(query: str) -> str:
    """搜索扫地机器人产品知识库，获取使用手册、故障排查、维护指南等相关文档。

    适用场景：
    - 用户询问产品功能、使用方法、参数规格
    - 用户报告故障、错误代码
    - 用户询问保养和维护方法

    Args:
        query: 自然语言搜索查询，尽量具体描述用户问题

    Returns:
        str: 最相关的知识库文档片段（最多 Top-4），含来源标注
    """
    retriever = KnowledgeRetriever()
    docs = retriever.retrieve(query)

    if not docs:
        return "未在知识库中找到相关信息。建议引导用户联系人工客服或提交工单。"

    return retriever.format_results(docs)
```

- [ ] **Step 3: 验证工具可被调用**

```bash
python -c "from src.tools.knowledge_search import search_knowledge_base; print(search_knowledge_base.invoke('如何清理HEPA滤网')[:300])"
```

Expected: 返回滤网清理相关的知识库内容，带 `[来源X: ...]` 标注。

- [ ] **Step 4: Commit**

```bash
git add src/tools/__init__.py src/tools/knowledge_search.py
git commit -m "feat: add RAG knowledge base search tool"
```

---

### Task 2.2: 创建环境感知类工具（天气 + 定位）

**Files:**
- Create: `src/tools/weather.py`
- Create: `src/tools/location.py`

- [ ] **Step 1: 创建 src/tools/weather.py**

```python
"""天气查询工具（Mock）。

查询城市天气信息，为扫地机器人使用场景提供环境建议。
后续可接入真实天气 API（如和风天气、OpenWeatherMap）。
"""

import json
import os

from langchain_core.tools import tool

MOCK_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "mock", "mock_data.json")


def _load_weather_data() -> dict:
    with open(MOCK_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f).get("weather", {})


@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气信息，用于判断是否适合使用扫地机器人。

    适用场景：
    - 用户问「今天适合清扫吗」
    - 用户问「下雨天怎么用扫地机」
    - 需要根据湿度/温度给出清扫建议

    Args:
        city: 城市名称，如"北京"、"上海"、"广州"

    Returns:
        str: 天气信息文本，含温度、湿度、天气状况和使用建议
    """
    weather_data = _load_weather_data()

    city_lower = city.lower().replace("市", "")
    result = weather_data.get(city_lower)

    if not result:
        available = ", ".join(weather_data.keys())
        return f"暂不支持查询「{city}」的天气。当前支持的城市：{available}"

    return (
        f"城市：{result['city']}\n"
        f"温度：{result['temp_c']}°C\n"
        f"湿度：{result['humidity']}%\n"
        f"天气：{result['condition']}\n"
        f"清扫建议：{result['suggestion']}"
    )
```

- [ ] **Step 2: 创建 src/tools/location.py**

```python
"""用户位置定位工具（Mock）。

根据 IP 或用户输入获取城市信息。
后续可接入真实 IP 地理位置 API（如 ip-api.com）。
"""

from langchain_core.tools import tool

# Mock IP → 城市映射
_MOCK_IP_MAP = {
    "127.0.0.1": "北京",
    "192.168.1.1": "上海",
    "10.0.0.1": "深圳",
    "114.114.114.114": "广州",
    "8.8.8.8": "杭州",
    "1.1.1.1": "成都",
}


@tool
def get_user_location(ip_address: str = "") -> str:
    """获取用户当前所在城市位置。

    用于辅助天气查询和本地化服务推荐。
    如果用户未提供 IP，返回默认位置。

    Args:
        ip_address: 用户的 IP 地址（可选），如不提供则返回默认城市

    Returns:
        str: 城市名称（含省份）
    """
    if ip_address and ip_address in _MOCK_IP_MAP:
        city = _MOCK_IP_MAP[ip_address]
        return f"根据 IP（{ip_address}）定位：{city}"

    if ip_address:
        # 未知 IP，返回默认
        return f"IP（{ip_address}）无法识别，默认定位：北京"

    return "默认定位：北京（用户未提供 IP 信息）"
```

- [ ] **Step 3: Commit**

```bash
git add src/tools/weather.py src/tools/location.py
git commit -m "feat: add weather and location tools (mock)"
```

---

### Task 2.3: 创建设备数据类工具（设备状态 + 产品规格 + 故障排查）

**Files:**
- Create: `src/tools/device_status.py`
- Create: `src/tools/product_spec.py`
- Create: `src/tools/troubleshooting.py`

- [ ] **Step 1: 创建 src/tools/device_status.py**

```python
"""设备状态查询工具（Mock）。

获取扫地机器人实时状态：电量、工作模式、耗材寿命、故障代码等。
后续可对接真实 IoT 设备 API。
"""

import json
import os

from langchain_core.tools import tool

MOCK_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "mock", "mock_data.json")


@tool
def get_device_status(device_id: str) -> str:
    """查询指定扫地机器人的实时状态信息。

    返回电量、当前工作模式、耗材寿命、故障代码、上次清扫记录等。
    用户报告故障时优先调用此工具查看设备状态。

    Args:
        device_id: 设备编号，如"RB-001"、"RB-002"等

    Returns:
        str: JSON 格式的设备状态信息
    """
    with open(MOCK_DATA_PATH, "r", encoding="utf-8") as f:
        mock_data = json.load(f)

    devices = mock_data.get("devices", {})
    device = devices.get(device_id)

    if not device:
        available = ", ".join(devices.keys())
        return f"未找到设备「{device_id}」。已知设备：{available}"

    info = {
        "设备编号": device["device_id"],
        "型号": device["model"],
        "昵称": device["nickname"],
        "固件版本": device["firmware_version"],
        "电量": f"{device['battery']}%",
        "工作模式": device["mode"],
        "运行状态": device["status"],
        "HEPA滤网寿命": f"{device['filter_life']}%",
        "边刷寿命": f"{device['side_brush_life']}%",
        "主刷寿命": f"{device['main_brush_life']}%",
        "故障代码": device["error_code"] or "无故障",
        "上次清扫": device["last_clean"],
        "累计清扫面积": f"{device['total_area']}㎡",
        "累计工作时长": f"{device['total_hours']}小时",
        "保修到期": device["warranty_expire"],
    }

    return json.dumps(info, ensure_ascii=False, indent=2)
```

- [ ] **Step 2: 创建 src/tools/product_spec.py**

```python
"""产品规格查询工具（Mock）。

根据产品型号查询详细规格参数，用于售前咨询中的产品对比。
"""

import json
import os

from langchain_core.tools import tool

MOCK_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "mock", "mock_data.json")


@tool
def query_product_spec(model: str, attribute: str = "") -> str:
    """查询灵悦扫地机器人产品的详细规格参数。

    适用场景：
    - 用户对比不同型号（L1/L3/L5/L5 Pro）
    - 用户询问某个具体参数（吸力、续航、噪音等）
    - 售前咨询中的产品推荐

    Args:
        model: 产品型号，如"L1"、"L3"、"L5"、"L5 Pro"
        attribute: 可选，指定查询的属性（如"吸力"、"续航"、"价格"）。
                   为空则返回全部规格。

    Returns:
        str: 产品规格信息文本
    """
    with open(MOCK_DATA_PATH, "r", encoding="utf-8") as f:
        mock_data = json.load(f)

    products = mock_data.get("products", {})

    if model.upper() == "ALL":
        lines = []
        for m, spec in products.items():
            lines.append(f"{spec['model']}：吸力 {spec['suction_power_pa']}Pa | "
                         f"续航 {spec['runtime_min']}min | "
                         f"噪音 {spec['noise_db']}dB | "
                         f"价格 ¥{spec['price_cny']}")
        return "所有型号概览：\n" + "\n".join(lines)

    product = products.get(model)
    if not product:
        available = ", ".join(products.keys())
        return f"未找到型号「{model}」。可选型号：{available}（或使用 'ALL' 查看全部对比）"

    if attribute:
        attr_map = {
            "吸力": f"吸力 {product['suction_power_pa']}Pa",
            "续航": f"续航 {product['runtime_min']}分钟",
            "电池": f"电池容量 {product['battery_capacity_mah']}mAh",
            "尘盒": f"尘盒容量 {product['dust_bin_ml']}ml",
            "水箱": f"水箱容量 {product['water_tank_ml']}ml",
            "噪音": f"噪音 {product['noise_db']}dB",
            "价格": f"参考价格 ¥{product['price_cny']}",
            "导航": f"导航方式 {product['navigation']}",
            "避障": f"避障方式 {product['obstacle_avoidance']}",
            "面积": f"适用面积 ≤{product['max_area_sqm']}㎡",
            "重量": f"重量 {product['weight_kg']}kg",
        }
        if attribute in attr_map:
            return f"{product['model']}：{attr_map[attribute]}"
        return f"未找到属性「{attribute}」，可查询：{', '.join(attr_map.keys())}"

    return (
        f"型号：{product['model']}\n"
        f"吸力：{product['suction_power_pa']}Pa\n"
        f"电池容量：{product['battery_capacity_mah']}mAh\n"
        f"续航：{product['runtime_min']}分钟\n"
        f"尘盒容量：{product['dust_bin_ml']}ml\n"
        f"水箱容量：{product['water_tank_ml']}ml\n"
        f"导航方式：{product['navigation']}\n"
        f"避障方式：{product['obstacle_avoidance']}\n"
        f"噪音：{product['noise_db']}dB\n"
        f"适用面积：≤{product['max_area_sqm']}㎡\n"
        f"WiFi：{product['wifi_band']}\n"
        f"语音控制：{'支持' if product['voice_control'] else '不支持'}\n"
        f"参考价格：¥{product['price_cny']}\n"
        f"重量：{product['weight_kg']}kg\n"
        f"尺寸：{product['dimensions_mm']}"
    )
```

- [ ] **Step 3: 创建 src/tools/troubleshooting.py**

```python
"""故障排查工具（Mock）。

根据故障代码或故障现象返回排查步骤。
同时调用知识库检索获取更详细的排查文档。
"""

from langchain_core.tools import tool

# Mock 故障代码 → 排查步骤映射
_TROUBLESHOOTING_STEPS = {
    "E01": "【E01 无法开机】排查步骤：\n1. 将机器连接到充电底座，充电 30 分钟\n2. 长按电源键 10 秒执行强制重启\n3. 检查电源适配器指示灯是否亮起\n4. 若以上步骤无效，请联系售后维修",
    "E02": "【E02 无法充电】排查步骤：\n1. 用干布清洁机器底部和充电底座的充电触点\n2. 检查电源适配器指示灯是否亮\n3. 尝试更换电源插座\n4. 确认充电底座周围没有金属物体干扰",
    "E03": "【E03 边刷不转】排查步骤：\n1. 将机器翻转，取下边刷\n2. 清理边刷卡槽中的毛发和灰尘\n3. 检查边刷卡扣是否磨损（磨损需更换）\n4. 安装备用边刷测试是否恢复正常",
    "E04": "【E04 主刷不转】排查步骤：\n1. 翻转机器，打开主刷仓盖板\n2. 取出主刷组件，检查是否有异物卡住\n3. 清理刷头两端轴承的缠绕物\n4. 检查主刷皮带是否断裂\n5. 重新安装主刷并确保盖板扣紧",
    "E05": "【E05 边刷缠绕】排查步骤：\n1. 将机器翻转，取下边刷\n2. 用清洁工具上的刀片割断缠绕的毛发/线头\n3. 清理边刷卡槽内的残留物\n4. 重新安装边刷，确认旋转顺畅\n5. 建议每周定期清理边刷以预防此问题",
    "E06": "【E06 主刷缠绕】排查步骤：\n1. 取出主刷组件\n2. 用清洁工具沿刷体沟槽切割缠绕的毛发\n3. 特别注意两端轴承位置，取下轴承清理\n4. 清理滚刷仓内的残留毛发\n5. 全部清理干净后重新安装",
    "E07": "【E07 无法回充】排查步骤：\n1. 检查充电底座电源指示灯是否亮\n2. 用干布清洁机器前方和充电底座的红外传感器窗口\n3. 将机器手动放到基站前方 1 米处，按\"回充\"键测试\n4. 长按充电底座配对键 5 秒重新配对\n5. 确保底座周围左右各 0.5m、前方 1.5m 无遮挡",
    "E08": "【E08 悬崖传感器异常】排查步骤：\n1. 用柔软干布清洁机器底部 4 个悬崖传感器窗口\n2. 检查传感器窗口是否有贴膜未撕或异物遮挡\n3. 避免在深黑色高反光地板上使用\n4. 将机器移到浅色地板重新测试",
    "E09": "【E09 激光头异常】排查步骤：\n1. 检查顶部激光塔是否被物体压住或遮挡\n2. 用手轻轻转动激光头，确认转动顺畅\n3. 用干布擦拭激光塔透明窗口\n4. 关机重启，观察激光头是否开始旋转\n5. 若激光头不转，需联系售后维修",
    "E10": "【E10 风机异常】排查步骤：\n1. 取出尘盒，检查进风口是否有异物堵塞\n2. 清洗或更换 HEPA 滤网（建议 3 个月更换）\n3. 确认尘盒安装到位，密封圈完好\n4. 空转测试：不装尘盒开机，听风机声音是否正常\n5. 若风机不转或有异响，联系售后",
    "E11": "【E11 水箱异常】排查步骤：\n1. 取出水箱，检查是否安装到位\n2. 检查水箱底部出水孔是否堵塞\n3. 用湿布清洁水箱与机器的接触触点\n4. 确认水箱密封圈完好无老化\n5. 如水泵不工作，联系售后更换水箱模块",
    "E12": "【E12 轮子悬空】排查步骤：\n1. 检查机器人是否卡在门槛或家具边缘\n2. 将机器移到平坦地面上\n3. 检查驱动轮是否被异物卡住无法转动\n4. 用手按压驱动轮测试回弹是否正常\n5. 若轮子卡死，联系售后维修",
    "E13": "【E13 电池过热】排查步骤：\n1. 立即停止使用，将机器移到阴凉通风处\n2. 等待至少 30 分钟让电池冷却\n3. 检查电池仓周围是否有灰尘堆积影响散热\n4. 注意使用环境温度不应超过 40°C\n5. 冷却后仍报错，联系售后检测电池",
    "E14": "【E14 传感器校准失败】排查步骤：\n1. 将机器放在平整干净的地面上\n2. 关机后重新开机\n3. 通过\"灵悦智家\"APP → 设备设置 → 传感器校准\n4. 校准过程中不要移动机器\n5. 多次校准失败请联系售后",
    "E15": "【E15 固件更新失败】排查步骤：\n1. 确保 WiFi 连接正常且信号稳定\n2. 确保机器电量 > 30%（建议放在充电底座上）\n3. 重启路由器和机器人后重试\n4. 通过 APP → 设备设置 → 固件升级 → 手动更新\n5. 若多次失败，联系售后获取离线升级包",
}


@tool
def get_troubleshooting_guide(error_code: str) -> str:
    """根据故障代码或故障现象，获取详细的排查步骤指南。

    适用场景：
    - 用户报告错误代码（E01-E15）
    - 用户描述故障现象（如「不充电」「噪音大」「不工作」）

    Args:
        error_code: 故障代码（如"E05"）或故障现象描述（如"无法回充"）

    Returns:
        str: 详细排查步骤
    """
    # 精确匹配错误代码
    code_upper = error_code.upper().strip()
    if code_upper in _TROUBLESHOOTING_STEPS:
        return _TROUBLESHOOTING_STEPS[code_upper]

    # 模糊匹配关键词
    keyword_map = {
        "不充电": "E02", "充不": "E02", "充电": "E02",
        "边刷不转": "E03", "边刷": "E03",
        "主刷不转": "E04", "主刷": "E04",
        "缠绕": "E05", "毛发": "E05",
        "回充": "E07", "回不去": "E07", "充电桩": "E07",
        "悬崖": "E08", "传感器": "E08",
        "激光": "E09", "激光头": "E09",
        "风机": "E10", "吸不": "E10", "吸力": "E10",
        "水箱": "E11", "不出水": "E11", "拖地": "E11",
        "悬空": "E12", "卡住": "E12", "不动": "E12",
        "过热": "E13", "烫": "E13",
        "校准": "E14",
        "固件": "E15", "升级": "E15",
    }

    for keyword, code in keyword_map.items():
        if keyword in error_code:
            return _TROUBLESHOOTING_STEPS[code]

    # 未匹配到，列出所有已知代码
    codes = ", ".join(sorted(_TROUBLESHOOTING_STEPS.keys()))
    return (
        f"未识别到与「{error_code}」匹配的故障代码。\n"
        f"已知故障代码：{codes}\n"
        f"建议：请描述具体故障现象，或告知故障代码以便精准排查。"
    )
```

- [ ] **Step 4: Commit**

```bash
git add src/tools/device_status.py src/tools/product_spec.py src/tools/troubleshooting.py
git commit -m "feat: add device status, product spec, and troubleshooting tools"
```

---

### Task 2.4: 创建外部数据 + 服务闭环工具（网页抓取 + 工单）

**Files:**
- Create: `src/tools/web_scraper.py`
- Create: `src/tools/ticket.py`

- [ ] **Step 1: 创建 src/tools/web_scraper.py**

```python
"""网页内容抓取工具。

抓取外部网页内容并返回文本摘要，用于获取最新产品信息、固件更新日志等。
"""

import re

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool


@tool
def fetch_web_content(url: str) -> str:
    """抓取指定网页的内容并返回纯文本摘要。

    适用场景：
    - 用户询问最新固件更新内容
    - 需要获取官方产品页面信息
    - 查询外部来源的扫地机器人相关信息

    注意：仅抓取公开可访问的网页。

    Args:
        url: 要抓取的网页 URL

    Returns:
        str: 网页文本内容摘要（最多 2000 字符）
    """
    if not url.startswith(("http://", "https://")):
        return f"无效的 URL：{url}。请提供完整的 HTTP/HTTPS 链接。"

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
    except requests.exceptions.Timeout:
        return f"请求超时：{url} 响应时间超过 10 秒，请稍后重试。"
    except requests.exceptions.ConnectionError:
        return f"无法连接到 {url}，请检查 URL 是否正确或网站是否可访问。"
    except requests.exceptions.HTTPError as e:
        return f"请求失败：HTTP {resp.status_code} - {e}"
    except Exception as e:
        return f"抓取失败：{type(e).__name__} - {e}"

    try:
        soup = BeautifulSoup(resp.text, "lxml")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

        if len(text) > 2000:
            text = text[:2000] + "\n\n... [内容被截断，全文共 {} 字符]".format(len(text))

        return f"来源：{url}\n\n{text}" if text.strip() else f"来源：{url}\n\n页面内容为空或无法解析。"

    except Exception as e:
        return f"页面解析失败：{type(e).__name__} - {e}"
```

- [ ] **Step 2: 创建 src/tools/ticket.py**

```python
"""售后工单提交工具（Mock）。

模拟提交售后工单，形成服务闭环。
后续可对接真实工单系统或 CRM。
"""

import json
import os
import uuid
from datetime import datetime

from langchain_core.tools import tool

MOCK_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "mock", "mock_data.json")


@tool
def submit_support_ticket(user_name: str, issue: str, device_id: str = "", contact: str = "") -> str:
    """提交售后支持工单。

    当以下情况时引导用户提交工单：
    - 故障排查步骤无法解决问题
    - 需要专业维修人员上门
    - 保修期内设备故障需要维修
    - 用户要求退换货

    Args:
        user_name: 用户姓名
        issue: 问题描述，尽量详细
        device_id: 设备编号（可选），如"RB-001"
        contact: 联系方式（可选），手机号或邮箱

    Returns:
        str: 工单提交结果，含工单号和预计处理时间
    """
    ticket_id = f"TK-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

    ticket = {
        "ticket_id": ticket_id,
        "user_name": user_name,
        "device_id": device_id or "未提供",
        "issue": issue,
        "contact": contact or "未提供",
        "status": "已提交",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "estimated_response": "2 小时内",
    }

    # 持久化到 mock_data.json
    try:
        with open(MOCK_DATA_PATH, "r", encoding="utf-8") as f:
            mock_data = json.load(f)

        mock_data.setdefault("tickets", []).append(ticket)

        with open(MOCK_DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(mock_data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return (
        f"✅ 工单提交成功！\n"
        f"工单编号：{ticket_id}\n"
        f"处理状态：{ticket['status']}\n"
        f"提交时间：{ticket['created_at']}\n"
        f"预计响应：{ticket['estimated_response']}\n"
        f"设备编号：{device_id or '未提供'}\n\n"
        f"我们的售后团队将在 2 小时内通过您提供的联系方式与您联系。"
        f"如需紧急支持，请拨打服务热线：400-888-XXXX"
    )
```

- [ ] **Step 3: Commit**

```bash
git add src/tools/web_scraper.py src/tools/ticket.py
git commit -m "feat: add web scraper and support ticket tools"
```

---

### Task 2.5: 完成工具注册

**Files:**
- Modify: `src/tools/__init__.py`

- [ ] **Step 1: 更新 src/tools/__init__.py**

Replace entire file content:

```python
"""工具函数层 — 统一注册所有 @tool 工具。

每个工具都是独立的 LangChain @tool 函数，可单独测试。
通过 ALL_TOOLS 列表统一注册到 Agent。
"""

from src.tools.knowledge_search import search_knowledge_base
from src.tools.weather import get_weather
from src.tools.location import get_user_location
from src.tools.device_status import get_device_status
from src.tools.web_scraper import fetch_web_content
from src.tools.product_spec import query_product_spec
from src.tools.troubleshooting import get_troubleshooting_guide
from src.tools.ticket import submit_support_ticket

# 全部 8 个工具
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

# 场景化工具子集
PRESALE_TOOLS = [
    search_knowledge_base,
    query_product_spec,
    fetch_web_content,
    get_user_location,
]

AFTERSALE_TOOLS = [
    search_knowledge_base,
    get_device_status,
    get_troubleshooting_guide,
    submit_support_ticket,
    fetch_web_content,
]

MAINTENANCE_TOOLS = [
    search_knowledge_base,
    get_device_status,
    get_troubleshooting_guide,
    get_weather,
    get_user_location,
]


def get_tools_for_scenario(scenario: str = "default") -> list:
    """根据业务场景返回不同的工具集合。

    Args:
        scenario: "presale" | "aftersale" | "maintenance" | "default"

    Returns:
        list: 对应场景的工具列表
    """
    scenario_map = {
        "presale": PRESALE_TOOLS,
        "aftersale": AFTERSALE_TOOLS,
        "maintenance": MAINTENANCE_TOOLS,
        "default": ALL_TOOLS,
    }
    return scenario_map.get(scenario, ALL_TOOLS)
```

- [ ] **Step 2: 验证所有工具可正常导入和调用**

```bash
python -c "
from src.tools import ALL_TOOLS
print(f'Total tools registered: {len(ALL_TOOLS)}')
for t in ALL_TOOLS:
    print(f'  - {t.name}: {t.description[:60]}...')
"
```

Expected output: 8 tools listed with names and descriptions.

- [ ] **Step 3: Commit**

```bash
git add src/tools/__init__.py
git commit -m "feat: complete tool registration with scenario subsets"
```

---

## Phase 3: Agent 层 — ReAct 引擎 + Prompt + Memory

### Task 3.1: 创建 System Prompt 模板

**Files:**
- Create: `src/agent/__init__.py`
- Create: `src/agent/prompt.py`

- [ ] **Step 1: 创建 src/agent/__init__.py**

```python
from src.agent.agent import CustomerServiceAgent
from src.agent.memory import ConversationManager
from src.agent.prompt import get_system_prompt_for_scenario

__all__ = ["CustomerServiceAgent", "ConversationManager", "get_system_prompt_for_scenario"]
```

- [ ] **Step 2: 创建 src/agent/prompt.py**

```python
"""System Prompt 模板模块。

定义 Agent 的角色、行为规范和场景化变体。
支持售前咨询、售后支持、日常维护三个场景的动态切换。
"""

SYSTEM_PROMPT_BASE = """你是「小扫」，一位专业友好的扫地机器人智能客服助手。

## 你的身份
- 你是灵悦科技（LingYue Tech）官方智能客服
- 你服务于购买了灵悦扫地机器人的用户
- 你对产品充满热情，但始终保持专业和诚实

## 你的能力
你可以通过调用以下工具来获取信息并帮助用户：
- **search_knowledge_base**: 搜索产品知识库（使用手册、故障排查、维护指南）
- **get_device_status**: 查询用户设备的实时状态（电量、模式、耗材寿命、故障代码）
- **get_weather**: 查询城市天气，给出清扫建议
- **get_user_location**: 获取用户所在城市位置
- **query_product_spec**: 查询产品型号的详细规格参数
- **get_troubleshooting_guide**: 获取故障代码对应的排查步骤
- **fetch_web_content**: 获取外部网页的最新信息（如固件更新日志）
- **submit_support_ticket**: 为用户提交售后工单

## 工作原则
1. **先理解，再行动**：分析用户的真实意图，判断是售前咨询、故障排查还是日常维护
2. **工具优先，不编造**：涉及事实信息（产品参数、故障代码、天气等）必须调用工具获取，绝不自编
3. **整合信息，通俗输出**：将工具返回的信息翻译成用户能听懂的人话，多用分点、加粗和友好语气
4. **主动建议**：不只回答问题，还要根据上下文主动给出关联建议（如天气不好时建议调整清扫计划）
5. **诚实透明**：工具信息不足以回答时，诚实告知并主动建议提交工单或联系人工客服
6. **简洁高效**：回答直击要害，不啰嗦。对于简单的寒暄（你好、谢谢），友好回应即可

## 回复格式建议
- 涉及步骤时用编号列表
- 关键信息用**加粗**突出
- 故障排查先给出结论再给步骤
- 售前对比用简洁的表格或要点
"""

PRESALE_PROMPT = SYSTEM_PROMPT_BASE + """
## 当前场景：售前咨询
用户正在考虑购买扫地机器人，你需要：
- 主动了解用户需求（户型大小、是否有宠物、预算等）
- 推荐合适型号并解释为什么
- 对比不同型号的差异时，优先使用 query_product_spec 工具
- 用通俗语言解释技术参数的实际意义（如"4000Pa 吸力意味着可以吸起黄豆大的颗粒"）
- 告知当前促销活动和购买渠道
"""

AFTERSALE_PROMPT = SYSTEM_PROMPT_BASE + """
## 当前场景：售后支持
用户的产品出现了问题，你需要：
- 首先共情安抚用户情绪
- 优先用 get_device_status 查看设备状态
- 有故障代码时，先用 get_troubleshooting_guide 获取排查步骤
- 排查步骤按难度排序（从简单到复杂）
- 如果工具提供的排查步骤无法解决，主动建议 submit_support_ticket
- 告知用户保修信息（从设备状态中获取保修到期日）
"""

MAINTENANCE_PROMPT = SYSTEM_PROMPT_BASE + """
## 当前场景：日常维护
用户询问产品的保养和日常使用技巧，你需要：
- 优先用 search_knowledge_base 搜索维护指南
- 结合设备状态中的耗材寿命给出更换建议
- 根据季节和天气给出针对性的保养提醒
- 主动告知耗材购买渠道和价格
"""

# 场景关键词 → Prompt 映射
SCENARIO_PROMPTS = {
    "presale": PRESALE_PROMPT,
    "aftersale": AFTERSALE_PROMPT,
    "maintenance": MAINTENANCE_PROMPT,
    "default": SYSTEM_PROMPT_BASE,
}


def get_system_prompt_for_scenario(scenario: str = "default") -> str:
    """根据场景获取对应的 System Prompt。

    Args:
        scenario: "presale" | "aftersale" | "maintenance" | "default"

    Returns:
        str: 对应的 System Prompt
    """
    return SCENARIO_PROMPTS.get(scenario, SYSTEM_PROMPT_BASE)
```

- [ ] **Step 3: Commit**

```bash
git add src/agent/__init__.py src/agent/prompt.py
git commit -m "feat: add system prompt templates with scenario variants"
```

---

### Task 3.2: 创建对话记忆管理器

**Files:**
- Create: `src/agent/memory.py`

- [ ] **Step 1: 创建 src/agent/memory.py**

```python
"""对话历史管理模块。

基于 LangChain InMemoryChatMessageHistory，
支持多会话隔离、最近 N 轮保留和手动清除。
"""

from langchain_community.chat_message_histories import ChatMessageHistory


class ConversationManager:
    """会话对话历史管理器。

    Usage:
        cm = ConversationManager()
        history = cm.get_history("session-123")
        history.add_user_message("你好")
        history.add_ai_message("你好！有什么可以帮助你的？")
    """

    def __init__(self, max_history_per_session: int = 20):
        """初始化。

        Args:
            max_history_per_session: 每个会话保留的最大消息数（user+ai 成对计数）
        """
        self._store: dict[str, ChatMessageHistory] = {}
        self._max_history = max_history_per_session

    def get_history(self, session_id: str) -> ChatMessageHistory:
        """获取或创建会话历史。

        Args:
            session_id: 会话唯一标识

        Returns:
            ChatMessageHistory: 该会话的消息历史
        """
        if session_id not in self._store:
            self._store[session_id] = ChatMessageHistory()
        return self._store[session_id]

    def get_formatted_messages(self, session_id: str, system_prompt: str) -> list:
        """获取格式化的消息列表，用于传入 Agent。

        自动在开头插入 system message，然后拼接历史消息。

        Args:
            session_id: 会话 ID
            system_prompt: System Prompt 文本

        Returns:
            list: 格式化的消息列表 [("system", prompt), ("human", ...), ("ai", ...), ...]
        """
        history = self.get_history(session_id)
        messages = [("system", system_prompt)]

        for msg in history.messages[-self._max_history:]:
            role = "human" if msg.type == "human" else "ai"
            messages.append((role, msg.content))

        return messages

    def add_user_message(self, session_id: str, content: str) -> None:
        """添加用户消息到历史。

        Args:
            session_id: 会话 ID
            content: 用户消息文本
        """
        history = self.get_history(session_id)
        history.add_user_message(content)

    def add_ai_message(self, session_id: str, content: str) -> None:
        """添加 AI 回复到历史。

        Args:
            session_id: 会话 ID
            content: AI 回复文本
        """
        history = self.get_history(session_id)
        history.add_ai_message(content)

    def clear(self, session_id: str) -> None:
        """清除指定会话的历史。

        Args:
            session_id: 会话 ID
        """
        if session_id in self._store:
            del self._store[session_id]

    def get_session_count(self) -> int:
        """获取当前活跃会话数。"""
        return len(self._store)

    def get_message_count(self, session_id: str) -> int:
        """获取指定会话的消息数。"""
        history = self.get_history(session_id)
        return len(history.messages)
```

- [ ] **Step 2: Commit**

```bash
git add src/agent/memory.py
git commit -m "feat: add conversation memory manager"
```

---

### Task 3.3: 实现 ReAct Agent 引擎

**Files:**
- Create: `src/agent/agent.py`

- [ ] **Step 1: 创建 src/agent/agent.py**

```python
"""ReAct Agent 引擎。

基于 LangGraph create_react_agent，封装 Agent 创建、同步/异步执行和流式输出。
"""

from langgraph.prebuilt import create_react_agent

from configs.settings import settings
from src.agent.memory import ConversationManager
from src.agent.prompt import get_system_prompt_for_scenario
from src.llm.chat_model import get_chat_model
from src.tools import ALL_TOOLS

# 全局实例（延迟初始化）
_agent_instance = None
_memory_manager = None


def get_memory_manager() -> ConversationManager:
    """获取全局 ConversationManager 实例。"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = ConversationManager()
    return _memory_manager


def get_agent():
    """获取全局 ReAct Agent 实例（单例模式）。

    在首次调用时自动初始化 LLM + 绑定工具 + 创建 Agent。

    Returns:
        CompiledGraph: LangGraph Agent
    """
    global _agent_instance

    if _agent_instance is None:
        llm = get_chat_model()
        _agent_instance = create_react_agent(
            model=llm,
            tools=ALL_TOOLS,
        )

    return _agent_instance


class CustomerServiceAgent:
    """扫地机器人智能客服 Agent 封装。

    Usage:
        agent = CustomerServiceAgent()

        # 同步调用
        result = agent.chat("我的扫地机报E05错误怎么办？", session_id="abc")

        # 异步流式
        async for event in agent.astream_chat("如何清理滤网？", session_id="abc"):
            print(event)
    """

    def __init__(self):
        self._agent = None
        self._memory = get_memory_manager()

    @property
    def agent(self):
        if self._agent is None:
            self._agent = get_agent()
        return self._agent

    def chat(self, query: str, session_id: str = "default", scenario: str = "default") -> dict:
        """同步对话 — 返回完整结果。

        Args:
            query: 用户问题
            session_id: 会话 ID
            scenario: 业务场景 "presale" | "aftersale" | "maintenance" | "default"

        Returns:
            dict: {"answer": str, "session_id": str}
        """
        system_prompt = get_system_prompt_for_scenario(scenario)
        self._memory.add_user_message(session_id, query)

        messages = self._memory.get_formatted_messages(session_id, system_prompt)

        config = {"recursion_limit": settings.AGENT_MAX_ITERATIONS * 2}
        result = self.agent.invoke({"messages": messages}, config=config)

        answer = result["messages"][-1].content
        self._memory.add_ai_message(session_id, answer)

        return {"answer": answer, "session_id": session_id}

    async def astream_chat(self, query: str, session_id: str = "default", scenario: str = "default"):
        """异步流式对话 — 逐事件返回。

        使用 astream_events API 获取：
        - on_chat_model_stream: LLM 输出的每个 token
        - on_tool_start: 工具调用开始
        - on_tool_end: 工具调用完成

        Args:
            query: 用户问题
            session_id: 会话 ID
            scenario: 业务场景

        Yields:
            dict: {"type": "token"|"tool_start"|"tool_end"|"done", "data": str}
        """
        system_prompt = get_system_prompt_for_scenario(scenario)
        self._memory.add_user_message(session_id, query)

        messages = self._memory.get_formatted_messages(session_id, system_prompt)

        config = {"recursion_limit": settings.AGENT_MAX_ITERATIONS * 2}
        full_answer = ""

        async for event in self.agent.astream_events(
            {"messages": messages},
            config=config,
            version="v2",
        ):
            kind = event.get("event", "")

            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    full_answer += chunk.content
                    yield {"type": "token", "data": chunk.content}

            elif kind == "on_tool_start":
                tool_name = event.get("name", "unknown")
                yield {"type": "tool_start", "data": tool_name}

            elif kind == "on_tool_end":
                tool_name = event.get("name", "unknown")
                yield {"type": "tool_end", "data": tool_name}

        # 保存完整回答到历史
        if full_answer:
            self._memory.add_ai_message(session_id, full_answer)

        yield {"type": "done", "data": ""}

    def clear_session(self, session_id: str) -> None:
        """清除指定会话的历史。"""
        self._memory.clear(session_id)
```

- [ ] **Step 2: Commit**

```bash
git add src/agent/agent.py
git commit -m "feat: implement ReAct agent engine with LangGraph"
```

---

## Phase 4: 中间件 + API — FastAPI + SSE + Callbacks

### Task 4.1: 实现中间件链（日志审计 + 性能监控 + Prompt切换）

**Files:**
- Create: `src/middleware/__init__.py`
- Create: `src/middleware/logging_mw.py`
- Create: `src/middleware/monitor_mw.py`
- Create: `src/middleware/prompt_switch.py`

- [ ] **Step 1: 创建 src/middleware/logging_mw.py**

```python
"""日志审计 Callback Handler。

基于 LangChain BaseCallbackHandler，记录每次 LLM 调用和工具调用的完整审计日志。
日志以 JSONL 格式写入 logs/audit.jsonl。
"""

import json
import os
import time
from datetime import datetime

from langchain_core.callbacks import BaseCallbackHandler


class AuditLoggingHandler(BaseCallbackHandler):
    """请求日志审计处理器。

    记录事件：
    - on_llm_start: LLM 调用开始（prompt 摘要 + 时间戳）
    - on_tool_start: 工具调用开始（工具名 + 输入）
    - on_tool_end: 工具调用结束（输出摘要）
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._log_file = os.path.join(log_dir, "audit.jsonl")

    def _write(self, entry: dict) -> None:
        entry["timestamp"] = datetime.now().isoformat()
        try:
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs) -> None:
        prompt_preview = prompts[0][:200] if prompts else ""
        self._write({
            "event": "llm_request",
            "model": serialized.get("kwargs", {}).get("model", "unknown"),
            "prompt_preview": prompt_preview,
            "prompt_length": len(prompts[0]) if prompts else 0,
        })

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
        self._write({
            "event": "tool_call",
            "tool": serialized.get("name", "unknown"),
            "input": str(input_str)[:500],
        })

    def on_tool_end(self, output: str, **kwargs) -> None:
        self._write({
            "event": "tool_result",
            "output_preview": str(output)[:500],
        })

    def on_llm_end(self, response, **kwargs) -> None:
        output = ""
        if hasattr(response, "generations") and response.generations:
            gen = response.generations[0][0]
            if hasattr(gen, "message"):
                output = gen.message.content or ""
            elif hasattr(gen, "text"):
                output = gen.text or ""

        self._write({
            "event": "llm_response",
            "response_preview": output[:500],
            "response_length": len(output),
        })
```

- [ ] **Step 2: 创建 src/middleware/monitor_mw.py**

```python
"""性能监控 Callback Handler。

基于 LangChain BaseCallbackHandler，监控全链路性能指标：
- 首字延迟（第一个 streaming token 的时间）
- 总耗时
- Token 使用量
- 工具调用次数
"""

import logging
import time

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)


class PerformanceMonitor(BaseCallbackHandler):
    """全链路性能监控处理器。

    在每次 Agent 执行中追踪：
    - latency_ms: 从请求开始到完整响应的总耗时
    - ttft_ms: 首字延迟 (Time To First Token)
    - total_tokens: LLM 消耗的 token 总数
    - tool_calls: 本次请求中工具调用的次数
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """重置本次请求的监控状态。"""
        self.start_time: float = 0
        self.first_token_time: float = 0
        self.first_token_recorded: bool = False
        self.total_tokens: int = 0
        self.tool_call_count: int = 0

    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs) -> None:
        self.reset()
        self.start_time = time.time()

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs) -> None:
        if not self.first_token_recorded:
            self.first_token_recorded = True

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if not self.first_token_time:
            self.first_token_time = time.time()

    def on_llm_end(self, response, **kwargs) -> None:
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            self.total_tokens += usage.get("total_tokens", 0)

        if not self.first_token_time:
            self.first_token_time = time.time()

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
        self.tool_call_count += 1

    def on_chain_end(self, outputs: dict, **kwargs) -> None:
        if not self.start_time:
            return

        end_time = time.time()
        latency_ms = (end_time - self.start_time) * 1000
        ttft_ms = ((self.first_token_time - self.start_time) * 1000) if self.first_token_time else 0

        logger.info(
            "Performance: latency=%.0fms, ttft=%.0fms, tokens=%d, tools=%d",
            latency_ms,
            ttft_ms,
            self.total_tokens,
            self.tool_call_count,
        )
```

- [ ] **Step 3: 创建 src/middleware/prompt_switch.py**

```python
"""动态 Prompt 切换模块。

根据用户问题中的关键词自动检测业务场景：
- presale: 售前咨询
- aftersale: 售后支持
- maintenance: 日常维护
- default: 通用

后续可升级为 LLM 意图分类器以获得更高准确率。
"""

# 场景关键词库
SCENARIO_KEYWORDS = {
    "presale": [
        "购买", "多少钱", "哪个型号", "对比", "推荐", "有什么区别",
        "哪款好", "性价比", "值得买", "价格", "贵吗", "适合",
        "参数", "配置", "规格", "多大面积", "续航", "噪音大吗",
        "能扫干净吗", "好不好用", "效果怎么样", "新款", "区别",
    ],
    "aftersale": [
        "坏了", "报错", "故障", "不工作", "报修", "维修", "售后",
        "E0", "E1", "错误代码", "异常", "不动了", "不转了",
        "不充电", "回不去", "不能回充", "卡住", "异响", "噪音大",
        "漏水", "不出水", "不吸", "吸力下降", "连不上", "配网失败",
        "APP连接", "地图丢了", "保修", "退换", "退货",
    ],
    "maintenance": [
        "保养", "清理", "更换滤网", "耗材", "怎么清洗", "多久换",
        "清洁", "维护", "滤网", "边刷", "主刷", "拖布",
        "尘盒", "传感器", "如何清洗", "怎么清理", "多长时间",
        "固件", "升级", "更新", "版本",
    ],
}


def detect_scenario(user_query: str) -> str:
    """根据用户查询文本检测业务场景。

    Args:
        user_query: 用户的原始问题文本

    Returns:
        str: "presale" | "aftersale" | "maintenance" | "default"
    """
    query_lower = user_query.lower()

    scores = {}
    for scenario, keywords in SCENARIO_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            scores[scenario] = score

    if not scores:
        return "default"

    return max(scores, key=scores.get)
```

- [ ] **Step 4: 创建 src/middleware/__init__.py**

```python
"""中间件链模块。

基于 LangChain BaseCallbackHandler 实现的可插拔中间件体系。
执行顺序：AuditLogging → PromptSwitch → Agent → PerformanceMonitor
"""

from src.middleware.logging_mw import AuditLoggingHandler
from src.middleware.monitor_mw import PerformanceMonitor
from src.middleware.prompt_switch import detect_scenario

# Callback handlers 链
CALLBACK_HANDLERS = [
    AuditLoggingHandler(),
    PerformanceMonitor(),
]

__all__ = ["AuditLoggingHandler", "PerformanceMonitor", "detect_scenario", "CALLBACK_HANDLERS"]
```

- [ ] **Step 5: Commit**

```bash
git add src/middleware/__init__.py src/middleware/logging_mw.py src/middleware/monitor_mw.py src/middleware/prompt_switch.py
git commit -m "feat: implement middleware chain (audit, monitor, prompt switch)"
```

---

### Task 4.2: 创建 FastAPI 数据模型和路由

**Files:**
- Create: `src/api/__init__.py`
- Create: `src/api/schemas.py`
- Create: `src/api/routes.py`

- [ ] **Step 1: 创建 src/api/__init__.py**

```python
from src.api.routes import router
from src.api.schemas import ChatRequest, ChatResponse, HealthResponse

__all__ = ["router", "ChatRequest", "ChatResponse", "HealthResponse"]
```

- [ ] **Step 2: 创建 src/api/schemas.py**

```python
"""API 数据模型 (Pydantic)。"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """对话请求模型。"""
    session_id: str = Field(..., description="会话唯一标识", min_length=1)
    query: str = Field(..., description="用户问题", min_length=1, max_length=2000)
    user_id: str | None = Field(None, description="用户 ID（可选）")
    device_id: str | None = Field(None, description="关联设备编号（可选）")


class ChatResponse(BaseModel):
    """对话响应模型（非流式）。"""
    answer: str = Field(..., description="Agent 的完整回答")
    session_id: str = Field(..., description="会话 ID")
    scenario: str = Field("default", description="检测到的业务场景")


class HealthResponse(BaseModel):
    """健康检查响应模型。"""
    status: str = Field(..., description="服务状态")
    version: str = Field(..., description="服务版本")
    model: str = Field(..., description="当前使用的 LLM 模型")
    vectordb_status: str = Field(..., description="向量库状态：ready / empty / error")


class ErrorResponse(BaseModel):
    """错误响应模型。"""
    error: str = Field(..., description="错误描述")
    detail: str | None = Field(None, description="详细错误信息")
```

- [ ] **Step 3: 创建 src/api/routes.py**

```python
"""FastAPI 路由定义。

端点：
- GET  /api/v1/health       — 健康检查
- POST /api/v1/chat          — 同步对话
- POST /api/v1/chat/stream   — SSE 流式对话
"""

import json
import logging
import traceback

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from configs.settings import settings
from src.agent.agent import CustomerServiceAgent
from src.api.schemas import ChatRequest, ChatResponse, HealthResponse
from src.middleware.prompt_switch import detect_scenario
from src.vectordb.store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

# 全局 Agent 实例
_agent = CustomerServiceAgent()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查 — 返回服务和依赖状态。"""
    try:
        store = VectorStore()
        stats = store.get_collection_stats()
        if stats["count"] > 0:
            vectordb_status = "ready"
        elif stats["exists"]:
            vectordb_status = "empty"
        else:
            vectordb_status = "empty"
    except Exception:
        vectordb_status = "error"

    return HealthResponse(
        status="running",
        version="1.0.0",
        model=settings.DEEPSEEK_MODEL,
        vectordb_status=vectordb_status,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """同步对话 — 返回完整 Agent 回答。"""
    try:
        scenario = detect_scenario(request.query)
        result = _agent.chat(
            query=request.query,
            session_id=request.session_id,
            scenario=scenario,
        )
        return ChatResponse(
            answer=result["answer"],
            session_id=result["session_id"],
            scenario=scenario,
        )
    except Exception as e:
        logger.error(f"Chat error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """SSE 流式对话 — 逐 token 推送 Agent 回答。"""
    scenario = detect_scenario(request.query)

    async def event_generator():
        try:
            async for event in _agent.astream_chat(
                query=request.query,
                session_id=request.session_id,
                scenario=scenario,
            ):
                event_type = event["type"]
                data = event["data"]

                if event_type == "token":
                    yield {"event": "token", "data": data}
                elif event_type == "tool_start":
                    yield {"event": "tool_start", "data": data}
                elif event_type == "tool_end":
                    yield {"event": "tool_end", "data": data}
                elif event_type == "done":
                    yield {"event": "done", "data": ""}

        except Exception as e:
            logger.error(f"Stream error: {traceback.format_exc()}")
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())
```

- [ ] **Step 4: Commit**

```bash
git add src/api/__init__.py src/api/schemas.py src/api/routes.py
git commit -m "feat: add FastAPI routes with SSE streaming support"
```

---

### Task 4.3: 创建 FastAPI 服务入口

**Files:**
- Create: `server.py`

- [ ] **Step 1: 创建 server.py**

```python
"""FastAPI 服务入口。

启动命令：
    uvicorn server:app --reload --port 8000

访问文档：
    http://localhost:8000/docs
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from configs.settings import settings
from src.api.routes import router

# 日志配置
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例。"""
    app = FastAPI(
        title="小扫 · 智能客服 API",
        description="基于 ReAct Agent 的扫地机器人智能客服系统",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — 允许 Streamlit 跨域访问
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    app.include_router(router)

    # 生命周期事件
    @app.on_event("startup")
    async def startup():
        """应用启动初始化。"""
        logger.info("=" * 50)
        logger.info(f"  App: {settings.APP_NAME}")
        logger.info(f"  Model: {settings.DEEPSEEK_MODEL}")
        logger.info(f"  Embedding: {settings.EMBEDDING_MODEL}")
        logger.info(f"  Docs: http://localhost:8000/docs")
        logger.info("=" * 50)

        # 预加载 Embedding 模型（首次加载会下载）
        try:
            from src.embedding.embeddings import get_embedding_model
            get_embedding_model()
            logger.info("Embedding model loaded.")
        except Exception as e:
            logger.warning(f"Embedding model load failed: {e}")

        # 检查向量库状态
        try:
            from src.vectordb.store import VectorStore
            store = VectorStore()
            stats = store.get_collection_stats()
            logger.info(f"VectorDB: collection='{stats['name']}', count={stats['count']}")
        except Exception as e:
            logger.warning(f"VectorDB init failed: {e}")

    @app.on_event("shutdown")
    async def shutdown():
        logger.info("Service shutting down.")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
```

- [ ] **Step 2: Commit**

```bash
git add server.py
git commit -m "feat: add FastAPI server entry point with lifecycle hooks"
```

---

## Phase 5: 前端 — Streamlit UI + 流式消费

### Task 5.1: 创建 Streamlit 聊天 UI

**Files:**
- Create: `app.py`

- [ ] **Step 1: 创建 app.py**

```python
"""Streamlit 聊天界面。

扫地机器人智能客服前端，提供：
- 流式对话渲染（打字机效果）
- 工具调用过程可视化
- 会话管理和状态检查

启动命令：
    streamlit run app.py
"""

import json
import uuid

import requests
import streamlit as st

# ── 页面配置 ──
st.set_page_config(
    page_title="小扫 · 智能客服",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000/api/v1"


# ── 样式 ──
st.markdown("""
<style>
    .stChatMessage { padding: 1rem; }
    .tool-call-box {
        background: #1a1a2e;
        border-left: 3px solid #4fc3f7;
        padding: 6px 12px;
        border-radius: 4px;
        margin: 4px 0;
        font-size: 0.85em;
        color: #aaa;
    }
</style>
""", unsafe_allow_html=True)


# ── 侧边栏 ──
with st.sidebar:
    st.title("🤖 小扫 · 智能客服")
    st.caption("灵悦扫地机器人专属助手")

    st.divider()

    # 新对话按钮
    if st.button("✨ 新对话", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # 关于
    with st.expander("ℹ️ 关于小扫"):
        st.markdown("""
        **小扫** 是灵悦科技官方智能客服助手。

        我可以帮你：
        - 📖 查询产品信息和参数
        - 🔧 排查设备故障
        - 🧹 提供维护保养建议
        - 🌤 根据天气给出清扫建议
        - 📝 提交售后工单

        支持型号：L1 / L3 / L5 / L5 Pro
        """)

    # 系统状态
    with st.expander("📊 系统状态"):
        try:
            resp = requests.get(f"{API_BASE}/health", timeout=5)
            if resp.status_code == 200:
                health = resp.json()
                st.success(f"🟢 服务运行中")
                st.info(f"模型: {health['model']}")
                st.info(f"向量库: {health['vectordb_status']}")
            else:
                st.error("⚠️ 服务异常")
        except Exception:
            st.error("🔴 无法连接到后端服务")
            st.caption("请确保已启动: uvicorn server:app --port 8000")


# ── 初始化会话状态 ──
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── 渲染历史消息 ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── 输入框 ──
if prompt := st.chat_input("输入你的问题，比如「E05错误怎么处理」..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用流式 API
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        tool_statuses = {}

        try:
            with requests.post(
                f"{API_BASE}/chat/stream",
                json={
                    "session_id": st.session_state.session_id,
                    "query": prompt,
                },
                stream=True,
                timeout=120,
            ) as resp:
                if resp.status_code != 200:
                    placeholder.error(f"服务错误 (HTTP {resp.status_code})")
                else:
                    current_event = None

                    for line in resp.iter_lines():
                        if not line:
                            continue

                        decoded = line.decode("utf-8")

                        if decoded.startswith("event:"):
                            current_event = decoded[6:].strip()
                        elif decoded.startswith("data:"):
                            data = decoded[5:].strip()

                            if current_event == "token":
                                full_response += data
                                placeholder.markdown(full_response + "▌")

                            elif current_event == "tool_start":
                                tool_statuses[data] = st.status(
                                    f"🔧 正在调用: {data}",
                                    expanded=False,
                                )

                            elif current_event == "tool_end":
                                if data in tool_statuses:
                                    tool_statuses[data].update(
                                        label=f"✅ 完成调用: {data}",
                                        state="complete",
                                        expanded=False,
                                    )

                            elif current_event == "error":
                                try:
                                    err = json.loads(data)
                                    placeholder.error(f"出错了: {err.get('error', data)}")
                                except Exception:
                                    placeholder.error(f"出错了: {data}")

                            elif current_event == "done":
                                placeholder.markdown(full_response)

        except requests.exceptions.ConnectionError:
            placeholder.error("无法连接到后端服务。请确保已启动 FastAPI（uvicorn server:app --port 8000）")
            full_response = "⚠️ 服务连接失败，请检查后端是否启动。"
        except requests.exceptions.Timeout:
            placeholder.error("请求超时，请稍后重试。")
            full_response = "⚠️ 请求超时，请稍后重试。"
        except Exception as e:
            placeholder.error(f"发生异常: {e}")
            full_response = f"⚠️ 发生异常: {e}"

        if full_response:
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
            })
```

- [ ] **Step 2: Commit**

```bash
git add app.py
git commit -m "feat: add Streamlit chat UI with SSE streaming"
```

---

## Phase 6: 集成测试 + 优化

### Task 6.1: 端到端集成测试

- [ ] **Step 1: 确保向量库有数据**

```bash
python -c "
from src.vectordb.retriever import KnowledgeRetriever
r = KnowledgeRetriever()
count = r.ingest_directory('data/kb', force=True)
print(f'VectorDB ready: {count} chunks')
"
```

- [ ] **Step 2: 启动 FastAPI 服务（后台）**

In a terminal:
```bash
uvicorn server:app --reload --port 8000
```

Wait for output: `Uvicorn running on http://127.0.0.1:8000`

- [ ] **Step 3: 测试健康检查端点**

```bash
curl http://localhost:8000/api/v1/health
```

Expected: JSON response with status="running", model info, vectordb_status.

- [ ] **Step 4: 测试同步对话端点**

Using curl or Python:
```bash
python -c "
import requests, json
resp = requests.post('http://localhost:8000/api/v1/chat', json={
    'session_id': 'test-001',
    'query': 'E05错误怎么处理？'
})
print(json.dumps(resp.json(), ensure_ascii=False, indent=2))
"
```

Expected: JSON with "answer" containing E05 troubleshooting steps and "scenario" likely "aftersale".

- [ ] **Step 5: 测试流式对话端点**

```bash
python -c "
import requests, json
resp = requests.post('http://localhost:8000/api/v1/chat/stream', json={
    'session_id': 'test-002',
    'query': 'L5和L3有什么区别？'
}, stream=True)
for line in resp.iter_lines():
    if line:
        print(line.decode())
"
```

Expected: SSE event stream with token/tool_start/tool_end/done events.

- [ ] **Step 6: 测试多轮对话记忆**

```bash
python -c "
import requests, json
sid = 'test-memory-001'
# 第一轮
r1 = requests.post('http://localhost:8000/api/v1/chat', json={'session_id': sid, 'query': '我的设备RB-002状态怎么样？'})
print('Round 1:', r1.json()['answer'][:200])
# 第二轮（依赖记忆）
r2 = requests.post('http://localhost:8000/api/v1/chat', json={'session_id': sid, 'query': '它有什么问题吗？'})
print('Round 2:', r2.json()['answer'][:200])
"
```

Expected: Round 2 should reference the device from Round 1 context.

- [ ] **Step 7: 测试场景检测**

```bash
python -c "
from src.middleware.prompt_switch import detect_scenario
tests = [
    ('L5和L3哪个好？', 'presale'),
    ('我的机器报E05错误', 'aftersale'),
    ('怎么清洗滤网？', 'maintenance'),
    ('你好', 'default'),
]
for query, expected in tests:
    result = detect_scenario(query)
    status = '✅' if result == expected else '❌'
    print(f'{status} \"{query}\" → {result} (expected: {expected})')
"
```

Expected: All tests pass.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "test: end-to-end integration verification passed"
```

---

### Task 6.2: 优化 .gitignore

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: 更新 .gitignore**

Ensure the following entries exist (append any missing):

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.pyd

# Environment
.env
.venv/
venv/
env/

# IDE
.vscode/
.idea/

# Logs
logs/*.log
logs/*.jsonl

# ChromaDB
data/chroma_db/

# Superpowers
.superpowers/

# OS
.DS_Store
Thumbs.db
```

- [ ] **Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: update .gitignore with superpowers and chroma_db"
```

---

## Final Verification Checklist

Run these commands after all phases to verify the system:

```bash
# 1. 验证所有模块可导入
python -c "
from src.embedding.embeddings import get_embedding_model
from src.vectordb.loader import DocumentLoader
from src.vectordb.store import VectorStore
from src.vectordb.retriever import KnowledgeRetriever
from src.tools import ALL_TOOLS, get_tools_for_scenario
from src.agent.agent import CustomerServiceAgent
from src.agent.prompt import get_system_prompt_for_scenario
from src.agent.memory import ConversationManager
from src.middleware import CALLBACK_HANDLERS, detect_scenario
from src.api.schemas import ChatRequest, ChatResponse
print('All modules imported successfully!')
print(f'Tools: {len(ALL_TOOLS)}')
"

# 2. 验证向量库
python -c "
from src.vectordb.retriever import KnowledgeRetriever
r = KnowledgeRetriever()
print(f'Chunks in DB: {r.ingest_directory(\"data/kb\", force=True)}')
results = r.retrieve('E05')
print(f'Search results for E05: {len(results)}')
"

# 3. 验证 FastAPI 启动正常
# (manual: uvicorn server:app --port 8000, then check /docs)

# 4. 验证 Streamlit 启动正常
# (manual: streamlit run app.py)
```
