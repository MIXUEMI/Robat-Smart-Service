"""Streamlit 聊天界面。

扫地机器人智能客服前端。

启动命令：
    streamlit run app.py
"""

import uuid

import requests
import streamlit as st

st.set_page_config(
    page_title="小扫 · 智能客服",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000/api/v1"


# ── 侧边栏 ──
with st.sidebar:
    st.title("🤖 小扫 · 智能客服")
    st.caption("灵悦扫地机器人专属助手")

    st.divider()

    if st.button("✨ 新对话", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()

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

    with st.expander("📊 系统状态"):
        try:
            resp = requests.get(f"{API_BASE}/health", timeout=5)
            if resp.status_code == 200:
                health = resp.json()
                st.success("🟢 服务运行中")
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
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

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
