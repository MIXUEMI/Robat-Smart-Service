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
