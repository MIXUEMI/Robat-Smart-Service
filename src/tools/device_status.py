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
