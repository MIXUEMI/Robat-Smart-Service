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
