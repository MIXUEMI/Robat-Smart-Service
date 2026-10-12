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
