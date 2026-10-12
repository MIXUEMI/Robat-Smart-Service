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
