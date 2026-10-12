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
