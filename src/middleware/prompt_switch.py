"""动态 Prompt 切换模块。

根据用户问题中的关键词自动检测业务场景。
"""

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
