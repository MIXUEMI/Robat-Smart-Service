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
