"""性能监控 Callback Handler。

追踪全链路性能指标：首字延迟、总耗时、Token 使用量、工具调用次数。
"""

import logging
import time

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)


class PerformanceMonitor(BaseCallbackHandler):
    """全链路性能监控处理器。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time: float = 0
        self.first_token_time: float = 0
        self.total_tokens: int = 0
        self.tool_call_count: int = 0

    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs) -> None:
        self.reset()
        self.start_time = time.time()

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
            latency_ms, ttft_ms, self.total_tokens, self.tool_call_count,
        )
