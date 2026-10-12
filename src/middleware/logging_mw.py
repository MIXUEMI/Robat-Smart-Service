"""日志审计 Callback Handler。

记录每次 LLM 调用和工具调用的完整审计日志。
日志以 JSONL 格式写入 logs/audit.jsonl。
"""

import json
import os
from datetime import datetime

from langchain_core.callbacks import BaseCallbackHandler


class AuditLoggingHandler(BaseCallbackHandler):
    """请求日志审计处理器。"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._log_file = os.path.join(log_dir, "audit.jsonl")

    def _write(self, entry: dict) -> None:
        entry["timestamp"] = datetime.now().isoformat()
        try:
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs) -> None:
        prompt_preview = prompts[0][:200] if prompts else ""
        self._write({
            "event": "llm_request",
            "model": serialized.get("kwargs", {}).get("model", "unknown"),
            "prompt_preview": prompt_preview,
            "prompt_length": len(prompts[0]) if prompts else 0,
        })

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
        self._write({
            "event": "tool_call",
            "tool": serialized.get("name", "unknown"),
            "input": str(input_str)[:500],
        })

    def on_tool_end(self, output: str, **kwargs) -> None:
        self._write({
            "event": "tool_result",
            "output_preview": str(output)[:500],
        })

    def on_llm_end(self, response, **kwargs) -> None:
        output = ""
        if hasattr(response, "generations") and response.generations:
            gen = response.generations[0][0]
            if hasattr(gen, "message"):
                output = gen.message.content or ""
            elif hasattr(gen, "text"):
                output = gen.text or ""

        self._write({
            "event": "llm_response",
            "response_preview": output[:500],
            "response_length": len(output),
        })
