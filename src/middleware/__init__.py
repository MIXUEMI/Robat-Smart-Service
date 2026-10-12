"""中间件链模块。

基于 LangChain BaseCallbackHandler 实现的可插拔中间件体系。
"""

from src.middleware.logging_mw import AuditLoggingHandler
from src.middleware.monitor_mw import PerformanceMonitor
from src.middleware.prompt_switch import detect_scenario

CALLBACK_HANDLERS = [
    AuditLoggingHandler(),
    PerformanceMonitor(),
]

__all__ = ["AuditLoggingHandler", "PerformanceMonitor", "detect_scenario", "CALLBACK_HANDLERS"]
