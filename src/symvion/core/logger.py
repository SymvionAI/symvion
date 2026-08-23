import logging
import json
import time
from typing import Any, Dict, Optional
from symvion.core.context import TenantContext

class SymvionLogger:
    """
    Structured JSON logger for production-grade observability.
    Produces machine-readable logs with correlation IDs.
    """
    def __init__(self, name: str = "symvion"):
        self.logger = logging.getLogger(name)
        self.logger.propagate = False
        self.show_json = True
        self.events = [] # Buffer for recent events
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def configure(self, enabled: bool = True, level: str = "INFO", file_path: Optional[str] = None, show_json: bool = True):
        """Reconfigure the logger based on TenantConfig."""
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        if not enabled:
            self.logger.setLevel(logging.CRITICAL + 1)
            return

        self.show_json = show_json
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        
        if file_path:
            handler = logging.FileHandler(file_path)
        else:
            handler = logging.StreamHandler()
            
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _log_event(self, event_type: str, context: Optional[TenantContext], data: Dict[str, Any], level: int = logging.INFO):
        if not self.show_json:
            # Fallback to plain text if JSON is disabled
            msg = f"[{event_type}] " + " ".join([f"{k}={v}" for k, v in data.items()])
            self.logger.log(level, msg)
            return

        log_payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": event_type,
            "tenant_id": context.tenant_id if context else "unknown",
            "request_id": context.request_id if context else "unknown",
        }
        log_payload.update(data)
        
        # Buffer for internal observability API
        self.events.append(log_payload)
        if len(self.events) > 50:
            self.events.pop(0)

        self.logger.log(level, json.dumps(log_payload))

    def info(self, event_type: str, context: Optional[TenantContext], **kwargs):
        self._log_event(event_type, context, kwargs, logging.INFO)

    def error(self, event_type: str, context: Optional[TenantContext], **kwargs):
        # Ensure 'error' key is present in kwargs for error logs
        if "error" not in kwargs and "message" in kwargs:
            kwargs["error"] = kwargs.pop("message")
        self._log_event(event_type, context, kwargs, logging.ERROR)

    def warning(self, event_type: str, context: Optional[TenantContext], **kwargs):
        self._log_event(event_type, context, kwargs, logging.WARNING)

# Global logger instance for convenience
logger = SymvionLogger()
