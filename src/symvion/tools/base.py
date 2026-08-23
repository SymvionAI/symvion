import asyncio
import time
from typing import Any, Dict, Callable, Optional, List
from symvion.core.context import TenantContext
from symvion.core.logger import logger
from symvion.core.exceptions import ToolExecutionError

class ToolSafetyWrapper:
    """
    Wrapping layer that ensures tool calls are safe, observable, and isolated.
    Ensures a single tool failure doesn't crash the orchestrator.
    """
    
    @staticmethod
    async def invoke(
        tool_func: Callable,
        context: TenantContext,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: float = 30.0,
        retries: int = 2,
        iam_policies: Optional[Dict[str, List[str]]] = None
    ) -> Any:
        """Execute a tool with safety guardrails and IAM checks."""
        
        # IAM Check — fail closed when policies are an empty dict.
        if iam_policies is not None:
            user_role = context.metadata.get("user_role", "user")
            allowed_tools = iam_policies.get(user_role, []) if iam_policies else []
            
            is_allowed = bool(allowed_tools) and (
                "*" in allowed_tools or tool_name in allowed_tools
            )
            if not is_allowed:
                err_msg = f"Unauthorized: Role '{user_role}' is not allowed to use tool '{tool_name}'"
                logger.error("TOOL_UNAUTHORIZED", context, tool_name=tool_name, user_role=user_role)
                raise ToolExecutionError(err_msg)

        # Do not log raw tool arguments — may contain PII or secrets.
        logger.info("TOOL_CALLED", context, tool_name=tool_name, arg_keys=list((arguments or {}).keys()))

        start_time = time.perf_counter()
        
        for attempt in range(retries):
            try:
                # Wrap sync/async tool execution safely with timeout on both paths.
                if asyncio.iscoroutinefunction(tool_func):
                    result = await asyncio.wait_for(tool_func(**arguments), timeout=timeout)
                else:
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: tool_func(**arguments)),
                        timeout=timeout,
                    )
                
                duration = time.perf_counter() - start_time
                logger.info("TOOL_SUCCESS", context, tool_name=tool_name, duration=duration)
                return result
                
            except asyncio.TimeoutError:
                logger.warning("TOOL_TIMEOUT", context, tool_name=tool_name, attempt=attempt+1)
                if attempt == retries - 1:
                    err_msg = f"Tool {tool_name} timed out after {timeout}s"
                    logger.error("TOOL_FAILED", context, tool_name=tool_name, error=err_msg)
                    raise ToolExecutionError(err_msg)
            except Exception as e:
                logger.warning("TOOL_ATTEMPT_FAILED", context, tool_name=tool_name, attempt=attempt+1, error=type(e).__name__)
                if attempt == retries - 1:
                    logger.error("TOOL_FAILED", context, tool_name=tool_name, error=type(e).__name__)
                    raise ToolExecutionError(f"Tool {tool_name} failed") from e
                
                await asyncio.sleep(0.5 * (attempt + 1))
        
        raise ToolExecutionError(f"Tool {tool_name} failed after {retries} attempts.")
