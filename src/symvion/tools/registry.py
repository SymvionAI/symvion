from typing import Callable, Dict, Any, Optional, List, Union
from symvion.tools.base import ToolSafetyWrapper
from symvion.core.context import TenantContext


class ToolRegistry:
    """
    Central registry for all Symvion tools.

    Supports two registration styles:

    1. **Function-based** (legacy): ``register(name, callable)``
       — invoked via ``invoke(name, context, args)``.

    2. **Object-based** (RAG / structured tools): ``register_tool(name, tool)``
       — tool must have an async ``execute(input_data, context)`` method.
       Invoked via ``execute(name, input_data, context)``.
    """

    def __init__(self, iam_policies: Optional[Dict[str, List[str]]] = None):
        self._tools: Dict[str, Callable] = {}
        self._tool_objects: Dict[str, Any] = {}
        self.iam_policies = iam_policies or {}

    # ------------------------------------------------------------------
    # Function-based registration (existing API — unchanged)
    # ------------------------------------------------------------------

    def register(self, name: str, func: Callable) -> None:
        """Register a plain callable as a tool."""
        self._tools[name] = func

    def get_tool(self, name: str) -> Callable:
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found in function registry.")
        return self._tools[name]

    async def invoke(
        self,
        name: str,
        context: TenantContext,
        args: Dict[str, Any],
        timeout: float = 30.0,
    ) -> Any:
        """
        Safely invoke a function-based tool by name via ToolSafetyWrapper
        (includes IAM checks, retries, and timeout handling).
        """
        tool_func = self.get_tool(name)
        return await ToolSafetyWrapper.invoke(
            tool_func,
            context,
            name,
            args,
            timeout=timeout,
            iam_policies=self.iam_policies,
        )

    # ------------------------------------------------------------------
    # Object-based registration (RAG tools, structured tools)
    # ------------------------------------------------------------------

    def register_tool(self, name: str, tool: Any) -> None:
        """
        Register a tool *object* (e.g. RetrievalTool).

        The object must expose an async method::

            async def execute(self, input_data: dict, context=None) -> dict

        Args:
            name: Unique tool identifier used when calling ``execute()``.
            tool: An object with an async ``execute`` method.

        Raises:
            TypeError: If ``tool`` does not have a callable ``execute`` attribute.
        """
        if not callable(getattr(tool, "execute", None)):
            raise TypeError(
                f"Tool '{name}' must have a callable 'execute' method. "
                f"Got {type(tool).__name__!r} which does not."
            )
        self._tool_objects[name] = tool

    def get_tool_object(self, name: str) -> Any:
        """Return a registered tool object by name."""
        if name not in self._tool_objects:
            raise ValueError(f"Tool object '{name}' not found in object registry.")
        return self._tool_objects[name]

    async def execute(
        self,
        name: str,
        input_data: Dict[str, Any],
        context: Optional[TenantContext] = None,
    ) -> Any:
        """
        Invoke a registered tool object's ``execute`` method.

        This is the agent-facing call pattern::

            rag_data = await context.tools.execute(
                "retrieve_knowledge",
                {"query": user_query},
                context,
            )

        Args:
            name:       Tool name as registered via ``register_tool``.
            input_data: Dict passed directly to ``tool.execute``.
            context:    Optional TenantContext for logging and IAM.

        Returns:
            Whatever the tool's ``execute`` method returns.

        Raises:
            ValueError: If the tool is not registered.
        """
        tool = self.get_tool_object(name)
        if context is None:
            return await tool.execute(input_data, context)

        async def _call(**_kwargs):
            return await tool.execute(input_data, context)

        return await ToolSafetyWrapper.invoke(
            _call,
            context,
            name,
            {},
            iam_policies=self.iam_policies,
        )

    # ------------------------------------------------------------------
    # Convenience: list all registered tools
    # ------------------------------------------------------------------

    @property
    def all_tools(self) -> Dict[str, str]:
        """Return a dict of {name: type} for every registered tool."""
        result = {name: "function" for name in self._tools}
        result.update({name: type(obj).__name__ for name, obj in self._tool_objects.items()})
        return result

