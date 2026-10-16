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
        self._langchain_tools: Dict[str, Any] = {}
        self._bindable: Dict[str, Any] = {}
        self.iam_policies = iam_policies or {}

    # ------------------------------------------------------------------
    # Function-based registration (existing API — unchanged)
    # ------------------------------------------------------------------

    def register(self, name: str, func: Callable) -> None:
        """Register a plain callable as a tool."""
        self._tools[name] = func
        self._bindable.pop(name, None)

    def register_langchain_tool(self, name: str, tool: Any) -> None:
        """Register a LangChain tool that ``bind_tools`` can use directly."""
        if not getattr(tool, "name", None):
            raise TypeError(
                f"Tool '{name}' must be a LangChain tool with a name. "
                f"Got {type(tool).__name__!r}."
            )
        self._langchain_tools[name] = tool
        self._bindable.pop(name, None)

    def lookup_bindable(self, name: str) -> Optional[Any]:
        """Return a LangChain tool for ``name``, wrapping registry entries when needed."""
        if name in self._bindable:
            return self._bindable[name]
        tool = None
        if name in self._langchain_tools:
            tool = self._langchain_tools[name]
        elif name in self._tools:
            tool = self._wrap_callable(name, self._tools[name])
        elif name in self._tool_objects:
            tool = self._wrap_object(name, self._tool_objects[name])
        if tool is not None:
            self._bindable[name] = tool
        return tool

    @staticmethod
    def _wrap_callable(name: str, func: Callable) -> Any:
        import asyncio
        from langchain_core.tools import StructuredTool

        description = (getattr(func, "__doc__", None) or name).strip()
        if asyncio.iscoroutinefunction(func):
            return StructuredTool.from_function(
                coroutine=func, name=name, description=description
            )
        return StructuredTool.from_function(func=func, name=name, description=description)

    @staticmethod
    def _wrap_object(name: str, obj: Any) -> Any:
        import asyncio
        from langchain_core.tools import StructuredTool

        description = str(getattr(obj, "description", None) or name)
        execute = obj.execute
        if asyncio.iscoroutinefunction(execute):

            async def _run(**kwargs):
                return await execute(kwargs)

            return StructuredTool.from_function(
                coroutine=_run, name=name, description=description
            )

        def _run_sync(**kwargs):
            return execute(kwargs)

        return StructuredTool.from_function(
            func=_run_sync, name=name, description=description
        )

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
        self._bindable.pop(name, None)

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
        result.update({name: "langchain" for name in self._langchain_tools})
        return result

