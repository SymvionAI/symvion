from typing import Dict, Any, Optional
from langchain_core.runnables import RunnableConfig
from symvion.agents.base import BaseAgent
from symvion.core.context import TenantContext
from symvion.tools.base import ToolSafetyWrapper
from symvion.tools.weather_tools import get_weather
from symvion.tools.hitl import run_tool_with_hitl


class WeatherAgent(BaseAgent):
    """
    Example v0.3 Agent demonstrating lifecycle hooks and tool safety.
    """
    async def execute(
        self,
        context: TenantContext,
        input_data: Dict[str, Any],
        tools: Optional[Any] = None,
        config: Optional[RunnableConfig] = None,
    ) -> Dict[str, Any]:
        message = input_data.get("message", "").lower()
        
        # Simple intent matching
        if "weather in" in message:
            city = message.split("weather in")[-1].strip()

            async def _exec(**kwargs):
                if tools:
                    return await tools.invoke("get_weather", context, kwargs)
                return await ToolSafetyWrapper.invoke(
                    get_weather, context, "get_weather", kwargs
                )

            weather_data = await run_tool_with_hitl(
                tool_name="get_weather",
                tool_args={"city": city},
                call_id=f"weather_{city}",
                execute=_exec,
                config=config,
            )

            return {
                "agent_response": f"The weather in {city} is {weather_data['temperature']}°C and {weather_data['condition']}.",
                "token_usage": {"input_tokens": 10, "output_tokens": 20}
            }
        
        return {
            "agent_response": "I can tell you the weather! Just ask 'What is the weather in [city]?'",
            "token_usage": {"input_tokens": 5, "output_tokens": 10}
        }
