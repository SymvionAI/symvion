import asyncio
from typing import Dict, Any

async def get_weather(city: str) -> Dict[str, Any]:
    """Get the current weather for a city."""
    # Simulate API call latency
    await asyncio.sleep(0.5)
    return {
        "city": city,
        "temperature": 22,
        "condition": "Partly Cloudy",
        "unit": "celsius"
    }
