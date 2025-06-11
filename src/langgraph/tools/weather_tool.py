from langchain_core.tools import StructuredTool
from typing import Any, Dict
from pydantic import BaseModel, Field
from weather_integration.weather_api import get_weather_data

class WeatherInput(BaseModel):
    location: str = Field(description="The city name to get weather data for")
    date_range: Dict[str, str] = Field(description="Dictionary containing start and end dates in YYYY-MM-DD format")

# MODIFIED FUNCTION: Accepts keyword arguments instead of a single object
def get_weather(location: str, date_range: Dict[str, str]) -> Dict[str, Any]:
    """Fetches weather data based on location and date range."""
    return get_weather_data(
        city_name=location,
        start_date=date_range["start"],
        end_date=date_range["end"]
    )

class WeatherTool(StructuredTool):
    name: str = "weather_tool"
    description: str = "Get weather and air quality data for a location and date range."
    args_schema: type[BaseModel] = WeatherInput
    func: Any = get_weather