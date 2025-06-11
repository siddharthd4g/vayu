import unittest
from unittest.mock import patch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# Note: We are patching the target function in the tool's file now
from src.langgraph.tools.weather_tool import WeatherTool

class TestWeatherTool(unittest.TestCase):
    def test_weather_tool_call(self):
        # MODIFIED INPUT: Use a dictionary, not a Pydantic object
        input_data = {
            "location": "London",
            "date_range": {"start": "2025-08-06", "end": "2025-08-19"}
        }
        
        # Mock the weather API response
        mock_response = {
            "aqi": 45,
            "pm2_5": 12.5,
            "pm10": 25.0
        }
        
        # Create tool instance
        tool = WeatherTool()
        
        # MODIFIED PATCH: The target for the patch is where the function is *used*
        with patch('src.langgraph.tools.weather_tool.get_weather_data', return_value=mock_response) as mock_api_call:
            # MODIFIED CALL: Call the tool with the dictionary
            result = tool.invoke(input_data)
            
            # Assert the result is correct
            self.assertEqual(result, mock_response)
            
            # Assert that the underlying API function was called with the correct arguments
            mock_api_call.assert_called_once_with(
                city_name="London",
                start_date="2025-08-06",
                end_date="2025-08-19"
            )

if __name__ == '__main__':
    unittest.main()