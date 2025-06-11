import requests
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Union
import json
from dataclasses import dataclass

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('weather_api.log')
    ]
)
logger = logging.getLogger(__name__)

# API Constants
GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
AIR_QUALITY_API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
FORECAST_DAYS = 16  # Maximum forecast days available

@dataclass
class CityCoordinates:
    name: str
    latitude: float
    longitude: float
    country: str
    admin1: str  # State/Province

class WeatherAPIError(Exception):
    """Custom exception for weather API errors"""
    pass

def get_city_coordinates(city_name: str) -> CityCoordinates:
    """
    Get coordinates for a city using the Open-Meteo Geocoding API.
    
    Args:
        city_name (str): Name of the city to search for
        
    Returns:
        CityCoordinates: Dataclass containing city information and coordinates
        
    Raises:
        WeatherAPIError: If city is not found or API request fails
    """
    logger.info(f"Fetching coordinates for city: {city_name}")
    try:
        params = {
            "name": city_name,
            "count": 1,
            "language": "en",
            "format": "json"
        }
        
        logger.debug(f"Making geocoding API request with params: {params}")
        response = requests.get(GEOCODING_API_URL, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get("results"):
            logger.error(f"No results found for city: {city_name}")
            raise WeatherAPIError(f"City '{city_name}' not found")
            
        result = data["results"][0]
        city = CityCoordinates(
            name=result["name"],
            latitude=result["latitude"],
            longitude=result["longitude"],
            country=result["country"],
            admin1=result.get("admin1", "")
        )
        logger.info(f"Successfully found coordinates for {city.name}: lat={city.latitude}, lon={city.longitude}, country={city.country}")
        return city
        
    except requests.RequestException as e:
        logger.error(f"Geocoding API request failed: {str(e)}")
        raise WeatherAPIError(f"Failed to fetch city coordinates: {str(e)}")
    except (KeyError, IndexError) as e:
        logger.error(f"Unexpected geocoding API response format: {str(e)}")
        raise WeatherAPIError("Invalid API response format")

def is_within_forecast_range(date: datetime) -> bool:
    """
    Check if a date is within the forecast range (16 days from today).
    
    Args:
        date (datetime): Date to check
        
    Returns:
        bool: True if date is within forecast range, False otherwise
    """
    today = datetime.now().date()
    max_forecast_date = today + timedelta(days=FORECAST_DAYS)
    is_within = today <= date.date() <= max_forecast_date
    logger.info(f"Checking forecast range for date {date.date()}: within_range={is_within}, max_forecast_date={max_forecast_date}")
    return is_within

def get_air_quality_data(
    lat: float,
    lon: float,
    start_date: datetime,
    end_date: datetime
) -> Dict:
    """
    Get air quality data for a location.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        start_date (datetime): Start date for data
        end_date (datetime): End date for data
        
    Returns:
        Dict: Air quality data with the following structure:
        {
            "hourly": {
                "time": List[str],
                "european_aqi": List[float],
                "pm2_5": List[float],
                "pm10": List[float],
                "ozone": List[float],
                "nitrogen_dioxide": List[float],
                "sulphur_dioxide": List[float]
            },
            "hourly_units": {
                "time": str,
                "european_aqi": str,
                "pm2_5": str,
                "pm10": str,
                "ozone": str,
                "nitrogen_dioxide": str,
                "sulphur_dioxide": str
            }
        }
        
    Raises:
        WeatherAPIError: If API request fails or dates are invalid
    """
    logger.info(f"Fetching air quality data for coordinates: lat={lat}, lon={lon}, start={start_date.date()}, end={end_date.date()}")
    
    if not is_within_forecast_range(end_date):
        logger.error(f"End date {end_date.date()} is beyond forecast range of {FORECAST_DAYS} days")
        raise WeatherAPIError(f"End date is beyond forecast range of {FORECAST_DAYS} days")
        
    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": [
                "european_aqi",
                "pm2_5",
                "pm10",
                "ozone",
                "nitrogen_dioxide",
                "sulphur_dioxide"
            ],
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "timezone": "auto"
        }
        
        logger.debug(f"Making air quality API request with params: {params}")
        response = requests.get(AIR_QUALITY_API_URL, params=params)
        response.raise_for_status()
        
        data = response.json()
        logger.info(f"Successfully fetched air quality data for {len(data['hourly']['time'])} time points")
        return data
        
    except requests.RequestException as e:
        logger.error(f"Air quality API request failed: {str(e)}")
        raise WeatherAPIError(f"Failed to fetch air quality data: {str(e)}")

def get_weather_data(
    city_name: str,
    start_date: datetime,
    end_date: datetime
) -> Dict:
    """
    Get weather data for a city during a specific date range.
    
    Args:
        city_name (str): Name of the city
        start_date (datetime): Start date
        end_date (datetime): End date
        
    Returns:
        Dict: Weather data with the following structure:
        {
            "city": {
                "name": str,
                "country": str,
                "admin1": str,
                "coordinates": {
                    "latitude": float,
                    "longitude": float
                }
            },
            "date_range": {
                "start": str,
                "end": str
            },
            "air_quality": {
                "hourly": {
                    "time": List[str],
                    "european_aqi": List[float],
                    "pm2_5": List[float],
                    "pm10": List[float],
                    "ozone": List[float],
                    "nitrogen_dioxide": List[float],
                    "sulphur_dioxide": List[float]
                },
                "hourly_units": {
                    "time": str,
                    "european_aqi": str,
                    "pm2_5": str,
                    "pm10": str,
                    "ozone": str,
                    "nitrogen_dioxide": str,
                    "sulphur_dioxide": str
                }
            }
        }
    """
    logger.info(f"Getting weather data for {city_name} from {start_date.date()} to {end_date.date()}")
    
    try:
        # Get city coordinates
        city = get_city_coordinates(city_name)
        
        # Get air quality data
        air_quality_data = get_air_quality_data(
            city.latitude,
            city.longitude,
            start_date,
            end_date
        )
        
        result = {
            "city": {
                "name": city.name,
                "country": city.country,
                "admin1": city.admin1,
                "coordinates": {
                    "latitude": city.latitude,
                    "longitude": city.longitude
                }
            },
            "date_range": {
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d")
            },
            "air_quality": air_quality_data
        }
        
        logger.info(f"Successfully retrieved weather data for {city.name}")
        return result
        
    except WeatherAPIError as e:
        logger.error(f"Weather API error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_weather_data: {str(e)}")
        raise WeatherAPIError(f"Failed to get weather data: {str(e)}") 