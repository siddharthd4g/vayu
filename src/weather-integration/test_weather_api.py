import unittest
from datetime import datetime, timedelta
import logging
from weather_api import (
    get_city_coordinates,
    get_air_quality_data,
    get_weather_data,
    WeatherAPIError,
    is_within_forecast_range
)

# Configure test logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_weather_api.log')
    ]
)
logger = logging.getLogger(__name__)

class TestWeatherAPI(unittest.TestCase):
    def setUp(self):
        """Set up test data before each test"""
        logger.info("Setting up test data")
        self.test_city = "London"
        self.start_date = datetime.now()
        self.end_date = self.start_date + timedelta(days=5)
        logger.info(f"Test parameters: city={self.test_city}, start_date={self.start_date.date()}, end_date={self.end_date.date()}")
        
    def test_get_city_coordinates(self):
        """Test getting coordinates for a valid city"""
        logger.info("Testing get_city_coordinates with valid city")
        city = get_city_coordinates(self.test_city)
        
        # Test city name
        self.assertEqual(city.name, "London", "City name should be 'London'")
        logger.info(f"City name verified: {city.name}")
        
        # Test coordinate types
        self.assertIsInstance(city.latitude, float, "Latitude should be float")
        self.assertIsInstance(city.longitude, float, "Longitude should be float")
        logger.info(f"Coordinates verified: lat={city.latitude}, lon={city.longitude}")
        
        # Test country
        self.assertEqual(city.country, "United Kingdom", "Country should be 'United Kingdom'")
        logger.info(f"Country verified: {city.country}")
        
    def test_get_city_coordinates_invalid(self):
        """Test getting coordinates for an invalid city"""
        invalid_city = "NonExistentCity123"
        logger.info(f"Testing get_city_coordinates with invalid city: {invalid_city}")
        
        with self.assertRaises(WeatherAPIError):
            get_city_coordinates(invalid_city)
        logger.info("Successfully caught WeatherAPIError for invalid city")
            
    def test_is_within_forecast_range(self):
        """Test forecast range validation"""
        logger.info("Testing is_within_forecast_range")
        
        # Test date within range
        future_date = datetime.now() + timedelta(days=10)
        self.assertTrue(
            is_within_forecast_range(future_date),
            f"Date {future_date.date()} should be within forecast range"
        )
        logger.info(f"Verified date within range: {future_date.date()}")
        
        # Test date beyond range
        far_future_date = datetime.now() + timedelta(days=20)
        self.assertFalse(
            is_within_forecast_range(far_future_date),
            f"Date {far_future_date.date()} should be beyond forecast range"
        )
        logger.info(f"Verified date beyond range: {far_future_date.date()}")
        
    def test_get_air_quality_data(self):
        """Test getting air quality data for valid coordinates"""
        logger.info("Testing get_air_quality_data")
        
        # Get city coordinates first
        city = get_city_coordinates(self.test_city)
        logger.info(f"Got coordinates for {self.test_city}")
        
        # Get air quality data
        data = get_air_quality_data(
            city.latitude,
            city.longitude,
            self.start_date,
            self.end_date
        )
        logger.info("Successfully retrieved air quality data")
        
        # Check data structure
        self.assertIn("hourly", data, "Data should contain 'hourly' key")
        self.assertIn("hourly_units", data, "Data should contain 'hourly_units' key")
        logger.info("Verified data structure")
        
        # Check required fields
        required_fields = [
            "time",
            "european_aqi",
            "pm2_5",
            "pm10",
            "ozone",
            "nitrogen_dioxide",
            "sulphur_dioxide"
        ]
        
        for field in required_fields:
            self.assertIn(field, data["hourly"], f"Hourly data should contain {field}")
            self.assertIn(field, data["hourly_units"], f"Hourly units should contain {field}")
        logger.info("Verified all required fields")
            
    def test_get_weather_data(self):
        """Test getting complete weather data for a city"""
        logger.info("Testing get_weather_data")
        
        data = get_weather_data(self.test_city, self.start_date, self.end_date)
        logger.info("Successfully retrieved weather data")
        
        # Check data structure
        self.assertIn("city", data, "Data should contain 'city' key")
        self.assertIn("date_range", data, "Data should contain 'date_range' key")
        self.assertIn("air_quality", data, "Data should contain 'air_quality' key")
        logger.info("Verified top-level data structure")
        
        # Check city information
        self.assertEqual(data["city"]["name"], "London", "City name should be 'London'")
        self.assertIn("coordinates", data["city"], "City data should contain coordinates")
        logger.info("Verified city information")
        
        # Check date range
        self.assertEqual(
            data["date_range"]["start"],
            self.start_date.strftime("%Y-%m-%d"),
            "Start date should match"
        )
        self.assertEqual(
            data["date_range"]["end"],
            self.end_date.strftime("%Y-%m-%d"),
            "End date should match"
        )
        logger.info("Verified date range")
        
        # Check air quality data
        self.assertIn("hourly", data["air_quality"], "Air quality data should contain hourly data")
        self.assertIn("hourly_units", data["air_quality"], "Air quality data should contain hourly units")
        logger.info("Verified air quality data structure")
        
    def test_get_weather_data_invalid_dates(self):
        """Test getting weather data with invalid date range"""
        logger.info("Testing get_weather_data with invalid dates")
        
        far_future_date = datetime.now() + timedelta(days=20)
        logger.info(f"Using end date beyond forecast range: {far_future_date.date()}")
        
        with self.assertRaises(WeatherAPIError):
            get_weather_data(self.test_city, self.start_date, far_future_date)
        logger.info("Successfully caught WeatherAPIError for invalid date range")

if __name__ == "__main__":
    logger.info("Starting weather API tests")
    unittest.main() 