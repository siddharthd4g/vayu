import unittest
import logging
import json
from query_parser_tool import QueryParserTool, ConversationContext

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('query_parser_tests.log')
    ]
)
logger = logging.getLogger(__name__)

class TestQueryParserTool(unittest.TestCase):
    def setUp(self):
        logger.info("Setting up test case")
        self.parser = QueryParserTool()
        self.context = ConversationContext(
            user_info={
                "name": "John",
                "conditions": ["Asthma"]
            },
            previous_queries=[],
            extracted_info={}
        )
        logger.debug(f"Initialized test context: {json.dumps(self.context.user_info, indent=2)}")

    def tearDown(self):
        logger.info("Tearing down test case")
        logger.debug(f"Final context state: {json.dumps(self.context.extracted_info, indent=2)}")
        logger.debug(f"Total queries processed: {len(self.context.previous_queries)}")

    def test_initial_query(self):
        logger.info("Testing initial complete query")
        # Test a complete initial query
        query = "I'm going to Delhi from 15th to 20th June"
        logger.debug(f"Test query: {query}")
        
        model_response = """
        {
            "intent": "travel_planning",
            "extracted_info": {
                "location": "Delhi",
                "date_range": {
                    "start": "2024-06-15",
                    "end": "2024-06-20"
                },
                "specific_concerns": null
            },
            "relevance": {
                "is_relevant": true,
                "confidence": 0.9,
                "reason": "Complete travel planning query"
            },
            "required_actions": ["fetch_weather_data", "analyze_health_impact"],
            "is_complete": true,
            "can_answer": true,
            "context_used": []
        }
        """
        logger.debug(f"Model response: {model_response}")
        
        result = self.parser.parse_query(query, self.context, model_response)
        logger.debug(f"Parse result: {json.dumps(result, indent=2)}")
        
        self.assertEqual(result["intent"], "travel_planning")
        self.assertEqual(result["extracted_info"]["location"], "Delhi")
        self.assertTrue(result["is_complete"])
        self.assertTrue(result["can_answer"])
        self.assertEqual(len(self.context.previous_queries), 1)
        logger.info("Initial query test completed successfully")

    def test_follow_up_query(self):
        logger.info("Testing follow-up query scenario")
        # First query
        first_query = "I'm planning to visit Mumbai"
        logger.debug(f"First query: {first_query}")
        
        first_response = """
        {
            "intent": "travel_planning",
            "extracted_info": {
                "location": "Mumbai",
                "date_range": null,
                "specific_concerns": null
            },
            "relevance": {
                "is_relevant": true,
                "confidence": 0.8,
                "reason": "Travel planning with location"
            },
            "required_actions": ["ask_dates"],
            "is_complete": false,
            "can_answer": false,
            "context_used": []
        }
        """
        logger.debug(f"First model response: {first_response}")
        
        first_result = self.parser.parse_query(first_query, self.context, first_response)
        logger.debug(f"First parse result: {json.dumps(first_result, indent=2)}")
        
        # Follow-up query
        follow_up = "I'll be there from 1st to 5th July"
        logger.debug(f"Follow-up query: {follow_up}")
        
        follow_up_response = """
        {
            "intent": "travel_planning",
            "extracted_info": {
                "location": null,
                "date_range": {
                    "start": "2024-07-01",
                    "end": "2024-07-05"
                },
                "specific_concerns": null
            },
            "relevance": {
                "is_relevant": true,
                "confidence": 0.9,
                "reason": "Follow-up with dates"
            },
            "required_actions": ["fetch_weather_data", "analyze_health_impact"],
            "is_complete": true,
            "can_answer": true,
            "context_used": ["location"]
        }
        """
        logger.debug(f"Follow-up model response: {follow_up_response}")
        
        result = self.parser.parse_query(follow_up, self.context, follow_up_response)
        logger.debug(f"Follow-up parse result: {json.dumps(result, indent=2)}")
        
        self.assertTrue(result["is_complete"])
        self.assertTrue(result["can_answer"])
        self.assertEqual(len(self.context.previous_queries), 2)
        self.assertEqual(self.context.extracted_info["location"], "Mumbai")
        self.assertEqual(
            self.context.extracted_info["date_range"]["start"],
            "2024-07-01"
        )
        logger.info("Follow-up query test completed successfully")

    def test_irrelevant_query(self):
        logger.info("Testing irrelevant query")
        query = "What's the weather like in Paris?"
        logger.debug(f"Test query: {query}")
        
        model_response = """
        {
            "intent": "weather_inquiry",
            "extracted_info": {
                "location": "Paris",
                "date_range": null,
                "specific_concerns": null
            },
            "relevance": {
                "is_relevant": false,
                "confidence": 0.7,
                "reason": "General weather inquiry without health context"
            },
            "required_actions": ["explain_purpose"],
            "is_complete": false,
            "can_answer": false,
            "context_used": []
        }
        """
        logger.debug(f"Model response: {model_response}")
        
        result = self.parser.parse_query(query, self.context, model_response)
        logger.debug(f"Parse result: {json.dumps(result, indent=2)}")
        
        self.assertFalse(result["relevance"]["is_relevant"])
        self.assertFalse(result["can_answer"])
        self.assertIn("explain_purpose", result["required_actions"])
        logger.info("Irrelevant query test completed successfully")

    def test_error_handling(self):
        logger.info("Testing error handling")
        query = "I'm going to Delhi"
        logger.debug(f"Test query: {query}")
        
        invalid_response = "This is not a valid JSON response"
        logger.debug(f"Invalid model response: {invalid_response}")
        
        result = self.parser.parse_query(query, self.context, invalid_response)
        logger.debug(f"Error handling result: {json.dumps(result, indent=2)}")
        
        self.assertEqual(result["intent"], "error")
        self.assertFalse(result["can_answer"])
        self.assertIn("error_handling", result["required_actions"])
        logger.info("Error handling test completed successfully")

if __name__ == '__main__':
    unittest.main() 