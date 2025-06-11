from typing import Dict, List, Optional
import json
import logging
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('query_parser.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ConversationContext:
    """Maintains the state of the conversation."""
    user_info: Dict  # User's health information
    previous_queries: List[Dict]  # List of previous parsed queries
    extracted_info: Dict  # Accumulated information from the conversation
    last_intent: Optional[str] = None
    last_required_actions: Optional[List[str]] = None

    def __post_init__(self):
        """Log initialization of new conversation context."""
        logger.info(f"Initialized new ConversationContext for user: {self.user_info.get('name', 'Unknown')}")
        logger.debug(f"User conditions: {self.user_info.get('conditions', [])}")

class QueryParserTool:
    def __init__(self):
        self.system_prompt = """You are a health-focused weather advisory system. Your role is to:
1. Understand user queries about weather and health
2. Determine what information is needed
3. Structure responses in valid JSON format
4. Guide the conversation to collect missing information

IMPORTANT: Always return valid JSON. Never return plain text.

Response Structure:
{
    "intent": "weather_health" | "medical_research" | "general_health" | "irrelevant",
    "extracted_info": {
        "location": "string or null",
        "date_range": {
            "start": "YYYY-MM-DD or null",
            "end": "YYYY-MM-DD or null"
        },
        "health_condition": "string or null"
    },
    "relevance": {
        "is_relevant": boolean,
        "reason": "string if not relevant"
    },
    "required_actions": {
        "needs_weather_data": boolean,
        "needs_medical_research": boolean,
        "missing_info": ["location", "date_range", "health_condition"]
    },
    "is_complete": boolean,
    "can_answer": boolean,
    "context_used": {
        "previous_queries": boolean,
        "user_health_info": boolean
    },
    "response": {
        "text": "string",
        "type": "request_info" | "health_advisory" | "irrelevant"
    }
}"""

    def create_parser_prompt(self, query: str, context: ConversationContext) -> str:
        return f"""Analyze the following query and context to determine what information is needed.

Current Query: {query}

User Health Information:
{json.dumps(context.user_info, indent=2)}

Previous Queries:
{json.dumps(context.previous_queries, indent=2)}

Extracted Information:
{json.dumps(context.extracted_info, indent=2)}

Rules:
1. If location is missing, mark as incomplete and request the location
2. If date_range is missing or incomplete, mark as incomplete and request the date range
3. If health_condition is missing, mark as incomplete and request health information
4. If all required information is present, mark as complete and set needs_weather_data and needs_medical_research to true
5. Always return valid JSON matching the specified structure
6. For incomplete queries, provide a response that requests the missing information
7. For complete queries, provide a response that acknowledges the information and indicates data collection will begin

Example Response for Complete Query:
{{
    "intent": "weather_health",
    "extracted_info": {{
        "location": "London",
        "date_range": {{
            "start": "2025-08-06",
            "end": "2025-08-19"
        }},
        "health_condition": "asthma"
    }},
    "relevance": {{
        "is_relevant": true,
        "reason": null
    }},
    "required_actions": {{
        "needs_weather_data": true,
        "needs_medical_research": true,
        "missing_info": []
    }},
    "is_complete": true,
    "can_answer": true,
    "context_used": {{
        "previous_queries": true,
        "user_health_info": true
    }},
    "response": {{
        "text": "I'll analyze the weather conditions in London for your asthma during your stay from August 6th to August 19th, 2025. Let me gather the necessary information.",
        "type": "health_advisory"
    }}
}}"""

    def parse_query(self, query: str, context: ConversationContext, model_response: str) -> dict:
        try:
            # Parse the model response as JSON
            parsed_response = json.loads(model_response)
            
            # Update conversation context
            context.previous_queries.append(query)
            
            # Update extracted information
            if parsed_response["extracted_info"]["location"]:
                context.extracted_info["location"] = parsed_response["extracted_info"]["location"]
            
            if parsed_response["extracted_info"]["date_range"]:
                context.extracted_info["date_range"] = parsed_response["extracted_info"]["date_range"]
            
            if parsed_response["extracted_info"]["health_condition"]:
                context.extracted_info["health_condition"] = parsed_response["extracted_info"]["health_condition"]
            
            # If the query is complete, ensure we trigger data collection
            if parsed_response["is_complete"]:
                parsed_response["required_actions"]["needs_weather_data"] = True
                parsed_response["required_actions"]["needs_medical_research"] = True
            
            return parsed_response
        except json.JSONDecodeError:
            return {
                "intent": "error",
                "extracted_info": {},
                "relevance": {
                    "is_relevant": False,
                    "reason": "Invalid response format"
                },
                "required_actions": {
                    "needs_weather_data": False,
                    "needs_medical_research": False,
                    "missing_info": []
                },
                "is_complete": False,
                "can_answer": False,
                "context_used": {
                    "previous_queries": False,
                    "user_health_info": False
                },
                "response": {
                    "text": "I apologize, but I encountered an error processing your request. Please try again.",
                    "type": "error"
                }
            }

    def _format_previous_queries(self, previous_queries: List[Dict]) -> str:
        """Format previous queries for the prompt."""
        logger.debug(f"Formatting {len(previous_queries)} previous queries")
        
        if not previous_queries:
            logger.debug("No previous queries to format")
            return "No previous queries."
        
        formatted = []
        for i, query in enumerate(previous_queries, 1):
            formatted.append(f"Query {i}:")
            formatted.append(f"Intent: {query['intent']}")
            formatted.append(f"Extracted Info: {json.dumps(query['extracted_info'])}")
            formatted.append(f"Required Actions: {', '.join(query['required_actions'])}")
            formatted.append("---")
        
        return "\n".join(formatted)

    def _update_context(self, context: ConversationContext, parsed_response: Dict):
        """Update the conversation context with new information."""
        logger.info("Updating conversation context")
        logger.debug(f"Previous context state: {json.dumps(context.extracted_info, indent=2)}")
        
        # Update last intent and actions
        context.last_intent = parsed_response["intent"]
        context.last_required_actions = parsed_response["required_actions"]
        logger.debug(f"Updated intent: {context.last_intent}")
        logger.debug(f"Updated required actions: {context.last_required_actions}")
        
        # Update extracted information
        new_info = parsed_response["extracted_info"]
        for key, value in new_info.items():
            if value is not None:  # Only update if new information is provided
                if key == "date_range":
                    if context.extracted_info.get("date_range", {}).get("start") is None:
                        context.extracted_info["date_range"] = value
                        logger.debug(f"Updated date range: {value}")
                elif key in ["weather_data", "medical_insights"]:
                    # Update weather and medical data if available
                    context.extracted_info[key] = value
                    logger.debug(f"Updated {key}: {value}")
                else:
                    context.extracted_info[key] = value
                    logger.debug(f"Updated {key}: {value}")
        
        # Add to previous queries
        context.previous_queries.append(parsed_response)
        logger.debug(f"Added to previous queries. Total queries: {len(context.previous_queries)}")
        
        logger.info("Context update complete")
        logger.debug(f"New context state: {json.dumps(context.extracted_info, indent=2)}")

    def _create_error_response(self, error_message: str) -> Dict:
        """Create a standardized error response."""
        logger.error(f"Creating error response: {error_message}")
        return {
            "intent": "error",
            "extracted_info": {},
            "relevance": {
                "is_relevant": False,
                "confidence": 0.0,
                "reason": f"Error: {error_message}"
            },
            "required_actions": ["error_handling"],
            "is_complete": False,
            "can_answer": False,
            "context_used": [],
            "response": {
                "text": f"I apologize, but I'm having trouble processing your request. {error_message}",
                "type": "error",
                "missing_info": None,
                "data_sources": None
            }
        }

    def get_accumulated_info(self, context: ConversationContext) -> Dict:
        """Get the accumulated information from the conversation."""
        logger.info("Retrieving accumulated information")
        logger.debug(f"Accumulated info: {json.dumps(context.extracted_info, indent=2)}")
        return context.extracted_info

    def is_conversation_complete(self, context: ConversationContext) -> bool:
        """Check if we have all the information needed to proceed."""
        logger.info("Checking if conversation is complete")
        required_info = ["location", "date_range"]
        is_complete = all(
            context.extracted_info.get(key) is not None 
            for key in required_info
        )
        
        # Check if we have weather data and medical insights if needed
        if is_complete and context.last_intent in ["weather_inquiry", "health_impact"]:
            has_weather = context.extracted_info.get("weather_data") is not None
            has_medical = context.extracted_info.get("medical_insights") is not None
            is_complete = has_weather and has_medical
            
        logger.debug(f"Conversation complete: {is_complete}")
        logger.debug(f"Missing information: {[key for key in required_info if context.extracted_info.get(key) is None]}")
        return is_complete 