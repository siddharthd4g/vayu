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
        logger.info("Initializing QueryParserTool")
        self.system_prompt = """
        You are a query parser for a health-focused weather advisory system. Your task is to:
        1. Understand the user's intent
        2. Extract relevant information
        3. Determine if the query is relevant to our purpose
        4. Suggest next actions
        5. Maintain context from previous queries

        The system's purpose is to help individuals with respiratory issues understand:
        - Weather conditions during their travel
        - Air quality impact on their health
        - Precautionary measures they should take

        You should NOT:
        - Give medical advice
        - Suggest treatments
        - Interpret symptoms
        - Provide emergency contacts

        Output should be in JSON format with the following structure:
        {
            "intent": str,  # One of: travel_planning, weather_inquiry, health_impact, general_inquiry
            "extracted_info": {
                "location": str | null,
                "date_range": {
                    "start": str | null,  # ISO format date
                    "end": str | null     # ISO format date
                },
                "specific_concerns": List[str] | null
            },
            "relevance": {
                "is_relevant": bool,
                "confidence": float,
                "reason": str
            },
            "required_actions": List[str],  # Possible actions: ask_location, ask_dates, explain_purpose, fetch_weather_data, analyze_health_impact
            "is_complete": bool,  # Whether we have all information needed to proceed
            "can_answer": bool,   # Whether this query can be answered by our system
            "context_used": List[str]  # What context from previous queries was used
        }
        """
        logger.debug("System prompt initialized")

    def create_parser_prompt(self, query: str, context: ConversationContext) -> str:
        """Create a prompt that includes the current query and conversation context."""
        logger.info(f"Creating parser prompt for query: {query}")
        logger.debug(f"Current context - Previous queries: {len(context.previous_queries)}")
        
        prompt = f"""
        Current Query: {query}

        User Context:
        - Name: {context.user_info.get('name', 'Not provided')}
        - Conditions: {', '.join(context.user_info.get('conditions', []))}

        Previous Queries Context:
        {self._format_previous_queries(context.previous_queries)}

        Currently Extracted Information:
        {json.dumps(context.extracted_info, indent=2)}

        Analyze this query in the context of the conversation and provide the structured response as specified.
        Consider:
        1. Is this a follow-up to a previous query?
        2. Does it provide missing information from previous queries?
        3. Can we answer this query with our current knowledge?
        4. Do we need more information to proceed?
        """
        logger.debug("Parser prompt created successfully")
        return prompt

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

    def parse_query(self, query: str, context: ConversationContext, model_response: str) -> Dict:
        """
        Parse the user query using the provided model response.
        
        Args:
            query (str): The user's input query
            context (ConversationContext): Current conversation context
            model_response (str): Response from the LLM model
            
        Returns:
            Dict: Parsed information including intent, extracted info, and required actions
        """
        logger.info(f"Parsing query: {query}")
        logger.debug(f"Model response: {model_response}")
        
        try:
            # Parse the model's response
            parsed_response = json.loads(model_response)
            logger.debug(f"Successfully parsed JSON response: {json.dumps(parsed_response, indent=2)}")
            
            # Update the conversation context
            self._update_context(context, parsed_response)
            logger.info(f"Updated context with new information. Intent: {parsed_response['intent']}")
            
            return parsed_response

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse model response as JSON: {model_response}")
            logger.error(f"JSON decode error: {str(e)}")
            return self._create_error_response("Invalid model response format")
        except Exception as e:
            logger.error(f"Error in query parsing: {str(e)}", exc_info=True)
            return self._create_error_response(str(e))

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
            "context_used": []
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
        logger.debug(f"Conversation complete: {is_complete}")
        logger.debug(f"Missing information: {[key for key in required_info if context.extracted_info.get(key) is None]}")
        return is_complete 