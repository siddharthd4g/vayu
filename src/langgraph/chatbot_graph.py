from typing import Dict, List, Tuple, Any, Union, TypedDict, Optional
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.query_parser.query_parser_tool import QueryParserTool, ConversationContext
from langgraph.tools.weather_tool import WeatherTool
from langgraph.tools.medical_research_tool import MedicalResearchTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.model_factory import get_model_response
import logging
import traceback
from config import (
    MODEL_PROVIDER,
    SHOW_MODEL_SELECTOR,
    GRANITE_MODELS,
    OPENAI_MODELS,
    IBM_MODEL,
    OPENAI_MODEL
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('chatbot.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize tools
query_parser = QueryParserTool()
weather_tool = WeatherTool()
medical_tool = MedicalResearchTool()

class AgentState(TypedDict):
    messages: List[Dict[str, str]]  # List of message dictionaries with role and content
    conversation_context: ConversationContext
    parsed_query: Dict[str, Any]
    weather_data: Dict[str, Any]
    medical_info: List[Dict[str, Any]]
    model_preferences: Optional[Dict[str, Any]]
    validation_status: Dict[str, bool]

def get_default_state() -> AgentState:
    """Create a default state with all required fields initialized."""
    # Initialize model preferences based on environment variables
    if SHOW_MODEL_SELECTOR:
        # When model selector is enabled, use default values that match the UI defaults
        model_preferences = {
            "provider": "ibm",  # Default to IBM
            "granite_model": "ibm/granite-3-2b-instruct",  # Default to Granite 3.2B Instruct
            "granite_model_label": "Granite 3.2B Instruct"
        }
    else:
        # When model selector is disabled, use environment variables
        model_preferences = {
            "provider": MODEL_PROVIDER.value,
            "granite_model": IBM_MODEL if MODEL_PROVIDER.value == "ibm" else None,
            "openai_model": OPENAI_MODEL if MODEL_PROVIDER.value == "openai" else None
        }

    return {
        "messages": [],
        "conversation_context": ConversationContext(
            user_info={"name": "", "conditions": []},
            previous_queries=[],
            extracted_info={},
            last_intent=None,
            last_required_actions=None
        ),
        "parsed_query": {},
        "weather_data": {},
        "medical_info": [],
        "model_preferences": model_preferences,
        "validation_status": {
            "is_complete": False,
            "needs_weather": False,
            "needs_medical": False
        }
    }

def parse_query(state: AgentState) -> AgentState:
    """Parse the user's query using the QueryParserTool."""
    try:
        # Get the last message
        if not state["messages"]:
            logger.warning("No messages in state")
            return state

        last_msg = state["messages"][-1]
        last_message = last_msg.get("content", None) if isinstance(last_msg, dict) else last_msg.content
        logger.info(f"Last message extracted, content: {last_message}")
        model_preferences = state.get("model_preferences", {})
        logger.info(f"Model Preference Value: {model_preferences}")

        # Get model response for parsing
        model_response = get_model_response(
            query_parser.create_parser_prompt(last_message, state["conversation_context"]),
            system_message=query_parser.system_prompt,
            provider=state.get("model_preferences", {}).get("provider"),
            granite_model=state.get("model_preferences", {}).get("granite_model"),
            openai_model=state.get("model_preferences", {}).get("openai_model")
        )

        # Parse the query and get the result
        parsed_result = query_parser.parse_query(
            query=last_message,
            context=state["conversation_context"],
            model_response=model_response
        )

        # Update state with parsed query
        state["parsed_query"] = parsed_result

        # Update conversation context based on the parsed result
        if parsed_result:
            # Update the conversation context with information from the parsed result
            state["conversation_context"].previous_queries.append(parsed_result)
            if "extracted_info" in parsed_result:
                state["conversation_context"].extracted_info.update(parsed_result["extracted_info"])
            state["conversation_context"].last_intent = parsed_result.get("intent")
            state["conversation_context"].last_required_actions = parsed_result.get("required_actions")

        return state
    except Exception as e:
        logger.error(f"Error in parse_query: {str(e)}")
        logger.error(traceback.format_exc())
        return state

def get_weather_data(state: AgentState) -> AgentState:
    """Get weather data if the query is about weather."""
    try:
        if not state.get("parsed_query"):
            logger.warning("No parsed query in state")
            return state

        parsed_query = state["parsed_query"]
        if parsed_query["required_actions"]["needs_weather_data"]:
            location = parsed_query["extracted_info"]["location"]
            date_range = parsed_query["extracted_info"]["date_range"]
            
            if location:
                logger.info(f"Getting weather data for {location}")
                weather_data = weather_tool.get_weather(
                    location=location,
                    start_date=date_range.get("start"),
                    end_date=date_range.get("end")
                )
                state["weather_data"] = weather_data
                logger.info("Weather data retrieved successfully")
            else:
                logger.warning("No location found in parsed query")

        return state
    except Exception as e:
        logger.error(f"Error in get_weather_data: {str(e)}")
        return state

def get_medical_info(state: AgentState) -> AgentState:
    """Get medical information if the query is about health."""
    try:
        if not state.get("parsed_query"):
            logger.warning("No parsed query in state")
            return state

        parsed_query = state["parsed_query"]
        if parsed_query["required_actions"]["needs_medical_research"]:
            # Get user's health conditions from conversation context
            conditions = state["conversation_context"].user_info.get("conditions", [])
            
            if conditions:
                logger.info(f"Getting medical info for conditions: {conditions}")
                medical_info = []
                for condition in conditions:
                    info = medical_tool.get_medical_info(condition=condition)
                    if info:
                        medical_info.extend(info)
                state["medical_info"] = medical_info
                logger.info("Medical info retrieved successfully")
            else:
                logger.warning("No health conditions found in user info")

        return state
    except Exception as e:
        logger.error(f"Error in get_medical_info: {str(e)}")
        return state

def format_weather_response(weather_data: Dict[str, Any]) -> str:
    """Format weather data into a readable response."""
    if not weather_data:
        return "I couldn't retrieve the weather information."
    
    return f"""Current weather in {weather_data.get('location', 'the specified location')}:
Temperature: {weather_data.get('temperature', 'N/A')}°C
Conditions: {weather_data.get('conditions', 'N/A')}
Humidity: {weather_data.get('humidity', 'N/A')}%
Wind Speed: {weather_data.get('wind_speed', 'N/A')} km/h"""

def format_medical_response(medical_info: List[Dict[str, Any]]) -> str:
    """Format medical information into a readable response."""
    if not medical_info:
        return "I couldn't find specific medical information for your query."
    
    response = "Here's what I found:\n\n"
    for info in medical_info:
        response += f"• {info.get('title', 'N/A')}\n"
        response += f"  {info.get('summary', 'No summary available')}\n\n"
    return response

def generate_response(state: AgentState) -> AgentState:
    """Generate a response based on the parsed query and gathered information."""
    try:
        if not state.get("parsed_query"):
            logger.warning("No parsed query in state")
            return state

        parsed_query = state["parsed_query"]
        response = ""

        # If the query is incomplete, use the response from QueryParserTool
        if not parsed_query.get("is_complete", False):
            response = parsed_query["response"]["text"]
        else:
            # Generate response based on gathered data
            if parsed_query["required_actions"]["needs_weather_data"]:
                if not state.get("weather_data"):
                    response = "I'm having trouble getting the weather information. Please try again."
                else:
                    weather_response = format_weather_response(state["weather_data"])
                    response += weather_response + "\n\n"

            if parsed_query["required_actions"]["needs_medical_research"]:
                if not state.get("medical_info"):
                    response += "\nI'm having trouble getting the medical information. Please try again."
                else:
                    medical_response = format_medical_response(state["medical_info"])
                    response += medical_response

            # If we still don't have a response, something went wrong
            if not response:
                logger.error("No response generated despite complete query")
                response = "I'm having trouble processing your request. Please try again with a different query."

        # Add the response to messages
        state["messages"].append({
            "role": "assistant",
            "content": response
        })

        return state
    except Exception as e:
        logger.error(f"Error in generate_response: {str(e)}")
        return state

def validate_info(state: AgentState) -> AgentState:
    """Update state with validation information."""
    try:
        parsed = state["parsed_query"]
        logger.info(f"Validating query with intent: {parsed.get('intent')}")
        
        # Add validation status to state
        state["validation_status"] = {
            "is_complete": parsed.get("is_complete", False),
            "needs_weather": parsed["required_actions"]["needs_weather_data"],
            "needs_medical": parsed["required_actions"]["needs_medical_research"]
        }
        
        logger.info(f"Validation status: {state['validation_status']}")
        return state
    except Exception as e:
        logger.error(f"Error in validate_info: {str(e)}")
        state["validation_status"] = {
            "is_complete": False,
            "needs_weather": False,
            "needs_medical": False
        }
        return state

def get_next_node(state: AgentState) -> str:
    """Determine the next node based on validation status."""
    try:
        validation = state.get("validation_status", {})
        
        if not validation.get("is_complete", False):
            logger.info("Query incomplete, requesting missing information")
            return "ask_missing_info"
        
        if validation.get("needs_weather", False):
            logger.info("Query needs weather data, proceeding to weather tool")
            return "get_weather_data"
        
        logger.info("Query complete, generating response")
        return "generate_response"
    except Exception as e:
        logger.error(f"Error in get_next_node: {str(e)}")
        return "generate_response"

def ask_missing_info(state: AgentState) -> AgentState:
    """Use parsed_query's response to ask for missing info."""
    try:
        parsed = state["parsed_query"]
        logger.info("Asking for missing information")
        
        # Add the response from QueryParserTool to messages
        state["messages"].append({
            "role": "assistant",
            "content": parsed["response"]["text"]
        })
        
        return state
    except Exception as e:
        logger.error(f"Error in ask_missing_info: {str(e)}")
        return state

def create_chatbot() -> StateGraph:
    """Create the chatbot graph."""
    # Create the graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("parse_query", parse_query)
    graph.add_node("validate_info", validate_info)
    graph.add_node("ask_missing_info", ask_missing_info)
    graph.add_node("get_weather_data", get_weather_data)
    graph.add_node("get_medical_info", get_medical_info)
    graph.add_node("generate_response", generate_response)

    # Add conditional edges based on validation status
    graph.add_conditional_edges(
        "validate_info",
        get_next_node,
        {
            "ask_missing_info": "ask_missing_info",
            "get_weather_data": "get_weather_data",
            "generate_response": "generate_response"
        }
    )

    # Add regular edges
    graph.add_edge("parse_query", "validate_info")
    graph.add_edge("ask_missing_info", "parse_query")  # Loop back for missing info
    graph.add_edge("get_weather_data", "get_medical_info")
    graph.add_edge("get_medical_info", "generate_response")

    # Set entry point
    graph.set_entry_point("parse_query")

    return graph.compile(interrupt_after=["ask_missing_info"]) 