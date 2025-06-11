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
        "model_preferences": model_preferences
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
            return state

        parsed_query = state["parsed_query"]
        if parsed_query.get("intent") == "weather_inquiry":
            location = parsed_query.get("extracted_info", {}).get("location")
            if location:
                weather_data = weather_tool.get_weather(location=location)
                state["weather_data"] = weather_data

        return state
    except Exception as e:
        logger.error(f"Error in get_weather_data: {str(e)}")
        return state

def get_medical_info(state: AgentState) -> AgentState:
    """Get medical information if the query is about health."""
    try:
        if not state.get("parsed_query"):
            return state

        parsed_query = state["parsed_query"]
        if parsed_query.get("intent") == "health_inquiry":
            condition = parsed_query.get("extracted_info", {}).get("condition")
            if condition:
                medical_info = medical_tool.get_medical_info(condition=condition)
                state["medical_info"] = medical_info

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
            return state

        # Get the last message for context
        last_msg = state["messages"][-1] if state["messages"] else None
        last_message = last_msg.get("content", None) if isinstance(last_msg, dict) else last_msg.content if last_msg else ""

        # Generate response based on intent and gathered data
        response = ""
        parsed_query = state["parsed_query"]
        
        if parsed_query.get("intent") == "weather_inquiry":
            response = format_weather_response(state["weather_data"])
        elif parsed_query.get("intent") == "health_inquiry":
            response = format_medical_response(state["medical_info"])
        else:
            response = "I understand your query. How can I help you further?"

        # Add the response to messages
        state["messages"].append({
            "role": "assistant",
            "content": response
        })

        return state
    except Exception as e:
        logger.error(f"Error in generate_response: {str(e)}")
        return state

def create_chatbot() -> StateGraph:
    """Create the chatbot graph."""
    # Create the graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("parse_query", parse_query)
    graph.add_node("get_weather_data", get_weather_data)
    graph.add_node("get_medical_info", get_medical_info)
    graph.add_node("generate_response", generate_response)

    # Add edges
    graph.add_edge("parse_query", "get_weather_data")
    graph.add_edge("get_weather_data", "get_medical_info")
    graph.add_edge("get_medical_info", "generate_response")

    # Set entry point
    graph.set_entry_point("parse_query")

    return graph.compile() 