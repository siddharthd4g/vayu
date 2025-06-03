from typing import Dict, List, Tuple, Any, Union, TypedDict
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ibm import ChatWatsonx
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# IBM Watson configuration
IBM_CLOUD_API_KEY = os.getenv("IBM_CLOUD_API_KEY")
IBM_CLOUD_ENDPOINT = os.getenv("IBM_CLOUD_ENDPOINT")
IBM_CLOUD_PROJECT_ID = os.getenv("IBM_CLOUD_PROJECT_ID")

# Define the state type
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    user_info: Dict[str, Any]
    weather_data: Dict[str, Any]
    medical_info: List[Dict[str, Any]]

def initialize_watson():
    """Initialize IBM watsonx.ai model."""
    parameters = {
        "decoding_method": "greedy",
        "max_new_tokens": 1000,
        "min_new_tokens": 1,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.0
    }
    
    chat = ChatWatsonx(
        model_id="meta-llama/llama-2-70b-chat",
        url=IBM_CLOUD_ENDPOINT,
        project_id=IBM_CLOUD_PROJECT_ID,
        params=parameters,
        api_key=IBM_CLOUD_API_KEY
    )
    
    return chat

def create_weather_tool():
    """Create a tool for querying weather data."""
    # TODO: Implement OpenWeather API integration
    pass

def create_medical_search_tool():
    """Create a tool for searching medical information."""
    # TODO: Integrate with our existing PDF search
    pass

def create_chatbot():
    """Create the LangGraph chatbot."""
    # Initialize the LLM
    llm = initialize_watson()

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that provides information about weather conditions 
        and their impact on respiratory health. You have access to:
        1. Current and forecasted weather data
        2. Medical information from research papers
        3. User's specific respiratory conditions
        
        Use this information to provide personalized advice and recommendations."""),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="user_info"),
        MessagesPlaceholder(variable_name="weather_data"),
        MessagesPlaceholder(variable_name="medical_info")
    ])

    # Define the nodes
    def should_use_tool(state: AgentState) -> Dict[str, Any]:
        """Determine if we need to use a tool."""
        # For now, always go to generate_response
        return {"next": "generate_response"}

    def generate_response(state: AgentState) -> AgentState:
        """Generate the final response."""
        messages = state["messages"]
        formatted_prompt = prompt.format_messages(
            messages=messages,
            user_info=state["user_info"],
            weather_data=state["weather_data"],
            medical_info=state["medical_info"]
        )
        
        # Generate response using Watson
        response = llm.invoke(formatted_prompt)
        state["messages"].append(response)
        return state

    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("should_use_tool", should_use_tool)
    workflow.add_node("generate_response", generate_response)

    # Add edges
    workflow.add_conditional_edges(
        "should_use_tool",
        should_use_tool,
        {
            "generate_response": "generate_response"
        }
    )

    # Set entry point
    workflow.set_entry_point("should_use_tool")

    # Compile the graph
    app = workflow.compile()

    return app

def process_message(message: str, user_info: Dict[str, Any]) -> str:
    """Process a user message and return a response."""
    # Initialize the state
    state = {
        "messages": [HumanMessage(content=message)],
        "user_info": user_info,
        "weather_data": {},
        "medical_info": []
    }

    # Run the graph
    result = create_chatbot().invoke(state)

    # Return the last message
    return result["messages"][-1].content 