import streamlit as st
import os
import json
import logging
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.chatbot_graph import create_chatbot, get_default_state
from langgraph.query_parser.query_parser_tool import ConversationContext
from config import (
    MODEL_PROVIDER,
    SHOW_MODEL_SELECTOR,
    GRANITE_MODELS,
    OPENAI_MODELS,
    IBM_MODEL,
    OPENAI_MODEL
)
from frontend.weather_visualization import display_weather_data
from frontend.medical_quotes import display_medical_quotes
from typing import Dict, List
import torch

# DO NOT DELETE THE LINE BELOW
torch.classes.__path__ = [] #

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Vayu - Weather the Change",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
RESPIRATORY_CONDITIONS = [
    "Asthma",
    "COPD",
    "Bronchitis",
    "Sinusitis",
    "Allergic Rhinitis",
    "Other"
]

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextInput>div>div>input {
        font-size: 1.2rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b313e;
    }
    .chat-message.assistant {
        background-color: #475063;
    }
    .chat-message .content {
        display: flex;
        margin-top: 0.5rem;
    }
    .error-message {
        color: #ff4b4b;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: rgba(255, 75, 75, 0.1);
        margin: 1rem 0;
    }
    .danger-button {
        background-color: #ff4b4b !important;
        color: white !important;
    }
    .stDialog {
        background-color: #2b313e;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "agent_state" not in st.session_state:
    logger.info("Initializing agent state")
    st.session_state.agent_state = get_default_state()

def clear_chat_history():
    """Clear only the chat history while preserving model preferences."""
    logger.info("Clearing chat history")
    # Reset messages in agent state
    st.session_state.agent_state["messages"] = []
    
    # Reset conversation context but keep user info
    st.session_state.agent_state["conversation_context"] = ConversationContext(
        user_info=st.session_state.agent_state["conversation_context"].user_info,
        previous_queries=[],
        extracted_info={},
        last_intent=None,
        last_required_actions=None
    )

def logout():
    """Complete reset of all session state and preferences."""
    logger.info("Performing logout - clearing all session state")
    # Clear all session state variables
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Reinitialize with default values
    st.session_state.agent_state = get_default_state()
    logger.info("Logout complete - reset to initial state")
    st.rerun()

def handle_api_error(error: Exception) -> str:
    """Handle different types of API errors and return user-friendly messages."""
    error_str = str(error).lower()
    logger.error(f"API Error: {str(error)}")
    
    if "api key" in error_str:
        return "API key is missing or invalid. Please check your API credentials in the .env file."
    elif "endpoint" in error_str:
        return "API endpoint is not accessible. Please check your internet connection and endpoint URL."
    elif "project" in error_str:
        return "Project ID is missing or invalid. Please check your IBM Cloud project settings."
    elif "timeout" in error_str:
        return "Request timed out. The service might be experiencing high load. Please try again in a few minutes."
    elif "rate limit" in error_str:
        return "Rate limit exceeded. Please wait a few minutes before trying again."
    else:
        return f"An error occurred: {str(error)}"

def process_message(message: str) -> str:
    """Process a user message and return the AI response."""
    try:
        logger.info(f"New chat message received: {message}")
        
        # Get the current agent state
        current_state = st.session_state.agent_state
        
        # Process the message through the graph
        response = create_chatbot().invoke(current_state)
        
        logger.info("Received response from AI model")
        
        # Update the agent state with the response
        st.session_state.agent_state = response
        
        # Get the last message from the response
        if response["messages"]:
            return response["messages"][-1]["content"]
        return "I apologize, but I couldn't generate a response."
    except Exception as e:
        logger.error(f"Error in AI response: {str(e)}")
        return f"Error: {str(e)}"

# Main content
st.title("🌪️ Vayu")
st.markdown("### Weather the Change")

# User information form
if not st.session_state.agent_state["conversation_context"].user_info.get("name"):
    logger.info("Showing user information form")
    with st.form("user_info_form", clear_on_submit=True):
        st.subheader("Welcome to Your Travel Health Assistant")
        st.markdown("""
        Vayu helps you understand how weather conditions might affect your respiratory health during travel.
        Please tell us about yourself to get personalized travel health advisories.
        """)
        
        name = st.text_input("Enter your name")
        conditions = st.multiselect(
            "Select your respiratory conditions",
            RESPIRATORY_CONDITIONS
        )
        
        submitted = st.form_submit_button("Start Your Journey")
        if submitted:
            if name:  # Basic validation
                logger.info(f"User info submitted - Name: {name}, Conditions: {conditions}")
                st.session_state.agent_state["conversation_context"].user_info["name"]=name
                st.session_state.agent_state["conversation_context"].user_info["conditions"]=conditions
                st.rerun()
            else:
                logger.warning("Form submitted without name")
                st.error("Please enter your name")
else:
    logger.info(f"User {st.session_state.agent_state['conversation_context'].user_info.get('name')} logged in, showing main app")
    st.markdown("Your AI companion for respiratory health and travel advisories")

    # Sidebar
    with st.sidebar:
        st.title("🌪️ Vayu")
        st.markdown("### Weather the Change")
        st.markdown("---")
        
        # User info display
        st.subheader("Your Health Profile")
        st.write(f"👤 {st.session_state.agent_state['conversation_context'].user_info.get('name')}")
        st.write("🏥 Conditions:", ", ".join(st.session_state.agent_state['conversation_context'].user_info.get("conditions")))
        st.markdown("---")
        
        # Model Settings section
        if SHOW_MODEL_SELECTOR:
            st.subheader("Model Settings")
            current_provider = st.selectbox(
                "Select Model Provider",
                ["IBM Watson", "OpenAI"],
                index=0 if st.session_state.agent_state["model_preferences"]["provider"] == "ibm" else 1,
                key="provider_selector"
            )
            
            # Update provider in session state
            new_provider = "ibm" if current_provider == "IBM Watson" else "openai"
            if new_provider != st.session_state.agent_state["model_preferences"]["provider"]:
                logger.info(f"Switching model provider from {st.session_state.agent_state['model_preferences']['provider']} to {new_provider}")
                st.session_state.agent_state["model_preferences"]["provider"] = new_provider
            
            # Show model variant selector
            st.markdown("---")
            st.subheader("Model Variant")
            if current_provider == "IBM Watson":
                selected_model_label = st.selectbox(
                    "Select Granite Model",
                    options=list(GRANITE_MODELS.keys()),
                    index=list(GRANITE_MODELS.keys()).index(st.session_state.agent_state["model_preferences"].get("granite_model_label", "Granite 3.2B Instruct")),
                    key="model_selector"
                )
                selected_model_id = GRANITE_MODELS[selected_model_label]
                if selected_model_id != st.session_state.agent_state["model_preferences"].get("granite_model"):
                    logger.info(f"Switching Granite model to {selected_model_label} ({selected_model_id})")
                    st.session_state.agent_state["model_preferences"]["granite_model"] = selected_model_id
                    st.session_state.agent_state["model_preferences"]["granite_model_label"] = selected_model_label
            else:  # OpenAI
                selected_model_label = st.selectbox(
                    "Select OpenAI Model",
                    options=list(OPENAI_MODELS.keys()),
                    index=list(OPENAI_MODELS.keys()).index(st.session_state.agent_state["model_preferences"].get("openai_model_label", list(OPENAI_MODELS.keys())[0])),
                    key="openai_model_selector"
                )
                selected_model_id = OPENAI_MODELS[selected_model_label]
                if selected_model_id != st.session_state.agent_state["model_preferences"].get("openai_model"):
                    logger.info(f"Switching OpenAI model to {selected_model_label} ({selected_model_id})")
                    st.session_state.agent_state["model_preferences"]["openai_model"] = selected_model_id
                    st.session_state.agent_state["model_preferences"]["openai_model_label"] = selected_model_label
        
        st.markdown("---")
        st.markdown("### Quick Actions")
        
        # Chat history clear button
        if st.button("🗑️ Clear Chat History", on_click=clear_chat_history):
            pass
        
        # Logout button with confirmation
        if st.button("🚪 Logout (Reset All)", type="primary", use_container_width=True):
            logger.info("Logout button clicked")
            st.session_state.show_logout_confirm = True
        
        # Logout confirmation dialog
        if st.session_state.get("show_logout_confirm", False):
            st.markdown("---")
            st.warning("Are you sure you want to logout? This will reset all settings and clear chat history.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, Logout", type="primary", use_container_width=True):
                    logout()
            with col2:
                if st.button("Cancel", use_container_width=True):
                    logger.info("Logout cancelled")
                    st.session_state.show_logout_confirm = False
                    st.rerun()
        
        st.markdown("---")
        st.markdown("### Example Questions")
        st.markdown("""
        - What's the air quality in New York today?
        - Should I travel to Delhi next week?
        - How's the pollen count in London?
        - Is it safe to visit Beijing this month?
        - What's the air quality forecast for Tokyo?
        """)

    # Display chat messages
    for message in st.session_state.agent_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about weather conditions and travel advisories..."):
        # Add user message to chat history
        st.session_state.agent_state["messages"].append({"role": "user", "content": prompt})
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = process_message(prompt)
                    st.markdown(response)
                    #st.session_state.agent_state["messages"].append({"role": "assistant", "content": response})
                except Exception as e:
                    error_message = handle_api_error(e)
                    logger.error(f"Error in AI response: {str(e)}")
                    st.markdown(f'<div class="error-message">{error_message}</div>', unsafe_allow_html=True)
                    st.info("Please check your API keys and model settings in the sidebar.")
        
        # Rerun to update the chat display
        st.rerun() 