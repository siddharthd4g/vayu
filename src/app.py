import streamlit as st
import os
import json
import logging
from dotenv import load_dotenv
from model_factory import get_model_response
from config import (
    MODEL_PROVIDER,
    SHOW_MODEL_SELECTOR,
    GRANITE_MODELS,
    OPENAI_MODELS,
    IBM_MODEL,
    OPENAI_MODEL
)
import torch

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

# Log configuration values
logger.info(f"SHOW_MODEL_SELECTOR value from env: {os.getenv('SHOW_MODEL_SELECTOR')}")
logger.info(f"SHOW_MODEL_SELECTOR parsed value: {SHOW_MODEL_SELECTOR}")

# DO NOT REMOVE THIS LINE - Fix for torch classes
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

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

def load_from_local_storage():
    """Load all data from local storage."""
    try:
        logger.info("Attempting to load data from local storage")
        # Load chat history
        if "chat_history" in st.session_state:
            logger.info("Found chat history in session state")
            st.session_state.messages = st.session_state.chat_history
        else:
            logger.info("No chat history found, initializing empty list")
            st.session_state.messages = []
        
        # Load user info
        if "user_info" in st.session_state:
            logger.info("Found user info in session state")
            return st.session_state.user_info
        logger.info("No user info found, returning default")
        return {"name": "", "conditions": []}
    except Exception as e:
        logger.error(f"Error loading from local storage: {str(e)}")
        return {"name": "", "conditions": []}

def save_to_local_storage():
    """Save all data to local storage."""
    try:
        logger.info("Saving data to local storage")
        # Save chat history
        st.session_state.chat_history = st.session_state.messages
        logger.info(f"Saved {len(st.session_state.messages)} chat messages")
        
        # Save user info
        st.session_state.user_info = st.session_state.user_info
        logger.info(f"Saved user info for: {st.session_state.user_info['name']}")
    except Exception as e:
        logger.error(f"Error saving to local storage: {str(e)}")

# Initialize session state
if "messages" not in st.session_state:
    logger.info("Initializing messages in session state")
    st.session_state.messages = []

if "model_preferences" not in st.session_state:
    logger.info("Initializing model preferences in session state")
    st.session_state.model_preferences = {
        "provider": MODEL_PROVIDER.value,
        "granite_model": list(GRANITE_MODELS.keys())[0] if SHOW_MODEL_SELECTOR else None
    }

if "user_info" not in st.session_state:
    logger.info("Loading user info from local storage")
    st.session_state.user_info = load_from_local_storage()

def clear_chat_history():
    """Clear only the chat history while preserving model preferences."""
    logger.info("Clearing chat history")
    st.session_state.messages = []
    save_to_local_storage()

def logout():
    """Complete reset of all session state and preferences."""
    logger.info("Performing logout - clearing all session state")
    # Clear all session state variables
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Reinitialize with default values
    st.session_state.messages = []
    st.session_state.model_preferences = {
        "provider": MODEL_PROVIDER.value,
        "granite_model": list(GRANITE_MODELS.keys())[0] if SHOW_MODEL_SELECTOR else None
    }
    st.session_state.user_info = {"name": "", "conditions": []}
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

# Main content
st.title("🌪️ Vayu")
st.markdown("### Weather the Change")

# User information form
if not st.session_state.user_info["name"]:
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
                st.session_state.user_info["name"] = name
                st.session_state.user_info["conditions"] = conditions
                save_to_local_storage()
                st.rerun()
            else:
                logger.warning("Form submitted without name")
                st.error("Please enter your name")
else:
    logger.info(f"User {st.session_state.user_info['name']} logged in, showing main app")
    st.markdown("Your AI companion for respiratory health and travel advisories")

    # Sidebar
    with st.sidebar:
        st.title("🌪️ Vayu")
        st.markdown("### Weather the Change")
        st.markdown("---")
        
        # User info display
        st.subheader("Your Health Profile")
        st.write(f"👤 {st.session_state.user_info['name']}")
        st.write("🏥 Conditions:", ", ".join(st.session_state.user_info["conditions"]))
        st.markdown("---")
        
        # Model Settings section
        if SHOW_MODEL_SELECTOR:
            st.subheader("Model Settings")
            current_provider = st.selectbox(
                "Select Model Provider",
                ["IBM Watson", "OpenAI"],
                index=0 if st.session_state.model_preferences["provider"] == "ibm" else 1,
                key="provider_selector"
            )
            
            # Update environment variable and session state
            new_provider = "ibm" if current_provider == "IBM Watson" else "openai"
            if new_provider != st.session_state.model_preferences["provider"]:
                logger.info(f"Switching model provider from {st.session_state.model_preferences['provider']} to {new_provider}")
                st.session_state.model_preferences["provider"] = new_provider
                os.environ["MODEL_PROVIDER"] = new_provider
                # Clear any existing model instances
                if "chat_model" in st.session_state:
                    logger.info("Clearing cached model instance")
                    del st.session_state.chat_model
            
            # Show model variant selector
            st.markdown("---")
            st.subheader("Model Variant")
            if current_provider == "IBM Watson":
                selected_model_label = st.selectbox(
                    "Select Granite Model",
                    options=list(GRANITE_MODELS.keys()),
                    index=list(GRANITE_MODELS.keys()).index(st.session_state.model_preferences.get("granite_model_label", list(GRANITE_MODELS.keys())[0])),
                    key="model_selector"
                )
                selected_model_id = GRANITE_MODELS[selected_model_label]
                if selected_model_id != st.session_state.model_preferences.get("granite_model"):
                    logger.info(f"Switching Granite model to {selected_model_label} ({selected_model_id})")
                    st.session_state.model_preferences["granite_model"] = selected_model_id
                    st.session_state.model_preferences["granite_model_label"] = selected_model_label
                    os.environ["IBM_MODEL"] = selected_model_id
                    if "chat_model" in st.session_state:
                        logger.info("Clearing cached model instance")
                        del st.session_state.chat_model
            else:  # OpenAI
                selected_model_label = st.selectbox(
                    "Select OpenAI Model",
                    options=list(OPENAI_MODELS.keys()),
                    index=list(OPENAI_MODELS.keys()).index(st.session_state.model_preferences.get("openai_model_label", list(OPENAI_MODELS.keys())[0])),
                    key="openai_model_selector"
                )
                selected_model_id = OPENAI_MODELS[selected_model_label]
                if selected_model_id != st.session_state.model_preferences.get("openai_model"):
                    logger.info(f"Switching OpenAI model to {selected_model_label} ({selected_model_id})")
                    st.session_state.model_preferences["openai_model"] = selected_model_id
                    st.session_state.model_preferences["openai_model_label"] = selected_model_label
                    os.environ["OPENAI_MODEL"] = selected_model_id
                    if "chat_model" in st.session_state:
                        logger.info("Clearing cached model instance")
                        del st.session_state.chat_model
        else:
            logger.info("Model selector is disabled, using default model from environment variables")
            # Use the model IDs from environment variables
            if MODEL_PROVIDER.value == "ibm":
                st.session_state.model_preferences["provider"] = "ibm"
                st.session_state.model_preferences["granite_model"] = IBM_MODEL
                logger.info(f"Using IBM model from env: {IBM_MODEL}")
            else:  # OpenAI
                st.session_state.model_preferences["provider"] = "openai"
                st.session_state.model_preferences["openai_model"] = OPENAI_MODEL
                logger.info(f"Using OpenAI model from env: {OPENAI_MODEL}")
        
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
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about weather conditions and travel advisories..."):
        logger.info(f"New chat message received: {prompt[:50]}...")
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_to_local_storage()
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    logger.info(f"Calling AI model: {st.session_state.model_preferences['provider']}")
                    if st.session_state.model_preferences["provider"] == "ibm":
                        logger.info(f"Using Granite model: {st.session_state.model_preferences['granite_model']}")
                    
                    response = get_model_response(
                        prompt,
                        system_message="You are Vayu, an AI assistant focused on providing weather-based travel health advisories. You help users make informed decisions about travel based on weather conditions, air quality, and respiratory health factors. Be concise, informative, and always prioritize user health and safety.",
                        provider=st.session_state.model_preferences["provider"],
                        granite_model=st.session_state.model_preferences.get("granite_model"),
                        openai_model=st.session_state.model_preferences.get("openai_model")
                    )
                    logger.info("Received response from AI model")
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    save_to_local_storage()
                except Exception as e:
                    error_message = handle_api_error(e)
                    logger.error(f"Error in AI response: {str(e)}")
                    st.markdown(f'<div class="error-message">{error_message}</div>', unsafe_allow_html=True)
                    st.info("Please check your API keys and model settings in the sidebar.") 