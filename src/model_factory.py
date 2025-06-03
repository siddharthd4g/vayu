from langchain_openai import ChatOpenAI
from langchain_ibm import ChatWatsonx
from langchain.schema import HumanMessage, SystemMessage
from config import (
    MODEL_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    IBM_CLOUD_API_KEY,
    IBM_CLOUD_ENDPOINT,
    IBM_CLOUD_PROJECT_ID,
    IBM_MODEL,
    MODEL_PARAMS
)
import logging

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


def get_chat_model(provider=None, granite_model=None, openai_model=None):
    """Get the appropriate chat model based on the configured provider."""
    # Use provided provider if available, otherwise fall back to config
    current_provider = provider if provider else MODEL_PROVIDER.value
    current_granite = granite_model if granite_model else IBM_MODEL
    current_openai = openai_model if openai_model else OPENAI_MODEL
    
    logger.info(f"Using model provider: {current_provider}")
    if current_provider == "ibm":
        logger.info(f"Using Granite model: {current_granite}")
    elif current_provider == "openai":
        logger.info(f"Using OpenAI model: {current_openai}")
        logger.info(f"OpenAI API Key present: {bool(OPENAI_API_KEY)}")
        logger.info(f"OpenAI API Key length: {len(OPENAI_API_KEY) if OPENAI_API_KEY else 0}")

    if current_provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found in environment variables")
        logger.info("OpenAI model getting created now...")
        return ChatOpenAI(
            model=current_openai,
            temperature=MODEL_PARAMS["temperature"],
            max_tokens=MODEL_PARAMS["max_tokens"],
            api_key=OPENAI_API_KEY
        )
    elif current_provider == "ibm":
        if not all([IBM_CLOUD_API_KEY, IBM_CLOUD_ENDPOINT, IBM_CLOUD_PROJECT_ID]):
            raise ValueError("IBM Cloud credentials not found in environment variables")
        logger.info("IBM model getting created now...")
        return ChatWatsonx(
            model_id=current_granite,
            url=IBM_CLOUD_ENDPOINT,
            api_key=IBM_CLOUD_API_KEY,
            project_id=IBM_CLOUD_PROJECT_ID,
            temperature=MODEL_PARAMS["temperature"],
            max_tokens=MODEL_PARAMS["max_tokens"],
            top_p=MODEL_PARAMS["top_p"],
            top_k=MODEL_PARAMS["top_k"],
            repetition_penalty=MODEL_PARAMS["repetition_penalty"]
        )
    else:
        raise ValueError(f"Unsupported model provider: {current_provider}")

def get_model_response(prompt: str, system_message: str = None, provider=None, granite_model=None, openai_model=None) -> str:
    """Get a response from the configured model."""
    chat = get_chat_model(provider=provider, granite_model=granite_model, openai_model=openai_model)
    messages = []
    
    # Log model identifier
    if hasattr(chat, 'model_name'):
        logger.info(f"Using model: {chat.model_name}")
    elif hasattr(chat, 'model_id'):
        logger.info(f"Using model: {chat.model_id}")
    else:
        logger.info("Model identifier not found")
    
    if system_message:
        messages.append(SystemMessage(content=system_message))
    
    messages.append(HumanMessage(content=prompt))
    
    response = chat.invoke(messages)
    return response.content 