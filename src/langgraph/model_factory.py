from typing import Any, Dict, List, Union
from langchain_core.messages import BaseMessage
from langchain_ibm import ChatWatsonx
from langchain_openai import ChatOpenAI
import os
import logging
from config import (
    MODEL_PROVIDER,
    IBM_MODEL,
    OPENAI_MODEL,
    MODEL_PARAMS,
    SHOW_MODEL_SELECTOR
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_factory.log')
    ]
)
logger = logging.getLogger(__name__)

def get_model_response(
    prompt: Union[str, List[BaseMessage]],
    system_message: str = None,
    provider: str = None,
    granite_model: str = None,
    openai_model: str = None
) -> str:
    """
    Get response from the specified model provider.
    
    Args:
        prompt: The prompt to send to the model
        system_message: Optional system message for the model
        provider: The model provider ("ibm" or "openai")
        granite_model: The specific IBM Granite model to use
        openai_model: The specific OpenAI model to use
    
    Returns:
        str: The model's response
    """
    # Determine which provider and model to use
    if SHOW_MODEL_SELECTOR:
        # When model selector is enabled, use the provided values
        current_provider = provider
        current_granite = granite_model
        current_openai = openai_model
    else:
        # When model selector is disabled, use environment variables
        current_provider = MODEL_PROVIDER.value
        current_granite = IBM_MODEL if current_provider == "ibm" else None
        current_openai = OPENAI_MODEL if current_provider == "openai" else None
    
    logger.info(f"Getting response from {current_provider} model")
    
    try:
        if current_provider == "ibm":
            return _get_ibm_response(prompt, system_message, current_granite)
        elif current_provider == "openai":
            return _get_openai_response(prompt, system_message, current_openai)
        else:
            raise ValueError(f"Unsupported provider: {current_provider}")
    except Exception as e:
        logger.error(f"Error getting model response: {str(e)}")
        raise

def _get_ibm_response(
    prompt: Union[str, List[BaseMessage]],
    system_message: str = None,
    model_id: str = None
) -> str:
    """Get response from IBM Watson model."""
    logger.info(f"Initializing IBM Watson model with model_id: {model_id}")
    
    # Initialize Watson model
    chat = ChatWatsonx(
        model_id=model_id,
        url=os.getenv("IBM_CLOUD_ENDPOINT"),
        project_id=os.getenv("IBM_CLOUD_PROJECT_ID"),
        params={
            "decoding_method": "greedy",
            "max_new_tokens": MODEL_PARAMS["max_tokens"],
            "min_new_tokens": 1,
            "temperature": MODEL_PARAMS["temperature"],
            "top_p": MODEL_PARAMS["top_p"],
            "top_k": MODEL_PARAMS["top_k"],
            "repetition_penalty": MODEL_PARAMS["repetition_penalty"]
        },
        api_key=os.getenv("IBM_CLOUD_API_KEY")
    )
    
    # Add system message if provided
    if system_message:
        if isinstance(prompt, list):
            prompt = [{"role": "system", "content": system_message}] + prompt
        else:
            prompt = f"System: {system_message}\n\n{prompt}"
    
    # Get response
    response = chat.invoke(prompt)
    logger.info("Received response from IBM Watson")
    return response.content

def _get_openai_response(
    prompt: Union[str, List[BaseMessage]],
    system_message: str = None,
    model: str = None
) -> str:
    """Get response from OpenAI model."""
    logger.info(f"Initializing OpenAI model with model: {model}")
    
    # Initialize OpenAI model
    chat = ChatOpenAI(
        model=model,
        temperature=MODEL_PARAMS["temperature"],
        max_tokens=MODEL_PARAMS["max_tokens"],
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Add system message if provided
    if system_message:
        if isinstance(prompt, list):
            prompt = [{"role": "system", "content": system_message}] + prompt
        else:
            prompt = f"System: {system_message}\n\n{prompt}"
    
    # Get response
    response = chat.invoke(prompt)
    logger.info("Received response from OpenAI")
    return response.content 