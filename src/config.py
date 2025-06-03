import os
from dotenv import load_dotenv
from enum import Enum
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

# Load environment variables
load_dotenv()

class ModelProvider(Enum):
    OPENAI = "openai"
    IBM = "ibm"

# Get model provider from environment variable, default to IBM
try:
    provider_value = os.getenv("MODEL_PROVIDER", "ibm").lower().strip()
    logger.info(f"Raw MODEL_PROVIDER value from env: {os.getenv('MODEL_PROVIDER')}")
    logger.info(f"Processed MODEL_PROVIDER value: {provider_value}")
    MODEL_PROVIDER = ModelProvider(provider_value)
    logger.info(f"Successfully initialized MODEL_PROVIDER: {MODEL_PROVIDER.value}")
except ValueError as e:
    logger.error(f"Invalid MODEL_PROVIDER value: {provider_value}")
    logger.error(f"Error: {str(e)}")
    logger.info("Defaulting to IBM provider")
    MODEL_PROVIDER = ModelProvider.IBM

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# Available OpenAI model variants
OPENAI_MODELS = {
    "GPT-3.5 Turbo": "gpt-3.5-turbo",
    "GPT-4o": "gpt-4o",
    "GPT-4 Turbo": "gpt-4-turbo",
    "GPT-4o Mini": "gpt-4o-mini"
}
# IBM Watson Configuration
IBM_CLOUD_API_KEY = os.getenv("IBM_CLOUD_API_KEY")
IBM_CLOUD_ENDPOINT = os.getenv("IBM_CLOUD_ENDPOINT")
IBM_CLOUD_PROJECT_ID = os.getenv("IBM_CLOUD_PROJECT_ID")
IBM_MODEL = os.getenv("IBM_MODEL", "ibm/granite-13b-chat-v2")

# Available Granite model variants
GRANITE_MODELS = {
    "Granite 13B Instruct": "ibm/granite-13b-instruct-v2",
    "Granite 20B Code Instruct": "ibm/granite-20b-code-instruct",
    "Granite 3.2 8B Instruct": "ibm/granite-3-2-8b-instruct",
    "Granite 3.2B Instruct": "ibm/granite-3-2b-instruct",
    "Granite 3.3 8B Instruct": "ibm/granite-3-3-8b-instruct",
    "Granite 3.8B Instruct": "ibm/granite-3-8b-instruct",
    "Granite 34B Code Instruct": "ibm/granite-34b-code-instruct",
    "Granite 3B Code Instruct": "ibm/granite-3b-code-instruct",
    "Granite 8B Code Instruct": "ibm/granite-8b-code-instruct",
    "Granite Guardian 3.2B": "ibm/granite-guardian-3-2b",
    "Granite Guardian 3.8B": "ibm/granite-guardian-3-8b",
    "Granite Vision 3.2 2B": "ibm/granite-vision-3-2-2b"
}

# Feature flags
raw_show_selector = os.getenv("SHOW_MODEL_SELECTOR", "false")
logger.info(f"Raw SHOW_MODEL_SELECTOR value from env: {raw_show_selector}")
logger.info(f"SHOW_MODEL_SELECTOR type: {type(raw_show_selector)}")
SHOW_MODEL_SELECTOR = str(raw_show_selector).lower() in ("true", "1", "yes", "y")
logger.info(f"Processed SHOW_MODEL_SELECTOR value: {SHOW_MODEL_SELECTOR}")
logger.info(f"SHOW_MODEL_SELECTOR type after processing: {type(SHOW_MODEL_SELECTOR)}")

# Model Parameters
MODEL_PARAMS = {
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.0
} 