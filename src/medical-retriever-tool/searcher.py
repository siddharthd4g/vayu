import os
import logging
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Elasticsearch configuration
ES_URL = os.getenv("ES_URL")
ES_USER = os.getenv("ES_USER")
ES_PASSWORD = os.getenv("ES_PASSWORD")
ES_CERT_FINGERPRINT = os.getenv("ES_CERT_FINGERPRINT")
MEDICAL_JOURNAL_INDEX_NAME = os.getenv("MEDICAL_JOURNAL_INDEX_NAME", "medical_journal")

# IBM Watson configuration
IBM_CLOUD_API_KEY = os.getenv("IBM_CLOUD_API_KEY")
IBM_CLOUD_ENDPOINT = os.getenv("IBM_CLOUD_ENDPOINT")
IBM_CLOUD_PROJECT_ID = os.getenv("IBM_CLOUD_PROJECT_ID")

# Validate required environment variables
required_vars = {
    "ES_URL": ES_URL,
    "ES_USER": ES_USER,
    "ES_PASSWORD": ES_PASSWORD,
    "ES_CERT_FINGERPRINT": ES_CERT_FINGERPRINT,
    "IBM_CLOUD_API_KEY": IBM_CLOUD_API_KEY,
    "IBM_CLOUD_ENDPOINT": IBM_CLOUD_ENDPOINT,
    "IBM_CLOUD_PROJECT_ID": IBM_CLOUD_PROJECT_ID,
    "MEDICAL_JOURNAL_INDEX_NAME": MEDICAL_JOURNAL_INDEX_NAME
}

missing_vars = [var for var, value in required_vars.items() if not value]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize Elasticsearch client
try:
    es = Elasticsearch(
        ES_URL,
        basic_auth=(ES_USER, ES_PASSWORD),
        verify_certs=True,
        ssl_assert_fingerprint=ES_CERT_FINGERPRINT,
        request_timeout=30,
        retry_on_timeout=True,
        max_retries=3
    )
except Exception as e:
    logger.error(f"Failed to initialize Elasticsearch client: {str(e)}", exc_info=True)
    raise

def initialize_watsonx():
    """Initialize IBM watsonx.ai embedding model."""
    model_id = "ibm/slate-125m-english-rtrvr-v2"
    embed_params = {
        EmbedParams.TRUNCATE_INPUT_TOKENS: 500,  # Restrict to 500 tokens
        EmbedParams.RETURN_OPTIONS: {
            'input_text': True
        }
    }
    credentials = {
        "url": IBM_CLOUD_ENDPOINT,
        "apikey": IBM_CLOUD_API_KEY,
    }
    embedding = Embeddings(
        model_id=model_id,
        credentials=credentials,
        params=embed_params,
        project_id=IBM_CLOUD_PROJECT_ID
    )
    return embedding

def search_documents(query, search_type="hybrid", k=5):
    """
    Search documents using the specified search type.
    
    Args:
        query (str): The search query
        search_type (str): One of "bm25", "vector", or "hybrid"
        k (int): Number of results to return
        
    Returns:
        list: List of search results
    """
    try:
        if search_type == "bm25":
            # BM25 search with image content
            response = es.search(
                index=MEDICAL_JOURNAL_INDEX_NAME,
                body={
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": [
                                "text^2",
                                "metadata.image_info.caption^1.5",
                                "metadata.image_info.description^1.2",
                                "metadata.image_info.ocr_text"
                            ]
                        }
                    },
                    "size": k
                }
            )
            
        elif search_type == "vector":
            # Vector search using IBM watsonx embeddings
            embeddings = initialize_watsonx()
            query_vector = embeddings.embed_query(query)
            
            response = es.search(
                index=MEDICAL_JOURNAL_INDEX_NAME,
                body={
                    "query": {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                                "params": {"query_vector": query_vector}
                            }
                        }
                    },
                    "size": k
                }
            )
            
        elif search_type == "hybrid":
            # Hybrid search combining BM25 and vector search with image content
            embeddings = initialize_watsonx()
            query_vector = embeddings.embed_query(query)
            
            response = es.search(
                index=MEDICAL_JOURNAL_INDEX_NAME,
                body={
                    "query": {
                        "bool": {
                            "should": [
                                {
                                    "multi_match": {
                                        "query": query,
                                        "fields": [
                                            "text^2",
                                            "metadata.image_info.caption^1.5",
                                            "metadata.image_info.description^1.2",
                                            "metadata.image_info.ocr_text"
                                        ]
                                    }
                                },
                                {
                                    "script_score": {
                                        "query": {"match_all": {}},
                                        "script": {
                                            "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                                            "params": {"query_vector": query_vector}
                                        }
                                    }
                                }
                            ]
                        }
                    },
                    "size": k
                }
            )
            
        else:
            raise ValueError(f"Invalid search type: {search_type}")
        
        # Process and return results
        results = []
        for hit in response["hits"]["hits"]:
            result = {
                "score": hit["_score"],
                "text": hit["_source"]["text"],
                "source": hit["_source"]["source_pdf"],
                "page": hit["_source"]["page_number"],
                "metadata": hit["_source"].get("metadata", {}),
                "has_image": hit["_source"]["metadata"].get("image_info", {}).get("has_image", False)
            }
            if result["has_image"]:
                result["image_info"] = hit["_source"]["metadata"]["image_info"]
            results.append(result)
            
        return results
        
    except Exception as e:
        logger.error(f"Error performing search: {str(e)}", exc_info=True)
        return []

def test_elasticsearch_connection():
    """Test Elasticsearch connection and return True if successful."""
    try:
        if not es.ping():
            logger.error("Failed to ping Elasticsearch")
            return False
            
        # Get cluster info
        cluster_info = es.info()
        logger.info(f"Connected to Elasticsearch cluster: {cluster_info.get('name', 'unknown')}")
        logger.info(f"Elasticsearch version: {cluster_info.get('version', {}).get('number', 'unknown')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    # Test the search functionality
    if test_elasticsearch_connection():
        query = "Asthma caused by humidity"
        results = search_documents(query, search_type="hybrid", k=5)
        logger.info(f"Search results for '{query}':")
        for i, result in enumerate(results, 1):
            logger.info(f"\nResult {i}:")
            logger.info(f"Score: {result['score']}")
            logger.info(f"Source: {result['source']} (Page {result['page']})")
            logger.info(f"Text: {result['text'][:200]}...") 