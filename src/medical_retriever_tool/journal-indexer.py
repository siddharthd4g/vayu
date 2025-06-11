import os
import logging
import argparse
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain.text_splitter import RecursiveCharacterTextSplitter
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from elasticsearch.exceptions import ConnectionError, AuthenticationException, TransportError
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
import glob
import urllib3

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
OUTPUT_DIR = "output"

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

def setup_elasticsearch_index():
    """Create Elasticsearch index with mappings for text and vector search."""
    try:
        # Test connection first
        if not test_elasticsearch_connection():
            logger.error("Failed to connect to Elasticsearch")
            return False
            
        logger.info("Checking if index exists...")
        if es.indices.exists(index=MEDICAL_JOURNAL_INDEX_NAME):
            logger.info(f"Index {MEDICAL_JOURNAL_INDEX_NAME} already exists")
            return True

        logger.info(f"Creating index {MEDICAL_JOURNAL_INDEX_NAME}...")
        settings = {
            "number_of_shards": 3,
            "number_of_replicas": 1,
            "refresh_interval": "30s",
            "analysis": {
                "analyzer": {
                    "medical_text_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "stop", "snowball"]
                    }
                }
            }
        }
        
        mappings = {
            "mappings": {
                "properties": {
                    "text": {
                        "type": "text",
                        "analyzer": "medical_text_analyzer",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    "vector": {
                        "type": "dense_vector",
                        "dims": 768,  # Updated to match IBM Watson slate-125m-english-rtrvr-v2 model dimensions
                        "index": True,
                        "similarity": "cosine"
                    },
                    "source_pdf": {
                        "type": "keyword"
                    },
                    "page_number": {
                        "type": "integer"
                    },
                    "chunk_index": {
                        "type": "integer"
                    },
                    "content_type": {
                        "type": "keyword",
                        "fields": {
                            "text": {
                                "type": "text",
                                "analyzer": "medical_text_analyzer"
                            }
                        }
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "text"},
                            "author": {"type": "text"},
                            "date": {"type": "date"},
                            "keywords": {"type": "keyword"},
                            "headings": {"type": "keyword"},
                            "tables": {
                                "type": "nested",
                                "properties": {
                                    "table_index": {"type": "integer"},
                                    "content": {"type": "text", "analyzer": "medical_text_analyzer"},
                                    "structure": {"type": "text"},
                                    "page_number": {"type": "integer"}
                                }
                            },
                            "images": {
                                "type": "nested",
                                "properties": {
                                    "page_number": {"type": "integer"},
                                    "has_image": {"type": "boolean"},
                                    "caption": {"type": "text", "analyzer": "medical_text_analyzer"},
                                    "description": {"type": "text", "analyzer": "medical_text_analyzer"},
                                    "ocr_text": {"type": "text", "analyzer": "medical_text_analyzer"},
                                    "image_type": {"type": "keyword"},
                                    "position": {"type": "keyword"}
                                }
                            }
                        }
                    }
                }
            },
            "settings": settings
        }
        
        es.indices.create(index=MEDICAL_JOURNAL_INDEX_NAME, body=mappings)
        logger.info(f"Successfully created index {MEDICAL_JOURNAL_INDEX_NAME}")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up Elasticsearch index: {str(e)}", exc_info=True)
        return False

def process_and_index_pdfs(embeddings, pdf_dir="pdfs"):
    """Process PDFs in the specified directory and index chunks in Elasticsearch."""
    try:
        # Get all PDFs in the directory
        pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
        if not pdf_files:
            logger.error(f"No PDFs found in {pdf_dir}")
            return False

        # Create output file for debugging
        output_file = "pdf_processing_output.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("PDF Processing Output\n")
            f.write("=" * 50 + "\n\n")

            # Initialize Docling converter with correct options
            pipeline_options = PdfPipelineOptions(
                do_table_structure=True,
                do_figure_caption=True,
                do_image_ocr=True,
                do_image_annotation=True
            )
            
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            
            # Use RecursiveCharacterTextSplitter for better compatibility with IBM embeddings
            chunker = RecursiveCharacterTextSplitter(
                chunk_size=500,  # Smaller chunks for better semantic preservation
                chunk_overlap=50,  # Overlap to maintain context
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],  # Natural text boundaries
                is_separator_regex=False
            )
            
            total_chunks = 0
            successful_files = 0
            
            for pdf_file in pdf_files:
                try:
                    f.write(f"\n{'='*50}\n")
                    f.write(f"Processing {pdf_file}...\n")
                    f.write(f"{'='*50}\n\n")
                    
                    # Validate PDF file
                    if not os.path.exists(pdf_file):
                        f.write(f"PDF file not found: {pdf_file}\n")
                        continue
                        
                    if os.path.getsize(pdf_file) == 0:
                        f.write(f"PDF file is empty: {pdf_file}\n")
                        continue
                    
                    # Convert PDF using Docling
                    result = converter.convert(pdf_file)
                    if not result or not result.document:
                        f.write(f"No content extracted from {pdf_file}\n")
                        continue
                    
                    # Process tables
                    tables = []
                    if hasattr(result.document, 'tables'):
                        f.write("\nProcessing tables:\n")
                        for i, table in enumerate(result.document.tables):
                            table_info = {
                                "table_index": i,
                                "content": str(table.content) if hasattr(table, 'content') else "",
                                "structure": str(table.structure) if hasattr(table, 'structure') else "",
                                "page_number": table.prov[0].page_no if hasattr(table, 'prov') and len(table.prov) > 0 else 0
                            }
                            tables.append(table_info)
                            f.write(f"Table {i+1} on page {table_info['page_number']}:\n")
                            f.write(f"Content: {table_info['content'][:200]}...\n")
                    
                    # Process images
                    images = []
                    f.write("\nProcessing images:\n")
                    for picture in result.document.pictures:
                        if picture.prov and len(picture.prov) > 0:
                            page_no = picture.prov[0].page_no
                            image_info = {
                                "page_number": page_no,
                                "has_image": True,
                                "caption": picture.caption if hasattr(picture, 'caption') else "",
                                "description": picture.description if hasattr(picture, 'description') else "",
                                "ocr_text": picture.ocr_text if hasattr(picture, 'ocr_text') else "",
                                "image_type": "figure",
                                "position": "unknown"
                            }
                            images.append(image_info)
                            f.write(f"Image on page {page_no}:\n")
                            f.write(f"Caption: {image_info['caption']}\n")
                            f.write(f"Description: {image_info['description']}\n")
                            f.write(f"OCR Text: {image_info['ocr_text'][:200]}...\n")
                    
                    # Get and clean text content
                    f.write("\nProcessing text content:\n")
                    markdown_content = result.document.export_to_markdown()
                    
                    # Clean up the markdown content while preserving table structure
                    cleaned_lines = []
                    in_table = False
                    for line in markdown_content.split('\n'):
                        line = line.strip()
                        
                        # Check if we're entering or exiting a table
                        if line.startswith('|') or line.startswith('+-'):
                            in_table = True
                            cleaned_lines.append(line)  # Keep table structure intact
                            continue
                        elif in_table and not (line.startswith('|') or line.startswith('+-')):
                            in_table = False
                        
                        # For non-table content, clean up unnecessary hyphens
                        if not in_table:
                            if line and not line.replace('-', '').strip() == '':
                                cleaned_lines.append(line)
                        else:
                            cleaned_lines.append(line)  # Keep all table lines
                    
                    cleaned_content = '\n'.join(cleaned_lines)
                    
                    # Split the content into chunks
                    chunks = chunker.split_text(cleaned_content)
                    f.write(f"\nCreated {len(chunks)} text chunks\n")
                    
                    actions = []
                    for i, chunk in enumerate(chunks):
                        try:
                            text = chunk
                            if not text.strip():
                                continue
                                
                            # Generate embedding
                            vector = embeddings.embed_query(text)
                            
                            # Extract page number from the chunk (if available in the markdown)
                            page_number = 0  # Default to first page
                            for line in text.split('\n'):
                                if line.startswith('<!-- Page'):
                                    try:
                                        page_number = int(line.split('Page')[1].split('-->')[0].strip())
                                        break
                                    except:
                                        pass
                            
                            # Determine content type
                            content_type = "text"
                            if any(table["page_number"] == page_number for table in tables):
                                content_type = "table"
                            if any(image["page_number"] == page_number for image in images):
                                content_type = "image"
                            
                            # Prepare metadata with proper type handling
                            metadata = {
                                "title": result.document.name if hasattr(result.document, 'name') else None,
                                "author": None,  # Would need to extract from document metadata
                                "date": None,    # Would need to extract from document metadata
                                "keywords": [],  # Would need to extract from document metadata
                                "headings": [],  # Would need to extract from document structure
                                "tables": tables,
                                "images": images
                            }
                            
                            # Remove None values from metadata
                            metadata = {k: v for k, v in metadata.items() if v is not None}
                            
                            action = {
                                "_index": MEDICAL_JOURNAL_INDEX_NAME,
                                "_id": f"{os.path.basename(pdf_file)}_{i}",
                                "_source": {
                                    "text": text,
                                    "vector": vector,
                                    "source_pdf": os.path.basename(pdf_file),
                                    "page_number": page_number,
                                    "chunk_index": i,
                                    "content_type": content_type,
                                    "metadata": metadata
                                }
                            }
                            actions.append(action)
                            
                            f.write(f"\nChunk {i+1} (Type: {content_type}):\n")
                            f.write(f"Page: {page_number}\n")
                            f.write(f"Content: {text[:200]}...\n")
                            
                        except Exception as chunk_error:
                            f.write(f"Error processing chunk {i} from {pdf_file}: {str(chunk_error)}\n")
                            continue
                    
                    # Bulk index to Elasticsearch
                    if actions:
                        try:
                            # Add detailed error handling for bulk operation
                            success, failed = bulk(es, actions, raise_on_error=False, stats_only=False)
                            
                            if failed:
                                f.write(f"\nBulk indexing errors for {pdf_file}:\n")
                                for item in failed:
                                    f.write(f"Error in item {item['index']['_id']}: {item['index']['error']}\n")
                            else:
                                total_chunks += len(actions)
                                successful_files += 1
                                f.write(f"\nSuccessfully indexed {len(actions)} chunks from {pdf_file}\n")
                                
                        except Exception as bulk_error:
                            f.write(f"Error bulk indexing {pdf_file}: {str(bulk_error)}\n")
                            # Print the first few actions to debug
                            f.write("\nFirst few actions that failed:\n")
                            for i, action in enumerate(actions[:3]):
                                f.write(f"Action {i}:\n")
                                f.write(f"ID: {action['_id']}\n")
                                f.write(f"Text length: {len(action['_source']['text'])}\n")
                                f.write(f"Vector length: {len(action['_source']['vector'])}\n")
                                f.write(f"Content type: {action['_source']['content_type']}\n")
                                f.write("-" * 30 + "\n")
                    
                except Exception as file_error:
                    f.write(f"Error processing file {pdf_file}: {str(file_error)}\n")
                    continue
            
            f.write(f"\nIndexing complete. Processed {successful_files}/{len(pdf_files)} files, {total_chunks} total chunks\n")
            logger.info(f"Output written to {output_file}")
        
        return successful_files > 0
        
    except Exception as e:
        logger.error(f"Error in PDF processing: {str(e)}", exc_info=True)
        return False

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

def delete_index_if_exists():
    """Delete the index if it exists."""
    try:
        if es.indices.exists(index=MEDICAL_JOURNAL_INDEX_NAME):
            logger.info(f"Deleting existing index {MEDICAL_JOURNAL_INDEX_NAME}")
            es.indices.delete(index=MEDICAL_JOURNAL_INDEX_NAME)
            logger.info(f"Successfully deleted index {MEDICAL_JOURNAL_INDEX_NAME}")
        return True
    except Exception as e:
        logger.error(f"Error deleting index: {str(e)}", exc_info=True)
        return False

def main():
    """Main function to setup and index PDFs."""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Index medical journal PDFs into Elasticsearch')
        parser.add_argument('--append', action='store_true', 
                          help='Append to existing index instead of deleting it')
        args = parser.parse_args()

        # Initialize embeddings
        embeddings = initialize_watsonx()
        
        # Check if index exists
        index_exists = es.indices.exists(index=MEDICAL_JOURNAL_INDEX_NAME)
        
        if not index_exists:
            logger.info(f"Index {MEDICAL_JOURNAL_INDEX_NAME} does not exist. Creating new index...")
            if not setup_elasticsearch_index():
                logger.error("Failed to setup Elasticsearch index")
                return
        elif not args.append:
            # Delete existing index if not in append mode
            logger.info(f"Deleting existing index {MEDICAL_JOURNAL_INDEX_NAME}...")
            if not delete_index_if_exists():
                logger.error("Failed to delete existing index")
                return
            # Create new index
            if not setup_elasticsearch_index():
                logger.error("Failed to setup Elasticsearch index")
                return
        else:
            logger.info(f"Appending to existing index {MEDICAL_JOURNAL_INDEX_NAME}")
        
        # Process and index PDFs
        if not process_and_index_pdfs(embeddings):
            logger.error("Failed to process and index PDFs")
            return
        
        logger.info("Indexing complete")
            
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()