import os
from elasticsearch import Elasticsearch

ES_URL = os.getenv("ES_URL")
ES_USER = os.getenv("ES_USER")
ES_PASSWORD = os.getenv("ES_PASSWORD")
ES_CERT_FINGERPRINT = os.getenv("ES_CERT_FINGERPRINT")

es = Elasticsearch(
            ES_URL,
            basic_auth=(ES_USER, ES_PASSWORD),
            verify_certs=True,
            ssl_assert_fingerprint=ES_CERT_FINGERPRINT,
            request_timeout=30,  # Increase timeout
            retry_on_timeout=True,
            max_retries=3
)

# Test connection
print(es.ping())
print(es.info())