from unittest.mock import patch
from src.medical_retriever_tool.searcher import search_documents

with patch('src.medical_retriever_tool.searcher.search_documents') as mock_search:
    mock_search.return_value = [{"score": 0.9, "text": "Sample text", "source": "Sample source"}]
    result = search_documents(query="test", search_type="hybrid", k=5)
    print(f"Mock result: {result}")
    mock_search.assert_called_once_with(query="test", search_type="hybrid", k=5)