import unittest
from unittest.mock import patch
from src.langgraph.tools.medical_research_tool import MedicalResearchTool, MedicalResearchInput
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

class TestMedicalResearchTool(unittest.TestCase):
    def test_medical_research_tool_call(self):
        # Arrange
        tool = MedicalResearchTool()
        input_data = {"query": "Asthma caused by humidity", "k": 5}  # Use dict instead
        expected_result = [{"score": 0.9, "text": "Sample text", "source": "Sample source"}]

        # Act
        with patch('src.langgraph.tools.medical_research_tool.search_documents', return_value=expected_result) as mock_search_documents:
            print(f"Input to invoke: {input_data}")
            result = tool.invoke(input_data)

        # Assert
        mock_search_documents.assert_called_once_with(
            query="Asthma caused by humidity",
            search_type="hybrid",
            k=5
        )
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()