from langchain_core.tools import StructuredTool
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.medical_retriever_tool.searcher import search_documents
from pydantic import BaseModel, Field

class MedicalResearchInput(BaseModel):
    query: str = Field(description="The query to search for in the medical research database")
    k: int = Field(description="The number of results to return")

class MedicalResearchTool(StructuredTool):
    name: str = "medical_research_tool"
    description: str = "Search medical research for health and weather queries."
    args_schema: type[BaseModel] = MedicalResearchInput

    def _run(self, query: str, k: int = 5, **kwargs) -> list:
        return search_documents(
            query=query,
            search_type="hybrid",
            k=k
        )