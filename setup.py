from setuptools import setup, find_packages

setup(
    name="vayu",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langgraph",
        "streamlit",
        "python-dotenv",
        "pydantic",
    ],
    python_requires=">=3.8",
) 