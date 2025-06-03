# Weather & Health Assistant

A Streamlit-based application that helps users with respiratory conditions understand how weather conditions might affect their health.

## Features

- User information collection (stored in browser)
- Respiratory conditions tracking
- Chat interface for weather and health queries
- Weather data visualization (coming soon)
- Integration with medical knowledge base (coming soon)

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run src/app.py
```

## Project Structure

```
weather-buddy/
├── src/
│   └── app.py
├── static/
├── data/
├── requirements.txt
└── README.md
```

## Development

- `src/app.py`: Main Streamlit application
- `static/`: Static assets (images, CSS, etc.)
- `data/`: Data files and configurations

## Coming Soon

- Weather API integration
- LangGraph chatbot integration
- Medical knowledge base search
- Weather data visualization 