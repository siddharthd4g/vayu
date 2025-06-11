import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime

def create_weather_graph(weather_data: dict):
    """
    Create an interactive weather visualization using Plotly.
    
    Args:
        weather_data (dict): Weather data including metrics and timestamps
    """
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add AQI trace
    fig.add_trace(
        go.Scatter(
            x=weather_data["metrics"]["european_aqi"]["times"],
            y=weather_data["metrics"]["european_aqi"]["values"],
            name="European AQI",
            line=dict(color="#1f77b4")
        ),
        secondary_y=False
    )
    
    # Add PM2.5 trace
    fig.add_trace(
        go.Scatter(
            x=weather_data["metrics"]["pm2_5"]["times"],
            y=weather_data["metrics"]["pm2_5"]["values"],
            name="PM2.5",
            line=dict(color="#ff7f0e")
        ),
        secondary_y=True
    )
    
    # Add PM10 trace
    fig.add_trace(
        go.Scatter(
            x=weather_data["metrics"]["pm10"]["times"],
            y=weather_data["metrics"]["pm10"]["values"],
            name="PM10",
            line=dict(color="#2ca02c")
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title=f"Weather Conditions in {weather_data['location']}",
        xaxis_title="Time",
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text="European AQI", secondary_y=False)
    fig.update_yaxes(title_text="PM2.5 / PM10 (μg/m³)", secondary_y=True)
    
    return fig

def display_weather_data(weather_data: dict):
    """
    Display weather data in Streamlit with summary statistics.
    
    Args:
        weather_data (dict): Weather data including metrics and timestamps
    """
    st.subheader("Weather Visualization")
    
    # Create and display the graph
    fig = create_weather_graph(weather_data)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Average AQI",
            f"{sum(weather_data['metrics']['european_aqi']['values']) / len(weather_data['metrics']['european_aqi']['values']):.1f}"
        )
    
    with col2:
        st.metric(
            "Average PM2.5",
            f"{sum(weather_data['metrics']['pm2_5']['values']) / len(weather_data['metrics']['pm2_5']['values']):.1f} μg/m³"
        )
    
    with col3:
        st.metric(
            "Average PM10",
            f"{sum(weather_data['metrics']['pm10']['values']) / len(weather_data['metrics']['pm10']['values']):.1f} μg/m³"
        ) 