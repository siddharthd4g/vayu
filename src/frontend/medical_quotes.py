import streamlit as st

def display_medical_quotes(quotes: list):
    """
    Display medical research quotes in a formatted way.
    
    Args:
        quotes (list): List of medical research quotes with their metadata
    """
    st.subheader("Research Findings")
    
    for quote in quotes:
        # Create an expandable section for each quote
        with st.expander(f"Research from {quote['source']}"):
            # Display the quote text
            st.markdown(f"_{quote['quote']}_")
            
            # Display source information
            st.markdown(f"**Source:** {quote['source']}")
            st.markdown(f"**Page:** {quote['page']}")
            
            # Display relevance score if available
            if "relevance_score" in quote:
                st.markdown(f"**Relevance Score:** {quote['relevance_score']:.2f}")
            
            # Display image information if available
            if quote.get("has_image", False) and "image_info" in quote:
                st.markdown("---")
                st.markdown("**Related Image Information:**")
                st.markdown(f"**Caption:** {quote['image_info']['caption']}")
                st.markdown(f"**Description:** {quote['image_info']['description']}")
            
            st.markdown("---") 