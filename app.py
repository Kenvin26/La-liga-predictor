import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="La Liga Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Data for Freshness Check ---
@st.cache_data
def get_data_freshness():
    try:
        df = pd.read_csv("la_liga_features.csv", parse_dates=['date'])
        latest_date = df['date'].max().strftime('%B %d, %Y')
        return f"Data last updated: **{latest_date}**"
    except FileNotFoundError:
        return "Dataset not found. Please run the data preparation scripts."
    except Exception:
        return "Could not determine data freshness."

st.title("Welcome to the La Liga Match Predictor! ⚽")
st.markdown(get_data_freshness())

st.sidebar.success("Select a page from the options above to begin.")

st.markdown(
    """
    ### Your Ultimate Tool for Spanish Football Insights!

    This web app provides data-driven predictions and statistical insights for La Liga matches.

    ---

    ### **How to Use:**

    **👈 Select a page from the sidebar navigation** to get started:

    - **🔮 Predictions:**
        - Choose a home and away team to see the predicted outcome.
        - View win/draw/loss probabilities powered by our ensemble model.
        - Analyze head-to-head history and recent team form.

    - **📊 Model Insights:**
        - Explore which statistical features are most important for our prediction models.
        - Understand what drives the predictions, from betting odds to team performance metrics.

    ---

    This project leverages a complete data science pipeline, from data cleaning and feature engineering to model training and deployment.
    """
) 