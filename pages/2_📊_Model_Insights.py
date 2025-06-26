import streamlit as st
from PIL import Image

st.title("📊 Model Insights")
st.write("This page shows which features are most important to our Random Forest model when making predictions.")

# Load and display the feature importance plot
try:
    img = Image.open('feature_importance.png')
    st.image(img, caption='Top 20 Most Important Features', use_column_width=True)
    st.info("""
    **How to Read This Chart:**
    - The features at the top have the most influence on the model's predictions.
    - As you can see, betting odds (like `b365a`, `b365h`) and our engineered features (like `away_avg_pts_5`) are highly predictive.
    - This gives us confidence that the model is learning from relevant information.
    """)
except FileNotFoundError:
    st.error("The feature importance plot has not been generated yet. Please run the `modeling.py` script first.") 