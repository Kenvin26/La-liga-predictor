import streamlit as st
import pandas as pd
from PIL import Image
import os
import joblib
from modeling import construct_feature_vector, get_head_to_head, get_team_form
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="Match Predictions",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 Match Predictions")

@st.cache_data
def load_data_and_model_artifacts():
    """Loads all necessary data and pre-trained model artifacts."""
    data = pd.read_csv("la_liga_features.csv", parse_dates=['date'])
    artifacts = joblib.load("trained_model.joblib")
    
    model = artifacts['model']
    scaler = artifacts['scaler']
    feature_cols = artifacts['feature_cols']
    feature_medians = artifacts['feature_medians']
    le = artifacts['le']
    
    return data, model, scaler, feature_cols, feature_medians, le

# --- Load Data and Model ---
try:
    data, model, scaler, feature_cols, feature_medians, le = load_data_and_model_artifacts()
    teams = sorted(data["hometeam"].unique())
except FileNotFoundError:
    st.error("The `la_liga_features.csv` or `trained_model.joblib` file was not found. Please run the data preparation and model training scripts first.")
    st.stop()

# --- UI Components ---
st.sidebar.header("Select Match")
home_team = st.sidebar.selectbox("Home Team", teams, index=teams.index("Barcelona"))
away_team = st.sidebar.selectbox("Away Team", teams, index=teams.index("Real Madrid"))

if home_team == away_team:
    st.error("Please select two different teams.")
else:
    st.header(f"📊 Prediction for {home_team} vs. {away_team}")

    # --- Prediction Logic ---
    predict_data, home_stats, away_stats = construct_feature_vector(data, home_team, away_team, feature_cols)

    if predict_data is None:
        st.warning("No recent match data available for one of the selected teams.")
    else:
        # Prepare data for prediction
        predict_data = predict_data.fillna(feature_medians)

        # Make prediction
        prediction_proba = model.predict_proba(scaler.transform(predict_data[feature_cols]))[0]

        # --- Display Results ---
        st.subheader("Match Outcome Prediction")
        
        # Get class indices from the fitted model
        classes = model.classes_.tolist()
        h_idx = classes.index(le.transform(['H'])[0])
        d_idx = classes.index(le.transform(['D'])[0])
        a_idx = classes.index(le.transform(['A'])[0])

        col1, col2, col3 = st.columns(3)
        col1.metric(f"{home_team} Win", f"{prediction_proba[h_idx]:.2%}", delta_color="inverse")
        col2.metric("Draw", f"{prediction_proba[d_idx]:.2%}", delta_color="off")
        col3.metric(f"{away_team} Win", f"{prediction_proba[a_idx]:.2%}", delta_color="inverse")

        # Head-to-Head
        st.header("Head-to-Head History")
        h2h_data = get_head_to_head(data, home_team, away_team)
        
        if not h2h_data.empty:
            st.dataframe(h2h_data, use_container_width=True)
            
            # H2H Pie Chart
            h2h_wins = h2h_data['ftr'].value_counts()
            h2h_labels = {'H': f'{home_team} Wins', 'A': f'{away_team} Wins', 'D': 'Draws'}
            pie_labels = [h2h_labels.get(x, x) for x in h2h_wins.index]
            
            fig = go.Figure(data=[go.Pie(labels=pie_labels, values=h2h_wins.values, hole=.3)])
            fig.update_layout(title_text="Head-to-Head Outcomes")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No past matches found between these two teams in the dataset.")
            
        # Recent Form
        st.header("Recent Form (Last 5 Matches)")
        home_form_df = get_team_form(data, home_team)
        away_form_df = get_team_form(data, away_team)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{home_team}**")
            st.dataframe(home_form_df, hide_index=True)
        with col2:
            st.markdown(f"**{away_team}**")
            st.dataframe(away_form_df, hide_index=True)
