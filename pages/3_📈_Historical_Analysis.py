import streamlit as st
from PIL import Image
import os

st.set_page_config(
    page_title="Historical Analysis",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Historical Analysis")
st.markdown("Dive into historical trends and statistics from over a decade of La Liga matches.")

# --- Load and Display Visualizations ---

st.header("Win/Draw/Loss Distribution")
st.markdown("""
This chart illustrates the overall distribution of match outcomes in La Liga. As is common in football, there's a clear advantage for the home team.
""")
if os.path.exists("win_draw_loss_distribution.png"):
    image = Image.open("win_draw_loss_distribution.png")
    st.image(image, caption="Overall Win/Draw/Loss Distribution", use_column_width=True)
else:
    st.warning("`win_draw_loss_distribution.png` not found. Please run the `visualize_la_liga.py` script to generate it.")

st.header("Home vs. Away Performance")
st.markdown("""
How much does home-field advantage matter? This visualization compares the number of wins teams achieve at home versus on the road, highlighting the disparity.
""")
if os.path.exists("home_vs_away_wins.png"):
    image = Image.open("home_vs_away_wins.png")
    st.image(image, caption="Comparison of Home and Away Wins for La Liga Teams", use_column_width=True)
else:
    st.warning("`home_vs_away_wins.png` not found. Please run the `visualize_la_liga.py` script to generate it.")

st.header("Team Home Wins Over Seasons")
st.markdown("""
This plot tracks the number of home victories for each team across different seasons, showing the ebb and flow of team dominance and form at their home stadiums.
""")
if os.path.exists("team_home_wins_over_seasons.png"):
    image = Image.open("team_home_wins_over_seasons.png")
    st.image(image, caption="Team Home Wins Across Seasons", use_column_width=True)
else:
    st.warning("`team_home_wins_over_seasons.png` not found. Please run the `visualize_la_liga.py` script to generate it.") 