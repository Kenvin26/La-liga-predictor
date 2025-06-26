# La Liga Match Predictor

This project is a comprehensive data science application that predicts the outcomes of Spanish La Liga football matches. It includes a full pipeline, from data gathering and cleaning to feature engineering, model training, and deployment as an interactive, multi-page web application using Streamlit.

## 🚀 Features

- **Match Predictions**: Predicts the outcome (Home Win, Draw, Away Win) for any given La Liga match-up.
- **Model Insights**: Visualizes the most important features that influence the model's predictions, providing transparency.
- **Historical Analysis**: Explores historical league-wide data, including win/loss distributions and home-field advantage trends.
- **Ensemble Modeling**: Uses a `VotingClassifier` that combines predictions from Logistic Regression, Random Forest, and XGBoost for improved accuracy.
- **Multi-page Web App**: A professional and user-friendly Streamlit application for easy interaction.

## 🛠️ Tech Stack

- **Python**: Core programming language.
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For machine learning (modeling, preprocessing, and evaluation).
- **XGBoost**: For an advanced gradient boosting model.
- **Streamlit**: For building the interactive web application.
- **Matplotlib & Seaborn**: For data visualization.

## 📂 Project Structure
```
la-liga-predictor/
│
├── app.py                      # Main Streamlit application entry point (Welcome Page)
├── modeling.py                 # Handles model training and prediction logic
├── feature_engineering.py      # Script for creating new features from raw data
├── combine_csvs.py             # Script to combine and clean raw seasonal data
├── visualize_la_liga.py        # Generates plots for historical analysis
│
├── pages/
│   ├── 1_🔮_Predictions.py       # Prediction page of the Streamlit app
│   ├── 2_📊_Model_Insights.py     # Model insights page
│   └── 3_📈_Historical_Analysis.py # Historical analysis page
│
├── data/                         # (Recommended) To store CSV files
│   ├── combined_la_liga_cleaned.csv
│   └── la_liga_features.csv
│
├── images/                       # (Recommended) To store generated plots
│   ├── feature_importance.png
│   ├── home_vs_away_wins.png
│   ├── team_home_wins_over_seasons.png
│   └── win_draw_loss_distribution.png
│
└── requirements.txt            # Project dependencies
```

## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Kenvin26/La-liga-predictor.git
    cd La-liga-predictor
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ How to Run

1.  **Run the data pipeline (if not already done):**
    -   Combine and clean the raw CSVs: `python combine_csvs.py`
    -   Generate features: `python feature_engineering.py`
    -   Generate visualizations: `python visualize_la_liga.py`

2.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    The application will open in your web browser.

## 📈 Model Performance

The final ensemble model achieves an accuracy of ~53-55% on the test set. While not perfect, this is a realistic result for the complex and often unpredictable nature of football matches. The most influential features were found to be betting odds and rolling point averages. 