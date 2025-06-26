import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

def train_model(df):
    """
    Trains an ensemble model on the provided dataframe.
    """
    le = LabelEncoder()
    df.loc[:, 'target'] = le.fit_transform(df['ftr'])

    exclude_cols = [
        'date', 'hometeam', 'awayteam', 'ftr', 'target', 'fthg', 'ftag', 'hthg', 'htag', 'htr',
        'hs', 'as', 'hst', 'ast', 'hf', 'af', 'hc', 'ac', 'hy', 'ay', 'hr', 'ar'
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [np.float64, np.int64]]
    
    X = df[feature_cols]
    feature_medians = X.median()
    X = X.fillna(feature_medians)
    y = df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    estimators = [('lr', lr), ('rf', rf)]

    if xgb_available:
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        estimators.append(('xgb', xgb))

    voting_clf = VotingClassifier(estimators=estimators, voting='soft')
    voting_clf.fit(X_scaled, y)
    
    return voting_clf, scaler, feature_cols, feature_medians, le

def construct_feature_vector(data, home_team, away_team, feature_cols):
    """
    Constructs a feature vector for a new match by using the latest stats of the two teams.
    Also returns the raw latest match data for both teams.
    """
    # Get the latest match row for each team
    home_latest = data[(data['hometeam'] == home_team) | (data['awayteam'] == home_team)].sort_values('date', ascending=False)
    if home_latest.empty: return None, None, None
    home_latest = home_latest.iloc[0]

    away_latest = data[(data['hometeam'] == away_team) | (data['awayteam'] == away_team)].sort_values('date', ascending=False)
    if away_latest.empty: return None, None, None
    away_latest = away_latest.iloc[0]

    feature_vector = {}

    # Map stats for the HOME team for the upcoming match
    home_prefix = 'home_' if home_latest['hometeam'] == home_team else 'away_'
    for col in feature_cols:
        if col.startswith('home_'):
            base_feature_name = col.replace('home_', '')
            source_col_name = home_prefix + base_feature_name
            if source_col_name in home_latest:
                feature_vector[col] = home_latest[source_col_name]

    # Map stats for the AWAY team for the upcoming match
    away_prefix = 'home_' if away_latest['hometeam'] == away_team else 'away_'
    for col in feature_cols:
        if col.startswith('away_'):
            base_feature_name = col.replace('away_', '')
            source_col_name = away_prefix + base_feature_name
            if source_col_name in away_latest:
                feature_vector[col] = away_latest[source_col_name]
    
    # Handle non-prefixed columns (like betting odds)
    for col in feature_cols:
        if not col.startswith('home_') and not col.startswith('away_'):
            if col in home_latest:
                feature_vector[col] = home_latest[col]

    # Ensure the final DataFrame has all the columns the model expects
    final_vector = pd.DataFrame([feature_vector], columns=feature_cols)

    return final_vector, home_latest, away_latest

def get_head_to_head(data, home_team, away_team):
    """
    Returns a dataframe of head-to-head matches.
    """
    h2h_data = data[((data['hometeam'] == home_team) & (data['awayteam'] == away_team)) | 
                      ((data['hometeam'] == away_team) & (data['awayteam'] == home_team))].copy()
    h2h_data = h2h_data.sort_values('date', ascending=False)
    h2h_data['date'] = h2h_data['date'].dt.date
    return h2h_data[['date', 'hometeam', 'fthg', 'ftag', 'awayteam', 'ftr']]

def get_team_form(data, team):
    """
    Returns the last 5 match results for a given team.
    """
    form_data = data[(data['hometeam'] == team) | (data['awayteam'] == team)].tail(5)
    form_data['result'] = form_data.apply(lambda row: 'W' if (row['hometeam'] == team and row['ftr'] == 'H') or (row['awayteam'] == team and row['ftr'] == 'A') else ('D' if row['ftr'] == 'D' else 'L'), axis=1)
    form_data['date'] = form_data['date'].dt.date
    return form_data[['date', 'hometeam', 'fthg', 'ftag', 'awayteam', 'result']]

if __name__ == '__main__':
    df = pd.read_csv('la_liga_features.csv', parse_dates=['date'])
    df = df.sort_values('date')
    
    # Train the model and get all artifacts
    print("Training model...")
    model, scaler, feature_cols, feature_medians, le = train_model(df.copy())
    
    # Save all artifacts to a single file
    model_artifacts = {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'feature_medians': feature_medians,
        'le': le
    }
    joblib.dump(model_artifacts, 'trained_model.joblib')
    print("Model artifacts saved to 'trained_model.joblib'")

    # Optional: For script-based evaluation if needed
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score

    X = df[feature_cols].fillna(df[feature_cols].median())
    y = le.transform(df['ftr'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    print(f"Ensemble Model Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Ensemble Model F1-Score: {f1_score(y_test, y_pred, average='weighted'):.3f}")

    # Feature Importance Plot
    rf_model = model.named_estimators_['rf']
    feature_importance = pd.DataFrame({'feature': feature_cols, 'importance': rf_model.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False).head(20)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Top 20 Feature Importances (Random Forest)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("\nFeature importance plot saved as 'feature_importance.png'") 