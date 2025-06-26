import pandas as pd
import numpy as np

# Load cleaned data
df = pd.read_csv('combined_la_liga_cleaned.csv')

# Ensure date is datetime
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
df = df.sort_values('date').reset_index(drop=True)

# Helper: get rolling stats for a team
team_stats = {}
features = []

for idx, row in df.iterrows():
    home = row['hometeam']
    away = row['awayteam']
    date = row['date']
    ftr = row['ftr']
    fthg = row['fthg']
    ftag = row['ftag']
    hs = row.get('hs', np.nan)
    as_ = row.get('as', np.nan)
    hst = row.get('hst', np.nan)
    ast = row.get('ast', np.nan)
    
    # Points for result
    def get_points(result):
        if result == 'W': return 3
        if result == 'D': return 1
        return 0
    
    # Initialize team history if not present
    for team in [home, away]:
        if team not in team_stats:
            team_stats[team] = []
    
    # Get last 5 games for home and away teams
    home_hist = [g for g in team_stats[home] if g['date'] < date]
    away_hist = [g for g in team_stats[away] if g['date'] < date]
    home_last5 = home_hist[-5:]
    away_last5 = away_hist[-5:]
    home_last3 = home_hist[-3:]
    away_last3 = away_hist[-3:]
    
    # Rolling avg goals
    home_avg_g_5 = np.mean([g['goals_for'] for g in home_last5]) if home_last5 else np.nan
    home_avg_g_3 = np.mean([g['goals_for'] for g in home_last3]) if home_last3 else np.nan
    away_avg_g_5 = np.mean([g['goals_for'] for g in away_last5]) if away_last5 else np.nan
    away_avg_g_3 = np.mean([g['goals_for'] for g in away_last3]) if away_last3 else np.nan
    
    # Rolling avg shots
    home_avg_hs_5 = np.mean([g['hs'] for g in home_last5 if not np.isnan(g['hs'])]) if home_last5 else np.nan
    home_avg_hs_3 = np.mean([g['hs'] for g in home_last3 if not np.isnan(g['hs'])]) if home_last3 else np.nan
    away_avg_as_5 = np.mean([g['as'] for g in away_last5 if not np.isnan(g['as'])]) if away_last5 else np.nan
    away_avg_as_3 = np.mean([g['as'] for g in away_last3 if not np.isnan(g['as'])]) if away_last3 else np.nan
    home_avg_hst_5 = np.mean([g['hst'] for g in home_last5 if not np.isnan(g['hst'])]) if home_last5 else np.nan
    home_avg_hst_3 = np.mean([g['hst'] for g in home_last3 if not np.isnan(g['hst'])]) if home_last3 else np.nan
    away_avg_ast_5 = np.mean([g['ast'] for g in away_last5 if not np.isnan(g['ast'])]) if away_last5 else np.nan
    away_avg_ast_3 = np.mean([g['ast'] for g in away_last3 if not np.isnan(g['ast'])]) if away_last3 else np.nan
    
    # Rolling avg points
    home_avg_pts_5 = np.mean([g['points'] for g in home_last5]) if home_last5 else np.nan
    home_avg_pts_3 = np.mean([g['points'] for g in home_last3]) if home_last3 else np.nan
    away_avg_pts_5 = np.mean([g['points'] for g in away_last5]) if away_last5 else np.nan
    away_avg_pts_3 = np.mean([g['points'] for g in away_last3]) if away_last3 else np.nan
    
    # Recent form (last 5 results)
    home_form = ''.join([g['result'] for g in home_last5]) if home_last5 else ''
    away_form = ''.join([g['result'] for g in away_last5]) if away_last5 else ''
    home_wins_5 = home_form.count('W')
    away_wins_5 = away_form.count('W')
    
    # Head-to-head (last 5 meetings)
    h2h = [g for g in team_stats[home] if g['opponent'] == away and g['date'] < date]
    h2h_last5 = h2h[-5:]
    h2h_home_wins = sum(1 for g in h2h_last5 if g['result'] == 'W')
    h2h_away_wins = sum(1 for g in h2h_last5 if g['result'] == 'L')
    h2h_draws = sum(1 for g in h2h_last5 if g['result'] == 'D')
    h2h_avg_gf = np.mean([g['goals_for'] for g in h2h_last5]) if h2h_last5 else np.nan
    h2h_avg_ga = np.mean([g['goals_against'] for g in h2h_last5]) if h2h_last5 else np.nan
    
    # Home/Away strength (win %, avg goals for/against at home/away)
    home_home_games = [g for g in team_stats[home] if g['venue'] == 'home']
    away_away_games = [g for g in team_stats[away] if g['venue'] == 'away']
    home_home_win_pct = np.mean([g['result'] == 'W' for g in home_home_games]) if home_home_games else np.nan
    away_away_win_pct = np.mean([g['result'] == 'W' for g in away_away_games]) if away_away_games else np.nan
    home_home_avg_gf = np.mean([g['goals_for'] for g in home_home_games]) if home_home_games else np.nan
    home_home_avg_ga = np.mean([g['goals_against'] for g in home_home_games]) if home_home_games else np.nan
    away_away_avg_gf = np.mean([g['goals_for'] for g in away_away_games]) if away_away_games else np.nan
    away_away_avg_ga = np.mean([g['goals_against'] for g in away_away_games]) if away_away_games else np.nan
    
    features.append({
        'home_avg_goals_5': home_avg_g_5,
        'home_avg_goals_3': home_avg_g_3,
        'away_avg_goals_5': away_avg_g_5,
        'away_avg_goals_3': away_avg_g_3,
        'home_avg_hs_5': home_avg_hs_5,
        'home_avg_hs_3': home_avg_hs_3,
        'away_avg_as_5': away_avg_as_5,
        'away_avg_as_3': away_avg_as_3,
        'home_avg_hst_5': home_avg_hst_5,
        'home_avg_hst_3': home_avg_hst_3,
        'away_avg_ast_5': away_avg_ast_5,
        'away_avg_ast_3': away_avg_ast_3,
        'home_avg_pts_5': home_avg_pts_5,
        'home_avg_pts_3': home_avg_pts_3,
        'away_avg_pts_5': away_avg_pts_5,
        'away_avg_pts_3': away_avg_pts_3,
        'home_form_last5': home_form,
        'away_form_last5': away_form,
        'home_wins_last5': home_wins_5,
        'away_wins_last5': away_wins_5,
        'h2h_home_wins_5': h2h_home_wins,
        'h2h_away_wins_5': h2h_away_wins,
        'h2h_draws_5': h2h_draws,
        'h2h_avg_gf_5': h2h_avg_gf,
        'h2h_avg_ga_5': h2h_avg_ga,
        'home_home_win_pct': home_home_win_pct,
        'away_away_win_pct': away_away_win_pct,
        'home_home_avg_gf': home_home_avg_gf,
        'home_home_avg_ga': home_home_avg_ga,
        'away_away_avg_gf': away_away_avg_gf,
        'away_away_avg_ga': away_away_avg_ga,
    })
    
    # Update team_stats for next matches
    # Home team
    home_result = 'W' if ftr == 'H' else ('D' if ftr == 'D' else 'L')
    home_points = get_points(home_result)
    team_stats[home].append({
        'date': date,
        'opponent': away,
        'venue': 'home',
        'goals_for': fthg,
        'goals_against': ftag,
        'result': home_result,
        'points': home_points,
        'hs': hs,
        'hst': hst,
        'as': as_,
        'ast': ast
    })
    # Away team
    away_result = 'W' if ftr == 'A' else ('D' if ftr == 'D' else 'L')
    away_points = get_points(away_result)
    team_stats[away].append({
        'date': date,
        'opponent': home,
        'venue': 'away',
        'goals_for': ftag,
        'goals_against': fthg,
        'result': away_result,
        'points': away_points,
        'hs': as_,
        'hst': ast,
        'as': hs,
        'ast': hst
    })

# Add features to DataFrame
features_df = pd.DataFrame(features)
df_features = pd.concat([df.reset_index(drop=True), features_df], axis=1)
df_features.to_csv('la_liga_features.csv', index=False)

print('Feature engineering complete. Saved as la_liga_features.csv.') 