import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
file = 'combined_la_liga_cleaned.csv'
df = pd.read_csv(file)

# 1. Win/Draw/Loss distributions
plt.figure(figsize=(6,4))
sns.countplot(x='ftr', data=df, order=['H','D','A'])
plt.title('Win/Draw/Loss Distribution')
plt.xlabel('Result (H=Home, D=Draw, A=Away)')
plt.ylabel('Number of Matches')
plt.tight_layout()
plt.savefig('win_draw_loss_distribution.png')
plt.close()

# 2. Home vs. Away wins
home_wins = (df['ftr'] == 'H').sum()
away_wins = (df['ftr'] == 'A').sum()
draws = (df['ftr'] == 'D').sum()
plt.figure(figsize=(6,4))
plt.bar(['Home Wins', 'Away Wins', 'Draws'], [home_wins, away_wins, draws], color=['blue','red','gray'])
plt.title('Home vs. Away Wins')
plt.ylabel('Number of Matches')
plt.tight_layout()
plt.savefig('home_vs_away_wins.png')
plt.close()

# 3. Team performance over seasons
# Extract season/year from date (assuming date is in DD/MM/YY or DD/MM/YYYY format)
df['season'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce').dt.year
team_season = df.groupby(['season', 'hometeam'])['ftr'].value_counts().unstack(fill_value=0).reset_index()
plt.figure(figsize=(12,6))
for team in team_season['hometeam'].unique():
    team_data = team_season[team_season['hometeam'] == team]
    plt.plot(team_data['season'], team_data['H'], label=team)
plt.title('Home Wins per Team Over Seasons')
plt.xlabel('Season')
plt.ylabel('Number of Home Wins')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)
plt.tight_layout()
plt.savefig('team_home_wins_over_seasons.png')
plt.close()

print('Visualizations saved as PNG files.') 