import pandas as pd
from datetime import datetime, timedelta
import requests

# DEFINIR RANGO DE FECHAS
start_date = datetime(2025, 3, 18)
end_date = datetime(2025, 9, 28)
delta = timedelta(days=1)

# CARGAR DATOS HISTÓRICOS
games_df = pd.read_csv("./data/mlb_games_2024_full.csv")
pitchers_df = pd.read_csv("./data/mlb_pitchers_2024.csv")

games_df['date'] = pd.to_datetime(games_df['date'], errors='coerce')
pitchers_df['date'] = pd.to_datetime(pitchers_df['date'], errors='coerce')

# FRECUENCIA DE UMPIRÉS
umpire_freq = games_df[['home_plate_umpire', 'inning_1_home', 'inning_1_away', 'game_id']].dropna()
umpire_freq['target'] = (umpire_freq['inning_1_home'] + umpire_freq['inning_1_away'] > 0).astype(int)
umpire_freq = umpire_freq.groupby('home_plate_umpire').agg(games=('game_id', 'count'), runs=('target', 'sum')).reset_index()
umpire_freq['home_plate_umpire_inning1_freq'] = umpire_freq['runs'] / umpire_freq['games']
umpire_freq_dict = umpire_freq.set_index('home_plate_umpire')['home_plate_umpire_inning1_freq'].to_dict()

# FRECUENCIA POR ESTADIO
stadium_freq = games_df[['stadium', 'inning_1_home', 'inning_1_away', 'game_id']].dropna()
stadium_freq['target'] = (stadium_freq['inning_1_home'] + stadium_freq['inning_1_away'] > 0).astype(int)
stadium_freq = stadium_freq.groupby('stadium').agg(games=('game_id', 'count'), runs=('target', 'sum')).reset_index()
stadium_freq['stadium_inning1_freq'] = stadium_freq['runs'] / stadium_freq['games']
stadium_freq_dict = stadium_freq.set_index('stadium')['stadium_inning1_freq'].to_dict()

# ESTADÍSTICAS DE PITCHERS
starters_df = pitchers_df[pitchers_df['pitcher_type'] == 'Abridor'].copy()
merged_df = pd.merge(starters_df, games_df[['game_id', 'home_team', 'away_team', 'inning_1_home', 'inning_1_away']], on='game_id', how='left')
merged_df['run_1st_inning'] = merged_df.apply(lambda r: 1 if (r['side'] == 'home' and r['inning_1_away'] > 0) or (r['side'] == 'away' and r['inning_1_home'] > 0) else 0, axis=1)

# % veces que permiten carreras en 1ra entrada
pitcher_side_stats = merged_df.groupby(['player_name', 'side']).agg(total_starts=('game_id', 'count'), runs_in_1st=('run_1st_inning', 'sum')).reset_index()
pitcher_side_stats['pct_runs_1st'] = pitcher_side_stats['runs_in_1st'] / pitcher_side_stats['total_starts']
home_pct_dict = pitcher_side_stats[pitcher_side_stats['side'] == 'home'].set_index('player_name')['pct_runs_1st'].to_dict()
away_pct_dict = pitcher_side_stats[pitcher_side_stats['side'] == 'away'].set_index('player_name')['pct_runs_1st'].to_dict()

# VS EQUIPO
merged_df['opponent_team'] = merged_df.apply(lambda r: r['away_team'] if r['side'] == 'home' else r['home_team'], axis=1)
vs_team = merged_df.groupby(['player_name', 'opponent_team']).agg(vs_team_freq=('game_id', 'count'), vs_team_runs_1st=('run_1st_inning', 'sum')).reset_index()
vs_team['pct_vs_team_runs_1st'] = vs_team['vs_team_runs_1st'] / vs_team['vs_team_freq']
vs_team_dict = vs_team.set_index(['player_name', 'opponent_team'])['pct_vs_team_runs_1st'].to_dict()

# ÚLTIMOS 3 JUEGOS
merged_sorted = merged_df.sort_values(['player_name', 'date'])
pitcher_3g_freq = []
for pitcher, group in merged_sorted.groupby('player_name'):
    rolling = group['run_1st_inning'].rolling(window=3, min_periods=3).mean()
    temp = pd.DataFrame({'player_name': group['player_name'], 'game_id': group['game_id'], 'pitcher_last3_freq_1st': rolling})
    pitcher_3g_freq.append(temp)
pitcher_3g_freq_df = pd.concat(pitcher_3g_freq)

# FRECUENCIA OFENSIVA POR EQUIPO
home_scored_freq = games_df.groupby('home_team')['inning_1_home'].apply(lambda x: (x > 0).mean()).to_dict()
away_scored_freq = games_df.groupby('away_team')['inning_1_away'].apply(lambda x: (x > 0).mean()).to_dict()
combined_freq = pd.Series(list(home_scored_freq.values()) + list(away_scored_freq.values()))
team_min, team_max = combined_freq.min(), combined_freq.max()
home_scored_scaled = {k: (v - team_min) / (team_max - team_min) for k, v in home_scored_freq.items()}
away_scored_scaled = {k: (v - team_min) / (team_max - team_min) for k, v in away_scored_freq.items()}

# ÚLTIMOS 10 JUEGOS DEL EQUIPO
records = []
for _, row in games_df.iterrows():
    records.append({'team': row['home_team'], 'date': row['date'], 'scored': row['inning_1_home'] > 0})
    records.append({'team': row['away_team'], 'date': row['date'], 'scored': row['inning_1_away'] > 0})
df_team_games = pd.DataFrame(records).dropna(subset=['date']).sort_values(['team', 'date'])
df_team_games['rolling_avg'] = df_team_games.groupby('team')['scored'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
latest_team_form = df_team_games.groupby('team').tail(1).set_index('team')['rolling_avg'].to_dict()

# CONSULTA DE JUEGOS DIARIOS Y ENRIQUECIMIENTO
all_games = []
current_date = start_date
while current_date <= end_date:
    date_str = current_date.strftime('%Y-%m-%d')
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}&hydrate=probablePitcher,officials,venue"
    try:
        response = requests.get(url)
        data = response.json()
        for date in data.get('dates', []):
            for game in date.get('games', []):
                home_team = game['teams']['home']['team']['name']
                away_team = game['teams']['away']['team']['name']
                home_pitcher = game['teams']['home'].get('probablePitcher', {}).get('fullName', '')
                away_pitcher = game['teams']['away'].get('probablePitcher', {}).get('fullName', '')
                stadium = game.get('venue', {}).get('name', '')
                umpire = ''
                for ump in game.get('officials', []):
                    if ump.get('officialType', '').lower() in ['home plate', 'hp', 'home']:
                        umpire = ump.get('official', {}).get('fullName', '')
                        break
                game_id = game.get('gamePk')

                row = {
                    'game_id': game_id,
                    'date': date_str,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_pitcher': home_pitcher,
                    'away_pitcher': away_pitcher,
                    'stadium': stadium,
                    'umpire': umpire,
                    'umpire_freq': umpire_freq_dict.get(umpire, 0),
                    'stadium_freq': stadium_freq_dict.get(stadium, 0),
                    'home_pitcher_freq': home_pct_dict.get(home_pitcher, 0),
                    'away_pitcher_freq': away_pct_dict.get(away_pitcher, 0),
                    'home_pitcher_vs_opp': vs_team_dict.get((home_pitcher, away_team), 0),
                    'away_pitcher_vs_opp': vs_team_dict.get((away_pitcher, home_team), 0),
                    'home_pitcher_last3': pitcher_3g_freq_df[pitcher_3g_freq_df['player_name'] == home_pitcher]['pitcher_last3_freq_1st'].dropna().values[-1] if home_pitcher in pitcher_3g_freq_df['player_name'].values else 0,
                    'away_pitcher_last3': pitcher_3g_freq_df[pitcher_3g_freq_df['player_name'] == away_pitcher]['pitcher_last3_freq_1st'].dropna().values[-1] if away_pitcher in pitcher_3g_freq_df['player_name'].values else 0,
                    'home_momentum': latest_team_form.get(home_team, 0) - home_scored_scaled.get(home_team, 0),
                    'away_momentum': latest_team_form.get(away_team, 0) - away_scored_scaled.get(away_team, 0)
                }
                all_games.append(row)
    except Exception as e:
        print(f"Error fetching data for {date_str}: {e}")
    current_date += delta

# GUARDAR CSV
df_all = pd.DataFrame(all_games)
df_all.to_csv("./data/2025/mlb_games_2025_enriched.csv", index=False)
print("✅ Datos enriquecidos guardados exitosamente.")


