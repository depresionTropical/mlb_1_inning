import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# DESCARGAR JUEGOS DEL DÍA DESDE LA API
start_date = datetime(2025, 5, 24)
end_date = datetime(2025, 5, 24)
delta = timedelta(days=1)

current_date=start_date

while current_date <= end_date:
  print(current_date)
  date_str = current_date.strftime('%Y-%m-%d')
  url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}&hydrate=probablePitcher,officials,venue"
  response = requests.get(url)
  data = response.json()
  try:
    data = response.json()
    if not data.get('dates'):
      print(f"No hay juegos el {date_str}")
      current_date += delta
      continue

    games_api = []
    for date in data.get('dates', []):
      for game in date.get('games', []):
        game_id = game.get('gamePk', '')
        home_team = game['teams']['home']['team']['name']
        away_team = game['teams']['away']['team']['name']
        home_pitcher = game['teams']['home'].get('probablePitcher', {}).get('fullName', '')
        away_pitcher = game['teams']['away'].get('probablePitcher', {}).get('fullName', '')
        stadium = game.get('venue', {}).get('name', '')
        day_or_night_raw = game.get('dayNight', '')
        day_or_night = 'Día' if day_or_night_raw == 'day' else 'Noche' if day_or_night_raw == 'night' else ''

        umpire_name = ''
        officials = game.get('officials', [])
        for ump in officials:
          if ump.get('officialType', '').lower() in ['home plate', 'hp', 'home']:
            umpire_name = ump.get('official', {}).get('fullName', '')
            break

        games_api.append({
          'game_id': game_id,
          'date': date_str,
          'home_team': home_team,
          'away_team': away_team,
          'home_pitcher': home_pitcher,
          'away_pitcher': away_pitcher,
          'stadium': stadium,
          'day_or_night': day_or_night,
          'home_plate_umpire': umpire_name
        })

    df_api = pd.DataFrame(games_api)

    # CARGAR ARCHIVOS HISTÓRICOS
    games_df = pd.read_csv("./data/mlb_games_2024_full.csv")
    pitchers_df = pd.read_csv("./data/mlb_pitchers_2024.csv")

    games_df['date'] = pd.to_datetime(games_df['date'], errors='coerce', dayfirst=True)
    pitchers_df['date'] = pd.to_datetime(pitchers_df['date'], errors='coerce', dayfirst=True)

    # CÁLCULOS AUXILIARES

    # Umpire
    umpire_freq = games_df[['home_plate_umpire', 'inning_1_home', 'inning_1_away', 'game_id']].dropna()
    umpire_freq['target'] = (umpire_freq['inning_1_home'] + umpire_freq['inning_1_away'] > 0).astype(int)
    umpire_freq = umpire_freq.groupby('home_plate_umpire').agg(games=('game_id', 'count'), runs=('target', 'sum')).reset_index()
    umpire_freq['home_plate_umpire_inning1_freq'] = umpire_freq['runs'] / umpire_freq['games']
    umpire_freq_dict = umpire_freq.set_index('home_plate_umpire')['home_plate_umpire_inning1_freq'].to_dict()

    # Estadio
    stadium_freq = games_df[['stadium', 'inning_1_home', 'inning_1_away', 'game_id']].dropna()
    stadium_freq['target'] = (stadium_freq['inning_1_home'] + stadium_freq['inning_1_away'] > 0).astype(int)
    stadium_freq = stadium_freq.groupby('stadium').agg(games=('game_id', 'count'), runs=('target', 'sum')).reset_index()
    stadium_freq['stadium_inning1_freq'] = stadium_freq['runs'] / stadium_freq['games']
    stadium_freq_dict = stadium_freq.set_index('stadium')['stadium_inning1_freq'].to_dict()

    # Pitchers
    starters_df = pitchers_df[pitchers_df['pitcher_type'] == 'Abridor'].copy()
    merged_df = pd.merge(
      starters_df,
      games_df[['game_id', 'home_team', 'away_team', 'inning_1_home', 'inning_1_away']],
      on='game_id',
      how='left'
    )

    def allowed_run_1st(row):
      if row['side'] == 'home':
        return 1 if row['inning_1_away'] > 0 else 0
      elif row['side'] == 'away':
        return 1 if row['inning_1_home'] > 0 else 0
      return 0

    merged_df['run_1st_inning'] = merged_df.apply(allowed_run_1st, axis=1)

    # Frecuencia general pitcher
    pitcher_side_stats = merged_df.groupby(['player_name', 'side']).agg(
      total_starts=('game_id', 'count'),
      runs_in_1st=('run_1st_inning', 'sum')
    ).reset_index()
    pitcher_side_stats['pct_runs_1st'] = pitcher_side_stats['runs_in_1st'] / pitcher_side_stats['total_starts']
    home_pct_dict = pitcher_side_stats[pitcher_side_stats['side'] == 'home'].set_index('player_name')['pct_runs_1st'].to_dict()
    away_pct_dict = pitcher_side_stats[pitcher_side_stats['side'] == 'away'].set_index('player_name')['pct_runs_1st'].to_dict()

    # Frecuencia pitcher vs equipo
    merged_df['opponent_team'] = merged_df.apply(lambda r: r['away_team'] if r['side'] == 'home' else r['home_team'], axis=1)
    vs_team = merged_df.groupby(['player_name', 'opponent_team']).agg(
      vs_team_freq=('game_id', 'count'),
      vs_team_runs_1st=('run_1st_inning', 'sum')
    ).reset_index()
    vs_team['pct_vs_team_runs_1st'] = vs_team['vs_team_runs_1st'] / vs_team['vs_team_freq']
    vs_team_dict = vs_team.set_index(['player_name', 'opponent_team'])['pct_vs_team_runs_1st'].to_dict()

    # Conteo de enfrentamientos pitcher vs equipo
    vs_team_count_dict = vs_team.set_index(['player_name', 'opponent_team'])['vs_team_freq'].to_dict()


    # Últimos 3 juegos
    merged_sorted = merged_df.sort_values(['player_name', 'date'])
    pitcher_3g_freq = []
    for pitcher, group in merged_sorted.groupby('player_name'):
      group = group.sort_values('date')
      rolling = group['run_1st_inning'].rolling(window=3, min_periods=3).mean()
      temp = pd.DataFrame({
        'player_name': group['player_name'],
        'game_id': group['game_id'],
        'pitcher_last3_freq_1st': rolling
      })
      pitcher_3g_freq.append(temp)
    pitcher_3g_freq_df = pd.concat(pitcher_3g_freq)

    # Frecuencia por localía (ofensiva)
    home_scored_freq = games_df.groupby('home_team')['inning_1_home'].apply(lambda x: (x > 0).mean()).to_dict()
    away_scored_freq = games_df.groupby('away_team')['inning_1_away'].apply(lambda x: (x > 0).mean()).to_dict()
    combined_freq = pd.Series(list(home_scored_freq.values()) + list(away_scored_freq.values()))
    team_min = combined_freq.min()
    team_max = combined_freq.max()
    home_scored_scaled = {k: (v - team_min) / (team_max - team_min) for k, v in home_scored_freq.items()}
    away_scored_scaled = {k: (v - team_min) / (team_max - team_min) for k, v in away_scored_freq.items()}

    # Últimos 10 juegos de cada equipo
    records = []
    for _, row in games_df.iterrows():
      records.append({'team': row['home_team'], 'date': row['date'], 'scored': row['inning_1_home'] > 0})
      records.append({'team': row['away_team'], 'date': row['date'], 'scored': row['inning_1_away'] > 0})
    df_team_games = pd.DataFrame(records).dropna(subset=['date']).sort_values(['team', 'date'])
    df_team_games['rolling_avg'] = df_team_games.groupby('team')['scored'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    latest_team_form = df_team_games.groupby('team').tail(1).set_index('team')['rolling_avg'].to_dict()

    # LLENAR PLANTILLA
    enriched_rows = []
    for _, row in df_api.iterrows():
      game_id = row['game_id']
      home_pitcher = row['home_pitcher']
      away_pitcher = row['away_pitcher']
      home_team = row['home_team']
      away_team = row['away_team']
      stadium = row['stadium']
      umpire = row['home_plate_umpire']

      # Últimos 3
      home_last3_value = merged_sorted[merged_sorted['player_name'] == home_pitcher].sort_values('date').tail(3)['run_1st_inning'].mean() if home_pitcher in merged_sorted['player_name'].values else 0
      away_last3_value = merged_sorted[merged_sorted['player_name'] == away_pitcher].sort_values('date').tail(3)['run_1st_inning'].mean() if away_pitcher in merged_sorted['player_name'].values else 0

      # Calcular momentums
      home_team_freq = home_scored_scaled.get(home_team, 0)
      away_team_freq = away_scored_scaled.get(away_team, 0)
      home_team_last10 = latest_team_form.get(home_team, 0)
      away_team_last10 = latest_team_form.get(away_team, 0)
      home_team_momentum = home_team_last10 - home_team_freq
      away_team_momentum = away_team_last10 - away_team_freq
      home_pitcher_freq = home_pct_dict.get(home_pitcher, 0)
      away_pitcher_freq = away_pct_dict.get(away_pitcher, 0)
      home_pitcher_momentum = home_last3_value - home_pitcher_freq
      away_pitcher_momentum = away_last3_value - away_pitcher_freq

      enriched_rows.append({
      'game_id': game_id,
      'home_team': home_team,
      'away_team': away_team,
      'stadium': stadium,
      'day_or_night': row['day_or_night'],
      'home_pitcher': home_pitcher,
      'away_pitcher': away_pitcher,
      'home_pitcher_true_freq': home_pitcher_freq,
      'away_pitcher_true_freq': away_pitcher_freq,
      'home_pitcher_vs_team_freq': vs_team_dict.get((home_pitcher, away_team), -1),
      'away_pitcher_vs_team_freq': vs_team_dict.get((away_pitcher, home_team), -1),
      'home_pitcher_vs_team_freq_count': vs_team_count_dict.get((home_pitcher, away_team), 0),
      'away_pitcher_vs_team_freq_count': vs_team_count_dict.get((away_pitcher, home_team), 0),
      'home_pitcher_last3_freq_1st': home_last3_value,
      'away_pitcher_last3_freq_1st': away_last3_value,
      'home_team_inning1_last10_freq': home_team_last10,
      'away_team_inning1_last10_freq': away_team_last10,
      'home_team_inning1_scaled': home_team_freq,
      'away_team_inning1_scaled': away_team_freq,
      'home_team_momentum': home_team_momentum,
      'away_team_momentum': away_team_momentum,
      'home_pitcher_momentum': home_pitcher_momentum,
      'away_pitcher_momentum': away_pitcher_momentum,
      'home_pitcher_vs_away_team_momentum': away_team_momentum - home_pitcher_momentum,
      'away_pitcher_vs_home_team_momentum': home_team_momentum - away_pitcher_momentum,
      'home_plate_umpire_inning1_freq': umpire_freq_dict.get(umpire, 0),
      'stadium_inning1_freq': stadium_freq_dict.get(stadium, 0)
    })

    final_df = pd.DataFrame(enriched_rows).fillna(0)

    # Escalar estadio
    stadium_min = final_df["stadium_inning1_freq"].min()
    stadium_max = final_df["stadium_inning1_freq"].max()
    final_df["stadium_inning1_scaled"] = (final_df["stadium_inning1_freq"] - stadium_min) / (stadium_max - stadium_min)

    # Escalar umpire
    umpire_min = final_df["home_plate_umpire_inning1_freq"].min()
    umpire_max = final_df["home_plate_umpire_inning1_freq"].max()
    final_df["umpire_inning1_scaled"] = (final_df["home_plate_umpire_inning1_freq"] - umpire_min) / (umpire_max - umpire_min)

    final_df.to_csv(f"data/2025/{current_date.strftime('%Y_%m_%d')}_game.csv", index=False)
  except Exception as e:
    print(f"Error procesando el {date_str}: {e}")
  current_date += delta

  
# Guardar

print("✅ Archivo generado: en data/2025)")