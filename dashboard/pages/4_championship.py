import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import joblib
import plotly.express as px

# Add the project root to Python's path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.data_loader import load_all_data, build_master_df, get_modern_era

st.title("Drivers' Championship Predictor")
st.write("""
This tool forecasts the Drivers' Championship standings based on the current season progress.
Select a season and a round to see win probabilities!
""")

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
model_path = os.path.join(MODELS_DIR, 'championship_model.pkl')

if not os.path.exists(model_path):
    st.warning("⚠️ Championship predictor model is not trained yet. Please run the notebooks first.")
else:
    @st.cache_resource
    def load_champ_model():
        return joblib.load(model_path)
        
    clf = load_champ_model()
    
    @st.cache_data
    def get_champ_data():
        data = load_all_data()
        master = build_master_df(data)
        df = get_modern_era(master, start_year=2000)
        
        # Build cumulative standings
        df = df.sort_values(['season', 'round', 'driver_id']).copy()
        df['cum_points'] = df.groupby(['season', 'driver_id'])['points'].cumsum()
        df['standing_pos'] = df.groupby(['season', 'round'])['cum_points'].rank(ascending=False, method='first').astype(int)
        leader_points = df.groupby(['season', 'round'])['cum_points'].transform('max')
        df['gap_to_leader'] = leader_points - df['cum_points']
        
        # Add max round info
        max_rounds = df.groupby('season')['round'].max().reset_index().rename(columns={'round': 'max_round'})
        df = pd.merge(df, max_rounds, on='season')
        df['races_remaining'] = df['max_round'] - df['round']
        df['avg_points_per_race'] = df['cum_points'] / df['round']
        
        return df
        
    df_all = get_champ_data()
    
    # Let's load the standings data or show options
    st.sidebar.header("Forecast Settings")
    available_seasons = sorted(df_all['season'].unique(), reverse=True)
    season = st.sidebar.selectbox("Select Season", available_seasons, index=available_seasons.index(2025) if 2025 in available_seasons else 0)
    
    # Get max round for this season
    season_data = df_all[df_all['season'] == season]
    max_round_season = int(season_data['max_round'].iloc[0])
    
    round_num = st.sidebar.slider("Select Race Round", 1, max_round_season, min(11, max_round_season))
    
    st.subheader(f"📊 Standings at Round {round_num} of Season {season}")
    
    # Filter to this specific round and get top 10
    round_data = season_data[season_data['round'] == round_num].copy()
    round_data = round_data.sort_values('standing_pos').head(10)
    
    if round_data.empty:
        st.info("No data available for this selection.")
    else:
        st.table(round_data[['standing_pos', 'driver_name', 'cum_points', 'gap_to_leader']].rename(
            columns={'driver_name': 'Driver', 'cum_points': 'Points', 'gap_to_leader': 'Gap to Leader'}
        ))
        
        # Prediction
        features = ['standing_pos', 'cum_points', 'gap_to_leader', 'races_remaining', 'round', 'avg_points_per_race']
        X = round_data[features]
        
        probs = clf.predict_proba(X)[:, 1]
        round_data['win_prob'] = probs
        
        # Normalize probabilities among the top 10
        total_prob = round_data['win_prob'].sum()
        if total_prob > 0:
            round_data['normalized_prob'] = round_data['win_prob'] / total_prob
        else:
            round_data['normalized_prob'] = 1.0 / len(round_data)
            
        # Plot win probabilities
        fig = px.bar(
            round_data.sort_values('normalized_prob'), x='normalized_prob', y='driver_name',
            orientation='h',
            title="Championship Winner Probability",
            labels={'normalized_prob': 'Championship Win Probability', 'driver_name': 'Driver'},
            color='normalized_prob',
            color_continuous_scale='Turbo'
        )
        st.plotly_chart(fig, use_container_width=True)
