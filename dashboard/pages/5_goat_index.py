import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px

st.title("Formula 1 GOAT Index")
st.write("""
Who is the Greatest of All Time? This page lets you define your own criteria and weights
to rank the top drivers in Formula 1 history.
""")

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

@st.cache_data
def load_goat_data():
    drivers = pd.read_csv(os.path.join(DATA_DIR, 'f1_drivers.csv'))
    results = pd.read_csv(os.path.join(DATA_DIR, 'f1_results.csv'))
    
    # Clean position and wins
    results['position_num'] = pd.to_numeric(results['position'], errors='coerce')
    results['is_winner'] = (results['position_num'] == 1).astype(int)
    results['is_podium'] = (results['position_num'] <= 3).astype(int)
    results['is_dnf'] = (~results['status'].isin(['Finished'] + [f'+{i} Lap' for i in range(1, 20)] + [f'+{i} Laps' for i in range(2, 20)])).astype(int)
    
    # Aggregate stats
    stats = results.groupby('driver_id').agg(
        total_races=('position', 'count'),
        total_wins=('is_winner', 'sum'),
        total_podiums=('is_podium', 'sum'),
        total_dnfs=('is_dnf', 'sum'),
        avg_position=('position_num', 'mean')
    ).reset_index()
    
    stats['win_rate'] = stats['total_wins'] / stats['total_races']
    stats['podium_rate'] = stats['total_podiums'] / stats['total_races']
    stats['dnf_rate'] = stats['total_dnfs'] / stats['total_races']
    
    stats = pd.merge(stats, drivers[['driver_id', 'given_name', 'family_name']], on='driver_id')
    stats['driver_name'] = stats['given_name'] + " " + stats['family_name']
    
    # Filter to drivers with at least 30 races
    stats = stats[stats['total_races'] >= 30].copy()
    
    # Add dummy championships for ranking display (pre-computed championships counts)
    champs_dict = {
        'hamilton': 7, 'michael_schumacher': 7, 'fangio': 5, 'prost': 4, 'vettel': 4,
        'senna': 3, 'lauda': 3, 'stewart': 3, 'jack_brabham': 3, 'piquet': 3,
        'alonso': 2, 'verstappen': 3, 'hakkinen': 2, 'emerson_fittipaldi': 2,
        'ascari': 2, 'graham_hill': 2, 'jim_clark': 2
    }
    stats['championships'] = stats['driver_id'].map(champs_dict).fillna(0).astype(int)
    
    return stats

stats = load_goat_data()

# Customize Weights in Sidebar
st.sidebar.header("⚖️ Custom Weights (Must sum to 100%)")
w_wins = st.sidebar.slider("Win Rate Weight (%)", 0, 100, 30)
w_champs = st.sidebar.slider("Championships Weight (%)", 0, 100, 30)
w_podiums = st.sidebar.slider("Podium Rate Weight (%)", 0, 100, 20)
w_finish = st.sidebar.slider("Finishing Position Weight (%)", 0, 100, 10)
w_dnfs = st.sidebar.slider("Reliability (Low DNF) Weight (%)", 0, 100, 10)

total_w = w_wins + w_champs + w_podiums + w_finish + w_dnfs

if total_w != 100:
    st.sidebar.error(f"Total weights must equal 100%. Current sum: {total_w}%")
else:
    # Normalize features between 0 and 1
    def norm(col, invert=False):
        scaled = (col - col.min()) / (col.max() - col.min() + 1e-10)
        return 1.0 - scaled if invert else scaled
        
    stats['n_wins'] = norm(stats['win_rate'])
    stats['n_champs'] = norm(stats['championships'])
    stats['n_podiums'] = norm(stats['podium_rate'])
    stats['n_finish'] = norm(stats['avg_position'], invert=True)
    stats['n_reliability'] = norm(stats['dnf_rate'], invert=True)
    
    # Calculate Custom GOAT Score
    stats['GOAT_Score'] = (
        (w_wins / 100.0) * stats['n_wins'] +
        (w_champs / 100.0) * stats['n_champs'] +
        (w_podiums / 100.0) * stats['n_podiums'] +
        (w_finish / 100.0) * stats['n_finish'] +
        (w_dnfs / 100.0) * stats['n_reliability']
    ) * 100
    
    ranked = stats.sort_values('GOAT_Score', ascending=False).reset_index(drop=True)
    ranked['Rank'] = np.arange(1, len(ranked) + 1)
    
    st.subheader("🏆 Custom GOAT Ranking Leaderboard")
    st.dataframe(
        ranked[['Rank', 'driver_name', 'total_races', 'total_wins', 'championships', 'win_rate', 'GOAT_Score']].head(15),
        column_config={
            'win_rate': st.column_config.NumberColumn("Win Rate", format="%.1%"),
            'GOAT_Score': st.column_config.NumberColumn("GOAT Score (0-100)", format="%.2f")
        },
        use_container_width=True
    )
    
    # Bar plot of top 15
    fig = px.bar(
        ranked.head(15).sort_values('GOAT_Score'), x='GOAT_Score', y='driver_name',
        orientation='h',
        title="Top 15 Drivers by Custom GOAT Index Score",
        labels={'GOAT_Score': 'GOAT Index Score', 'driver_name': 'Driver'},
        color='GOAT_Score',
        color_continuous_scale='Magma'
    )
    st.plotly_chart(fig, use_container_width=True)
