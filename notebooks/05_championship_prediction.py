"""
================================================================================
NOTEBOOK 05: CHAMPIONSHIP PREDICTION
================================================================================

WHY MID-SEASON FORECASTING?
    At the start of the season, anyone can win. By race 10, the picture is clearer.
    We build a machine learning model that looks at standings at any round, calculates
    things like "gap to leader" and "races remaining", and predicts the probability
    of each driver winning the Drivers' Championship.

RUN THIS SCRIPT:
    python notebooks/05_championship_prediction.py
================================================================================
"""

import sys
import os

# Add the project root to Python's path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

from src.data_loader import load_all_data, build_master_df, get_modern_era

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_cumulative_standings(df):
    """
    Build cumulative points standings race-by-race for each season.
    """
    df = df.sort_values(['season', 'round', 'driver_id']).copy()
    df['cum_points'] = df.groupby(['season', 'driver_id'])['points'].cumsum()
    df['standing_pos'] = df.groupby(['season', 'round'])['cum_points'].rank(ascending=False, method='first').astype(int)
    leader_points = df.groupby(['season', 'round'])['cum_points'].transform('max')
    df['gap_to_leader'] = leader_points - df['cum_points']
    
    return df


def main():
    print("=" * 70)
    print("  NOTEBOOK 05: CHAMPIONSHIP PREDICTION")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    data = load_all_data()
    master = build_master_df(data)
    
    # Focus on modern era (2000+)
    df = get_modern_era(master, start_year=2000)
    
    # -------------------------------------------------------------------------
    # 1. BUILD STANDINGS TIMELINE
    # -------------------------------------------------------------------------
    print("\n1. Building race-by-race standings...")
    standings = build_cumulative_standings(df)
    
    print("\nSample: Final Standings of 2021 Season:")
    s_2021 = standings[(standings['season'] == 2021) & (standings['round'] == 22)].sort_values('standing_pos')
    print(s_2021[['standing_pos', 'driver_name', 'cum_points', 'gap_to_leader']].head(5).to_string(index=False))
    
    # Save points progression plot for 2021
    fig, ax = plt.subplots(figsize=(12, 6))
    top_drivers_2021 = s_2021.head(5)['driver_name'].tolist()
    s_2021_all = standings[(standings['season'] == 2021) & (standings['driver_name'].isin(top_drivers_2021))]
    
    sns.lineplot(data=s_2021_all, x='round', y='cum_points', hue='driver_name', marker='o', ax=ax)
    ax.set_xlabel('Race Round', fontsize=12)
    ax.set_ylabel('Cumulative Points', fontsize=12)
    ax.set_title('2021 Drivers Championship Points Progression (Top 5)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '05_championship_progression_2021.png'), dpi=150)
    plt.close()
    print("   Chart saved: outputs/05_championship_progression_2021.png")
    
    # -------------------------------------------------------------------------
    # 2. FEATURE ENGINEERING FOR CHAMPIONSHIP PREDICTION
    # -------------------------------------------------------------------------
    print("\n2. Preparing championship classification features...")
    
    max_rounds = standings.groupby('season')['round'].max().reset_index().rename(columns={'round': 'max_round'})
    standings = pd.merge(standings, max_rounds, on='season')
    
    final_standings = standings[standings['round'] == standings['max_round']].copy()
    champions = final_standings[final_standings['standing_pos'] == 1][['season', 'driver_id']].copy()
    champions['is_champion'] = 1
    
    standings = pd.merge(standings, champions, on=['season', 'driver_id'], how='left')
    standings['is_champion'] = standings['is_champion'].fillna(0).astype(int)
    
    standings['races_remaining'] = standings['max_round'] - standings['round']
    standings['avg_points_per_race'] = standings['cum_points'] / standings['round']
    
    features = ['standing_pos', 'cum_points', 'gap_to_leader', 'races_remaining', 'round', 'avg_points_per_race']
    
    model_data = standings[standings['standing_pos'] <= 10].copy()
    
    train = model_data[model_data['season'] < 2018]
    test = model_data[model_data['season'] >= 2018]
    
    X_train, y_train = train[features], train['is_champion']
    X_test, y_test = test[features], test['is_champion']
    
    # -------------------------------------------------------------------------
    # 3. TRAIN CLASSIFIER
    # -------------------------------------------------------------------------
    print("\n3. Training Championship Classifier (Random Forest)...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"   Test Accuracy: {acc:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    
    joblib.dump(clf, os.path.join(os.path.dirname(__file__), '..', 'models', 'championship_model.pkl'))
    print("   Championship model saved to models/championship_model.pkl")
    
    # -------------------------------------------------------------------------
    # 4. BACKTEST PREDICTION (2021 MID-SEASON FORECAST)
    # -------------------------------------------------------------------------
    print("\n4. Backtest Forecast: 2021 Mid-Season (Round 11)...")
    mid_2021 = model_data[(model_data['season'] == 2021) & (model_data['round'] == 11)]
    
    if len(mid_2021) > 0:
        mid_2021_X = mid_2021[features]
        mid_2021_probs = clf.predict_proba(mid_2021_X)[:, 1]
        
        forecast = mid_2021.copy()
        forecast['prob'] = mid_2021_probs
        forecast['normalized_prob'] = forecast['prob'] / forecast['prob'].sum()
        
        forecast = forecast.sort_values('normalized_prob', ascending=False)
        
        print(f"\n2021 Round 11 Standing and Predicted Champion Probabilities:")
        print(f"{'Pos':<5} {'Driver':<20} {'Points':<10} {'Gap':<6} {'Championship Win Probability':<15}")
        print("-" * 70)
        for _, row in forecast.head(5).iterrows():
            print(f"{int(row['standing_pos']):<5} {row['driver_name']:<20} {row['cum_points']:<10} "
                  f"{int(row['gap_to_leader']):<6} {row['normalized_prob']:.1%}")
    else:
        print("Warning: 2021 mid-season data not found.")
        
    print("\nChampionship Prediction complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
