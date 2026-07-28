"""
================================================================================
NOTEBOOK 09: CONSTRUCTOR SUCCESS PREDICTION
================================================================================

WHAT YOU WILL LEARN:
    1. How to aggregate data at the Team (Constructor) level
    2. How to model team-based championship outcomes (Top 3 finish)
    3. How to create constructor-level features (driver quality, wins in last 3 seasons)
    4. How to evaluate classifier models on constructor success

WHY CONSTRUCTOR PREDICTION?
    In F1, teams compete for the Constructors' Championship, which determines their
    share of the sport's prize money. Predicting constructor success is a key task
    for analysts, sponsors, and teams planning budgets.

RUN THIS SCRIPT:
    python notebooks/09_constructor_prediction.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_all_data, build_master_df, get_modern_era

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_constructor_season_dataset(df, constructor_standings_df):
    """
    Build constructor-season level dataset with features and target variable.
    """
    team_season = df.groupby(['season', 'constructor_id', 'constructor_name']).agg(
        season_points=('points', 'sum'),
        season_wins=('is_winner', 'sum'),
        total_race_entries=('position', 'count')
    ).reset_index()
    
    standings = constructor_standings_df.copy()
    standings = standings.rename(columns={'position': 'final_standing_pos'})
    
    dataset = pd.merge(
        team_season,
        standings[['season', 'constructor_id', 'final_standing_pos']],
        on=['season', 'constructor_id'],
        how='left'
    )
    
    nan_mask = dataset['final_standing_pos'].isna()
    if nan_mask.any():
        dataset.loc[nan_mask, 'final_standing_pos'] = dataset[nan_mask].groupby('season')['season_points'].rank(ascending=False, method='first')
        
    dataset['is_top3'] = (dataset['final_standing_pos'] <= 3).astype(int)
    
    dataset = dataset.sort_values(['constructor_id', 'season']).reset_index(drop=True)
    
    dataset['prev_standing_pos'] = dataset.groupby('constructor_id')['final_standing_pos'].shift(1)
    
    dataset['wins_last_3_seasons'] = dataset.groupby('constructor_id')['season_wins'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).sum()
    )
    
    dataset['prev_season_points'] = dataset.groupby('constructor_id')['season_points'].shift(1)
    
    points_2_ago = dataset.groupby('constructor_id')['season_points'].shift(2)
    dataset['points_momentum'] = dataset['prev_season_points'] - points_2_ago
    
    dataset['prev_standing_pos'] = dataset['prev_standing_pos'].fillna(10)
    dataset['wins_last_3_seasons'] = dataset['wins_last_3_seasons'].fillna(0)
    dataset['prev_season_points'] = dataset['prev_season_points'].fillna(0)
    dataset['points_momentum'] = dataset['points_momentum'].fillna(0)
    
    return dataset


def main():
    print("=" * 70)
    print("  NOTEBOOK 09: CONSTRUCTOR SUCCESS PREDICTION")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    data = load_all_data()
    master = build_master_df(data)
    
    df = get_modern_era(master, start_year=2000)
    
    # Build dataset
    print("\n1. Building constructor-season level features...")
    dataset = build_constructor_season_dataset(df, data['constructor_standings'])
    
    print(f"   Constructor-season dataset size: {len(dataset)} rows")
    
    counts = dataset['is_top3'].value_counts()
    print(f"   Class balance (is_top3):")
    print(f"     No Top 3: {counts[0]} ({counts[0]/len(dataset)*100:.1f}%)")
    print(f"     Top 3:    {counts[1]} ({counts[1]/len(dataset)*100:.1f}%)")
    
    features = ['prev_standing_pos', 'wins_last_3_seasons', 'prev_season_points', 'points_momentum']
    X = dataset[features]
    y = dataset['is_top3']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    train_mask = dataset['season'] < 2020
    test_mask = dataset['season'] >= 2020
    
    X_train, y_train = X_scaled[train_mask], y[train_mask]
    X_test, y_test = X_scaled[test_mask], y[test_mask]
    
    print(f"   Train set size: {len(X_train)} constructor-seasons")
    print(f"   Test set size:  {len(X_test)} constructor-seasons")
    
    # -------------------------------------------------------------------------
    # 2. MODELS AND COMPARISON
    # -------------------------------------------------------------------------
    print("\n2. Training Constructor Prediction Models...")
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    }
    
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, probs)
        acc = accuracy_score(y_test, preds)
        
        print(f"\n   --- {name} Results ---")
        print(f"     Accuracy: {acc:.2%}")
        print(f"     ROC-AUC:  {auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, preds))
        
        import joblib
        joblib.dump(clf, os.path.join(os.path.dirname(__file__), '..', 'models', f'{name.lower().replace(" ", "_")}_constructor.pkl'))
        
    print("\n3. Constructor Success Predictions for 2024 Season...")
    data_2024 = dataset[dataset['season'] == 2024].copy()
    
    if len(data_2024) > 0:
        X_2024 = scaler.transform(data_2024[features])
        probs = models['Random Forest'].predict_proba(X_2024)[:, 1]
        
        data_2024['prob_top3'] = probs
        data_2024 = data_2024.sort_values('prob_top3', ascending=False)
        
        print(f"\nPredicted Probability of Finishing in Constructor Top 3 in 2024:")
        print(f"{'Constructor':<25} {'Prev Pos':<10} {'Wins Last 3 yrs':<18} {'Top 3 Prob':<10}")
        print("-" * 68)
        for _, row in data_2024.head(10).iterrows():
            print(f"{row['constructor_name']:<25} {int(row['prev_standing_pos']):<10} "
                  f"{int(row['wins_last_3_seasons']):<18} {row['prob_top3']:.1%}")
    else:
        print("Warning: 2024 season constructor data not found.")
        
    print("\nConstructor Success Prediction complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
