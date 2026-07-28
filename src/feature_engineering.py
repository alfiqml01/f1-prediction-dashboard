"""
feature_engineering.py — Shared Feature Pipeline
=================================================

PURPOSE:
    Create new "features" (columns) from raw data that help our models
    make better predictions. Raw data alone isn't enough — we need to
    calculate things like "how well has this driver been doing recently?"

WHAT IS FEATURE ENGINEERING?
    Imagine you're predicting who will win a race. Knowing the driver's
    starting position helps, but knowing their AVERAGE finishing position
    over the last 5 races helps even MORE. Feature engineering is the
    process of creating these helpful new pieces of information.

KEY CONCEPTS:
    - Rolling Average: Average of the last N values (like a moving window)
    - Cumulative: Running total from the beginning up to now
    - One-Hot Encoding: Turning categories into numbers (Red Bull -> 0/1)

STUDY RESOURCES:
    - Feature Engineering for ML: https://www.kaggle.com/learn/feature-engineering
    - Pandas GroupBy: https://pandas.pydata.org/docs/user_guide/groupby.html
    - Rolling Windows: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html
"""

import pandas as pd
import numpy as np


def add_rolling_features(df, windows=[3, 5, 10]):
    """
    Add rolling (moving) average features for each driver.
    """
    df = df.sort_values(['season', 'round']).copy()
    
    for w in windows:
        grouped = df.groupby('driver_id')
        
        df[f'driver_avg_pos_last{w}'] = grouped['position'].transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
        )
        
        df[f'driver_avg_points_last{w}'] = grouped['points'].transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
        )
        
        df[f'driver_avg_delta_last{w}'] = grouped['position_delta'].transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
        )
    
    return df


def add_cumulative_features(df):
    """
    Add career-to-date cumulative features.
    """
    df = df.sort_values(['season', 'round']).copy()
    
    grouped = df.groupby('driver_id')
    
    df['cumulative_wins'] = grouped['is_winner'].transform(
        lambda x: x.shift(1).cumsum()
    )
    
    df['cumulative_races'] = grouped['is_winner'].transform(
        lambda x: x.shift(1).expanding().count()
    )
    
    df['cumulative_win_rate'] = df['cumulative_wins'] / (df['cumulative_races'] + 1e-10)
    
    df['cumulative_podiums'] = grouped['is_podium'].transform(
        lambda x: x.shift(1).cumsum()
    )
    df['cumulative_podium_rate'] = df['cumulative_podiums'] / (df['cumulative_races'] + 1e-10)
    
    df['cumulative_dnfs'] = grouped['is_dnf'].transform(
        lambda x: x.shift(1).cumsum()
    )
    df['cumulative_dnf_rate'] = df['cumulative_dnfs'] / (df['cumulative_races'] + 1e-10)
    
    return df


def add_constructor_features(df):
    """
    Add team (constructor) performance features.
    """
    df = df.sort_values(['season', 'round']).copy()
    
    grouped = df.groupby('constructor_id')
    
    df['constructor_avg_points_last5'] = grouped['points'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )
    
    df['constructor_avg_pos_last5'] = grouped['position'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )
    
    df['constructor_cumulative_wins'] = grouped['is_winner'].transform(
        lambda x: x.shift(1).cumsum()
    )
    df['constructor_cumulative_races'] = grouped['is_winner'].transform(
        lambda x: x.shift(1).expanding().count()
    )
    df['constructor_win_rate'] = (
        df['constructor_cumulative_wins'] / (df['constructor_cumulative_races'] + 1e-10)
    )
    
    return df


def add_circuit_features(df):
    """
    Add circuit-specific features.
    """
    df = df.sort_values(['season', 'round']).copy()
    
    df['circuit_experience'] = df.groupby(['driver_id', 'circuit_id']).cumcount()
    
    df['driver_circuit_avg_pos'] = df.groupby(['driver_id', 'circuit_id'])['position'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    
    return df


def add_season_momentum(df):
    """
    Add "momentum" feature — is the driver on an upward or downward trend?
    """
    df = df.sort_values(['season', 'round']).copy()
    
    grouped = df.groupby('driver_id')
    
    short_avg = grouped['points'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    
    long_avg = grouped['points'].transform(
        lambda x: x.shift(1).rolling(window=10, min_periods=1).mean()
    )
    
    df['season_momentum'] = short_avg - long_avg
    
    return df


def add_championship_position(df, driver_standings, constructor_standings):
    """
    Add current championship standings as features.
    """
    driver_prev = driver_standings.copy()
    driver_prev['season'] = driver_prev['season'] + 1
    driver_prev = driver_prev.rename(columns={
        'position': 'prev_season_driver_pos',
        'points': 'prev_season_driver_points'
    })
    
    df = pd.merge(
        df,
        driver_prev[['season', 'driver_id', 'prev_season_driver_pos', 'prev_season_driver_points']],
        on=['season', 'driver_id'],
        how='left'
    )
    
    constructor_prev = constructor_standings.copy()
    constructor_prev['season'] = constructor_prev['season'] + 1
    constructor_prev = constructor_prev.rename(columns={
        'position': 'prev_season_constructor_pos',
        'points': 'prev_season_constructor_points'
    })
    
    df = pd.merge(
        df,
        constructor_prev[['season', 'constructor_id', 'prev_season_constructor_pos', 'prev_season_constructor_points']],
        on=['season', 'constructor_id'],
        how='left'
    )
    
    return df


def build_prediction_features(df, data_dict):
    """
    Master function that applies ALL feature engineering steps.
    """
    print("Adding rolling features...")
    df = add_rolling_features(df)
    
    print("Adding cumulative features...")
    df = add_cumulative_features(df)
    
    print("Adding constructor features...")
    df = add_constructor_features(df)
    
    print("Adding circuit features...")
    df = add_circuit_features(df)
    
    print("Adding season momentum...")
    df = add_season_momentum(df)
    
    print("Adding championship position...")
    df = add_championship_position(
        df, 
        data_dict['driver_standings'], 
        data_dict['constructor_standings']
    )
    
    print("Feature engineering complete!")
    print(f"   Total features: {df.shape[1]} columns")
    
    return df


# ---------------------------------------------------------------------------
# FEATURE LIST: These are the columns our models will use for prediction
# ---------------------------------------------------------------------------
PREDICTION_FEATURES = [
    'grid',
    'quali_position',
    'driver_avg_pos_last3',
    'driver_avg_pos_last5',
    'driver_avg_points_last5',
    'driver_avg_delta_last5',
    'cumulative_win_rate',
    'cumulative_podium_rate',
    'cumulative_dnf_rate',
    'constructor_avg_points_last5',
    'constructor_avg_pos_last5',
    'constructor_win_rate',
    'circuit_experience',
    'driver_circuit_avg_pos',
    'season_momentum',
    'prev_season_driver_pos',
    'prev_season_constructor_pos',
]


if __name__ == '__main__':
    from data_loader import load_all_data, build_master_df
    
    print("Loading data...")
    data = load_all_data()
    master = build_master_df(data)
    
    print("Building features...")
    featured = build_prediction_features(master, data)
    
    print(f"\nFinal shape: {featured.shape}")
    print(f"\nPrediction features ({len(PREDICTION_FEATURES)}):")
    for f in PREDICTION_FEATURES:
        non_null = featured[f].notna().sum()
        pct = non_null / len(featured) * 100
        print(f"  {f:40s} -> {pct:.1f}% non-null")
