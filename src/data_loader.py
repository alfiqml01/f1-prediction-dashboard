"""
data_loader.py — Central Data Loading & Merging Module
======================================================

PURPOSE:
    This module is the "single source of truth" for loading all F1 CSV data.
    Instead of writing pd.read_csv() in every notebook, we call functions here.

HOW IT WORKS:
    1. load_all_data()   → Loads all 8 CSV files into a dictionary
    2. build_master_df() → Merges them into one big DataFrame for analysis
    3. Helper functions  → Get specific slices of data (e.g., modern era only)
"""

import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
# __file__ gives us the path to THIS script.
# We go up one folder (from src/) to reach the project root, then into data/.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')


def load_all_data():
    """
    Load all 8 CSV files and return them as a dictionary of DataFrames.
    
    Returns:
        dict: Keys are descriptive names, values are pandas DataFrames.
        
    Example usage:
        data = load_all_data()
        results = data['results']      # 25,939 rows of race results
        drivers = data['drivers']      # 879 driver profiles
    """
    data = {
        'results': pd.read_csv(os.path.join(DATA_DIR, 'f1_results.csv')),
        'races': pd.read_csv(os.path.join(DATA_DIR, 'f1_races.csv')),
        'drivers': pd.read_csv(os.path.join(DATA_DIR, 'f1_drivers.csv')),
        'qualifying': pd.read_csv(os.path.join(DATA_DIR, 'f1_qualifying.csv')),
        'constructors': pd.read_csv(os.path.join(DATA_DIR, 'f1_constructors.csv')),
        'constructor_standings': pd.read_csv(os.path.join(DATA_DIR, 'f1_constructor_standings.csv')),
        'driver_standings': pd.read_csv(os.path.join(DATA_DIR, 'f1_driver_standings.csv')),
        'circuits': pd.read_csv(os.path.join(DATA_DIR, 'f1_circuits.csv')),
    }
    
    # --- Data Type Fixes ---
    # Convert 'date' column to actual date objects (not just text strings)
    # This lets us do things like: races[races['date'] > '2020-01-01']
    data['races']['date'] = pd.to_datetime(data['races']['date'])
    
    # Convert 'position' in results to numeric.
    # Some positions are text like "R" (retired) or "D" (disqualified).
    # pd.to_numeric with errors='coerce' turns those into NaN (missing value).
    data['results']['position'] = pd.to_numeric(
        data['results']['position'], errors='coerce'
    )
    
    # Same for grid — sometimes grid is 0 (pit lane start) or missing
    data['results']['grid'] = pd.to_numeric(
        data['results']['grid'], errors='coerce'
    )
    
    # Points should be numeric
    data['results']['points'] = pd.to_numeric(
        data['results']['points'], errors='coerce'
    )
    
    # Qualifying positions
    data['qualifying']['position'] = pd.to_numeric(
        data['qualifying']['position'], errors='coerce'
    )
    
    return data


def build_master_df(data=None):
    """
    Merge all datasets into a single master DataFrame.
    
    This is the "big table" that has everything we need for analysis:
    race results + race info + driver info + constructor info + qualifying.
    
    HOW THE MERGING WORKS (think of it like connecting puzzle pieces):
    
    Step 1: results + races     → We know WHICH race each result belongs to
            Connected by: season + round (e.g., 2024, round 5)
            
    Step 2: + drivers           → We know WHO each driver is (name, nationality)
            Connected by: driver_id (e.g., "hamilton")
            
    Step 3: + constructors      → We know WHICH TEAM each constructor is
            Connected by: constructor_id (e.g., "mercedes")
            
    Step 4: + circuits          → We know WHERE each race took place
            Connected by: circuit_id (e.g., "silverstone")
            
    Step 5: + qualifying        → We know qualifying results
            Connected by: season + round + driver_id
    
    Args:
        data: Optional dict from load_all_data(). If None, loads automatically.
        
    Returns:
        pd.DataFrame: The merged master DataFrame with all information.
    """
    if data is None:
        data = load_all_data()
    
    # Step 1: Start with results and add race information
    # 'left' merge means: keep ALL results, add race info where it matches
    master = pd.merge(
        data['results'], 
        data['races'],
        on=['season', 'round'],       # Match on these two columns
        how='left',                     # Keep all results even if no race info
        suffixes=('', '_race')          # If both have same column, add '_race' to the duplicate
    )
    
    # Step 2: Add driver biographical info (name, DOB, nationality)
    master = pd.merge(
        master,
        data['drivers'][['driver_id', 'given_name', 'family_name', 'dob', 'nationality']],
        on='driver_id',
        how='left',
        suffixes=('', '_driver')
    )
    
    # Step 3: Add constructor (team) info
    master = pd.merge(
        master,
        data['constructors'][['constructor_id', 'name', 'nationality']].rename(
            columns={'name': 'constructor_name', 'nationality': 'constructor_nationality'}
        ),
        on='constructor_id',
        how='left'
    )
    
    # Step 4: Add circuit details (location, country, coordinates)
    master = pd.merge(
        master,
        data['circuits'],
        on='circuit_id',
        how='left',
        suffixes=('', '_circuit')
    )
    
    # Step 5: Add qualifying data
    # We rename qualifying 'position' to 'quali_position' to avoid confusion
    quali = data['qualifying'].rename(columns={'position': 'quali_position'})
    master = pd.merge(
        master,
        quali[['season', 'round', 'driver_id', 'quali_position', 'q1', 'q2', 'q3']],
        on=['season', 'round', 'driver_id'],
        how='left'  # Left merge: keep results even if no qualifying data
    )
    
    # --- Create useful derived columns ---
    
    # is_winner: 1 if the driver won the race, 0 otherwise
    # This is our TARGET VARIABLE for the Race Winner Prediction model
    master['is_winner'] = (master['position'] == 1).astype(int)
    
    # is_podium: 1 if finished in top 3 (podium = 1st, 2nd, or 3rd)
    master['is_podium'] = (master['position'] <= 3).astype(int)
    
    # is_dnf: 1 if the driver Did Not Finish the race
    # If status is anything other than "Finished" or contains a lap count like "+1 Lap"
    master['is_dnf'] = (~master['status'].isin(
        ['Finished'] + [f'+{i} Lap' for i in range(1, 20)] + 
        [f'+{i} Laps' for i in range(2, 20)]
    )).astype(int)
    
    # position_delta: How many positions gained or lost during the race
    # Positive = gained positions (started 10th, finished 5th → gained 5)
    # Negative = lost positions (started 2nd, finished 8th → lost 6)
    master['position_delta'] = master['grid'] - master['position']
    
    # Sort by season, round, position for clean ordering
    master = master.sort_values(['season', 'round', 'position']).reset_index(drop=True)
    
    return master


def get_modern_era(master_df, start_year=2000):
    """
    Filter the master DataFrame to only include the "modern era" of F1.
    
    WHY: F1 has changed dramatically since 1950. For prediction models,
    modern data (2000+) is usually more relevant than 1950s races.
    
    Args:
        master_df: The merged master DataFrame
        start_year: First year to include (default 2000)
        
    Returns:
        pd.DataFrame: Filtered to modern era only
    """
    return master_df[master_df['season'] >= start_year].copy()


def get_driver_career_stats(master_df):
    """
    Calculate career-level statistics for every driver.
    
    Returns a DataFrame with one row per driver and columns like:
    - total_races, total_wins, win_rate, podium_rate, etc.
    
    This is used in GOAT Ranking (Notebook 06) and Clustering (Notebook 07).
    """
    stats = master_df.groupby('driver_id').agg(
        total_races=('position', 'count'),
        total_wins=('is_winner', 'sum'),
        total_podiums=('is_podium', 'sum'),
        total_points=('points', 'sum'),
        total_dnfs=('is_dnf', 'sum'),
        avg_position=('position', 'mean'),
        avg_grid=('grid', 'mean'),
        avg_position_delta=('position_delta', 'mean'),
        position_std=('position', 'std'),      # Standard deviation = consistency
        first_season=('season', 'min'),
        last_season=('season', 'max'),
    ).reset_index()
    
    # Calculate rates (percentages)
    stats['win_rate'] = stats['total_wins'] / stats['total_races']
    stats['podium_rate'] = stats['total_podiums'] / stats['total_races']
    stats['dnf_rate'] = stats['total_dnfs'] / stats['total_races']
    stats['points_per_race'] = stats['total_points'] / stats['total_races']
    stats['career_length'] = stats['last_season'] - stats['first_season'] + 1
    
    # Add driver name for readability
    driver_names = master_df.groupby('driver_id')['driver_name'].first().reset_index()
    stats = pd.merge(stats, driver_names, on='driver_id', how='left')
    
    return stats


def get_season_summary(master_df, season):
    """
    Get a summary of a specific season.
    
    Args:
        master_df: The master DataFrame
        season: Year (e.g., 2024)
    
    Returns:
        dict with season stats
    """
    season_data = master_df[master_df['season'] == season]
    
    return {
        'total_races': season_data['round'].nunique(),
        'total_drivers': season_data['driver_id'].nunique(),
        'total_constructors': season_data['constructor_id'].nunique(),
        'winner_counts': season_data[season_data['is_winner'] == 1]['driver_name'].value_counts().to_dict(),
    }


# ---------------------------------------------------------------------------
# QUICK TEST: Run this file directly to verify data loading works
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("Loading all F1 data...")
    data = load_all_data()
    
    print("\n📊 Dataset shapes:")
    for name, df in data.items():
        print(f"  {name:25s} → {df.shape[0]:>6,} rows × {df.shape[1]} columns")
    
    print("\nBuilding master DataFrame...")
    master = build_master_df(data)
    print(f"  Master DataFrame: {master.shape[0]:,} rows × {master.shape[1]} columns")
    
    print("\n🏆 Sample: Top 10 drivers by wins")
    stats = get_driver_career_stats(master)
    top10 = stats.nlargest(10, 'total_wins')[['driver_name', 'total_races', 'total_wins', 'win_rate']]
    print(top10.to_string(index=False))
    
    print("\n✅ Data loading successful!")
