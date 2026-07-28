"""
================================================================================
NOTEBOOK 01: DATA EXPLORATION & CLEANING
================================================================================

WHAT YOU WILL LEARN:
    1. How to load and inspect raw data
    2. How to identify and handle missing values
    3. How to merge multiple datasets
    4. How to create exploratory visualizations
    5. How to understand your data before building models

WHY THIS MATTERS:
    "Garbage in, garbage out" — if your data is messy, your models will be bad.
    Data exploration is ALWAYS the first step in any data science project.

STUDY RESOURCES:
    - Pandas basics: https://pandas.pydata.org/docs/getting_started/intro_tutorials/
    - Data cleaning guide: https://www.kaggle.com/learn/data-cleaning
    - EDA guide: https://www.kaggle.com/learn/data-visualization

RUN THIS SCRIPT:
    python notebooks/01_data_exploration.py
================================================================================
"""

import sys
import os

# Add the project root to Python's path so we can import from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_all_data, build_master_df, get_driver_career_stats

# Create output directory for saving charts
os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'outputs'), exist_ok=True)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')


def main():
    print("=" * 70)
    print("  NOTEBOOK 01: DATA EXPLORATION & CLEANING")
    print("=" * 70)
    
    # =========================================================================
    # SECTION 1: LOAD ALL DATASETS
    # =========================================================================
    # Here we load all 8 CSV files using our data_loader module.
    # pd.read_csv() reads a CSV file into a DataFrame (like an Excel table).
    
    print("\nSECTION 1: Loading all datasets...")
    data = load_all_data()
    
    print("\nDataset Overview:")
    print(f"{'Dataset':<30} {'Rows':>8} {'Columns':>8}")
    print("-" * 50)
    for name, df in data.items():
        print(f"  {name:<28} {df.shape[0]:>8,} {df.shape[1]:>8}")
    
    # =========================================================================
    # SECTION 2: INSPECT EACH DATASET
    # =========================================================================
    # .dtypes    → shows the data type of each column (int, float, string, etc.)
    # .describe() → shows statistics (count, mean, min, max, etc.)
    # .isnull().sum() → counts missing values per column
    
    print("\n\nSECTION 2: Inspecting data quality...\n")
    
    for name, df in data.items():
        print(f"\n{'='*50}")
        print(f"  {name.upper()}")
        print(f"{'='*50}")
        print(f"\nColumn types:")
        print(df.dtypes.to_string())
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"\nWarning: Missing values:")
            for col, count in missing[missing > 0].items():
                pct = count / len(df) * 100
                print(f"  {col:<30} {count:>6} ({pct:.1f}%)")
        else:
            print("\nNo missing values!")
        
        # Show first 3 rows as a sample
        print(f"\nSample rows:")
        print(df.head(3).to_string())
    
    # =========================================================================
    # SECTION 3: HANDLE MISSING VALUES
    # =========================================================================
    # Not all missing values are problems! Let's understand WHY they're missing.
    
    print("\n\nSECTION 3: Understanding Missing Values...\n")
    
    results = data['results']
    
    print("Results dataset missing values analysis:")
    print(f"  time:            {results['time'].isnull().sum():>6} missing")
    print(f"    -> Expected! Only the RACE WINNER gets a time value.")
    print(f"    -> Other drivers have gap times like '+5.123' or are DNFs.")
    
    print(f"\n  fastest_lap:     {results['fastest_lap'].isnull().sum():>6} missing")
    print(f"    -> Expected! Fastest lap data only exists from ~2004 onwards.")
    
    print(f"\n  fastest_lap_rank:{results['fastest_lap_rank'].isnull().sum():>6} missing")
    print(f"    -> Same as above — only available in modern races.")
    
    quali = data['qualifying']
    print(f"\n  Q2 times:        {quali['q2'].isnull().sum():>6} missing")
    print(f"    -> Expected! Drivers eliminated in Q1 don't have Q2 times.")
    print(f"  Q3 times:        {quali['q3'].isnull().sum():>6} missing")
    print(f"    -> Expected! Only top 10 make it to Q3.")
    
    # =========================================================================
    # SECTION 4: BUILD MASTER DATAFRAME
    # =========================================================================
    # Merge all datasets into one big table for analysis
    
    print("\n\nSECTION 4: Building master DataFrame...\n")
    master = build_master_df(data)
    
    print(f"Master DataFrame shape: {master.shape[0]:,} rows × {master.shape[1]} columns")
    print(f"\nColumns in master DataFrame:")
    for i, col in enumerate(master.columns):
        print(f"  {i+1:>2}. {col}")
    
    # Quick stats
    print(f"\nQuick Stats:")
    print(f"  Seasons covered:    {master['season'].min()} to {master['season'].max()}")
    print(f"  Total races:        {master[['season','round']].drop_duplicates().shape[0]:,}")
    print(f"  Total drivers:      {master['driver_id'].nunique():,}")
    print(f"  Total constructors: {master['constructor_id'].nunique():,}")
    print(f"  Total circuits:     {master['circuit_id'].nunique():,}")
    
    # =========================================================================
    # SECTION 5: EXPLORATORY VISUALIZATIONS
    # =========================================================================
    # Let's create charts to understand patterns in the data
    
    print("\n\nSECTION 5: Creating visualizations...\n")
    
    # --- Chart 1: Top 20 Drivers by Wins ---
    print("  Creating: Top 20 drivers by wins...")
    wins = master[master['is_winner'] == 1].groupby('driver_name').size()
    top20 = wins.nlargest(20).sort_values()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(top20)))
    bars = ax.barh(top20.index, top20.values, color=colors, edgecolor='white')
    for bar, val in zip(bars, top20.values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                str(int(val)), va='center', fontweight='bold')
    ax.set_xlabel('Total Race Wins', fontsize=12)
    ax.set_title('Top 20 F1 Drivers by Race Wins (All Time)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_top20_wins.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: outputs/01_top20_wins.png")
    
    # --- Chart 2: Wins by Constructor ---
    print("  Creating: Top constructors by wins...")
    constructor_wins = master[master['is_winner'] == 1].groupby('constructor').size()
    top_constructors = constructor_wins.nlargest(15).sort_values()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(top_constructors)))
    ax.barh(top_constructors.index, top_constructors.values, color=colors, edgecolor='white')
    ax.set_xlabel('Total Race Wins', fontsize=12)
    ax.set_title('Top 15 F1 Constructors by Race Wins', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_constructor_wins.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: outputs/01_constructor_wins.png")
    
    # --- Chart 3: Grid vs Finish Position Scatter ---
    print("  Creating: Grid vs finish position scatter...")
    valid = master.dropna(subset=['grid', 'position'])
    valid = valid[(valid['grid'] > 0) & (valid['position'] > 0)]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(valid['grid'], valid['position'], alpha=0.05, s=10, c='#E10600')
    ax.plot([0, 25], [0, 25], 'k--', alpha=0.5, linewidth=2, label='No change')
    ax.set_xlabel('Grid (Starting) Position', fontsize=12)
    ax.set_ylabel('Race (Finishing) Position', fontsize=12)
    ax.set_title('Starting Position vs Finishing Position\n'
                 '(Below diagonal = gained positions)', fontsize=14, fontweight='bold')
    ax.set_xlim(0.5, 24.5)
    ax.set_ylim(0.5, 24.5)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_grid_vs_finish.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: outputs/01_grid_vs_finish.png")
    
    # --- Chart 4: DNF Causes Over Decades ---
    print("  Creating: DNF analysis by decade...")
    dnf_data = master[master['is_dnf'] == 1].copy()
    dnf_data['decade'] = (dnf_data['season'] // 10) * 10
    
    # Get top DNF reasons
    top_reasons = dnf_data['status'].value_counts().head(10).index
    dnf_filtered = dnf_data[dnf_data['status'].isin(top_reasons)]
    
    dnf_pivot = dnf_filtered.groupby(['decade', 'status']).size().unstack(fill_value=0)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    dnf_pivot.plot(kind='bar', stacked=True, ax=ax, colormap='Set3', edgecolor='white')
    ax.set_xlabel('Decade', fontsize=12)
    ax.set_ylabel('Number of DNFs', fontsize=12)
    ax.set_title('Retirement Causes by Decade', fontsize=14, fontweight='bold')
    ax.legend(title='Cause', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_dnf_by_decade.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: outputs/01_dnf_by_decade.png")
    
    # --- Chart 5: Races Per Season ---
    print("  Creating: Races per season trend...")
    races_per_season = master.groupby('season')['round'].max()
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(races_per_season.index, races_per_season.values, alpha=0.3, color='#E10600')
    ax.plot(races_per_season.index, races_per_season.values, color='#E10600', linewidth=2)
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Number of Races', fontsize=12)
    ax.set_title('F1 Calendar Growth Over the Decades', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_races_per_season.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: outputs/01_races_per_season.png")
    
    # --- Chart 6: Points Distribution (Modern Era) ---
    print("  Creating: Points distribution (modern era)...")
    modern = master[master['season'] >= 2010]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(modern['points'].dropna(), bins=30, color='#E10600', edgecolor='white', alpha=0.8)
    ax.set_xlabel('Points Scored', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Points per Race Entry (2010+)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_points_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: outputs/01_points_distribution.png")
    
    # =========================================================================
    # SECTION 6: KEY FINDINGS SUMMARY
    # =========================================================================
    print("\n\nSECTION 6: KEY FINDINGS\n")
    print("  1. Starting position strongly predicts finishing position")
    print(f"     -> Correlation: {valid['grid'].corr(valid['position']):.3f}")
    print(f"  2. DNFs have decreased over the decades (better reliability)")
    print(f"  3. The F1 calendar has grown from ~8 races/year to 24+")
    
    # Win from pole stats
    pole_wins = master[(master['grid'] == 1) & (master['is_winner'] == 1)].shape[0]
    total_races_with_pole = master[master['grid'] == 1].shape[0]
    pole_win_pct = pole_wins / total_races_with_pole * 100
    print(f"  4. Pole position converts to win {pole_win_pct:.1f}% of the time")
    
    # Average position gain
    avg_delta = master['position_delta'].mean()
    print(f"  5. Average position change during race: {avg_delta:+.2f}")
    
    print("\nData exploration complete! All charts saved to outputs/")
    print("=" * 70)
    
    return master


if __name__ == '__main__':
    master = main()
