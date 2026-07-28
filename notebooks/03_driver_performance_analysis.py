"""
================================================================================
NOTEBOOK 03: DRIVER PERFORMANCE ANALYSIS
================================================================================

WHAT YOU WILL LEARN:
    1. How to compare drivers fairly using "Teammate Head-to-Head"
    2. How to measure consistency using Standard Deviation
    3. How to analyze overtaking ability (Position Gain)
    4. How to adjust stats for different eras (Era-Adjusted Ranking)

WHY THIS MATTERS:
    In F1, a driver's speed is heavily masked by how good their car is.
    We need metrics that look beyond wins to see who is actually driving well.
    The ultimate test is: "How did they perform compared to their teammate in the exact same car?"

RUN THIS SCRIPT:
    python notebooks/03_driver_performance_analysis.py
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

from src.data_loader import load_all_data, build_master_df, get_modern_era

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print("  NOTEBOOK 03: DRIVER PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    data = load_all_data()
    master = build_master_df(data)
    
    # Focus on modern era (2000+) for reliability and consistency
    df = get_modern_era(master, start_year=2000)
    
    # -------------------------------------------------------------------------
    # 1. POSITION GAIN ANALYSIS (Overtaking Ability)
    # -------------------------------------------------------------------------
    print("\n1. Overtaking and Position Gain Analysis...")
    driver_stats = df.groupby(['driver_id', 'driver_name']).agg(
        races=('position', 'count'),
        avg_grid=('grid', 'mean'),
        avg_finish=('position', 'mean'),
        avg_gain=('position_delta', 'mean'),
        consistency=('position', 'std') # lower = more consistent
    ).reset_index()
    
    min_races = 50
    experienced_drivers = driver_stats[driver_stats['races'] >= min_races]
    
    print(f"\nTop 10 Drivers by Avg Positions Gained (min {min_races} races):")
    top_gained = experienced_drivers.sort_values('avg_gain', ascending=False).head(10)
    print(top_gained[['driver_name', 'races', 'avg_grid', 'avg_finish', 'avg_gain']].to_string(index=False))
    
    # Plot top position gainers
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top_gained, x='avg_gain', y='driver_name', palette='crest', ax=ax)
    ax.set_xlabel('Average Positions Gained Per Race', fontsize=12)
    ax.set_ylabel('Driver', fontsize=12)
    ax.set_title('Top Overtakers: Avg Positions Gained (2000+, min 50 races)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_top_overtakers.png'), dpi=150)
    plt.close()
    print("   Chart saved: outputs/03_top_overtakers.png")

    # -------------------------------------------------------------------------
    # 2. DRIVER CONSISTENCY INDEX
    # -------------------------------------------------------------------------
    print("\n2. Consistency Analysis...")
    top_tier = experienced_drivers[experienced_drivers['avg_finish'] <= 8]
    most_consistent = top_tier.sort_values('consistency', ascending=True).head(10)
    
    print(f"\nMost Consistent Top-Tier Drivers (Avg Finish <= 8, Sorted by Std Dev):")
    print(most_consistent[['driver_name', 'races', 'avg_finish', 'consistency']].to_string(index=False))
    
    # -------------------------------------------------------------------------
    # 3. TEAMMATE HEAD-TO-HEAD ANALYSIS
    # -------------------------------------------------------------------------
    print("\n3. Teammate Head-to-Head Analysis...")
    
    teammate_wins = []
    
    # Iterate through races to compare teammates
    grouped_races = df.groupby(['season', 'round', 'constructor_id'])
    for (season, r, constructor), group in grouped_races:
        if len(group) == 2:  # Exactly two entries for the team in this race
            drivers = group.sort_values('position').reset_index()
            winner = drivers.loc[0, 'driver_id']
            loser = drivers.loc[1, 'driver_id']
            # Only record if both finished
            if not pd.isna(drivers.loc[0, 'position']) and not pd.isna(drivers.loc[1, 'position']):
                teammate_wins.append({'season': season, 'constructor': constructor, 'winner': winner, 'loser': loser})
                
    teammate_df = pd.DataFrame(teammate_wins)
    
    # Aggregate who beat who
    if len(teammate_df) > 0:
        matches = {}
        for _, row in teammate_df.iterrows():
            w, l = row['winner'], row['loser']
            matches[(w, l)] = matches.get((w, l), 0) + 1
            
        driver_h2h = {}
        for (w, l), count in matches.items():
            driver_h2h[w] = driver_h2h.get(w, {'wins': 0, 'losses': 0})
            driver_h2h[w]['wins'] += count
            
            driver_h2h[l] = driver_h2h.get(l, {'wins': 0, 'losses': 0})
            driver_h2h[l]['losses'] += count
            
        h2h_list = []
        for driver, stats in driver_h2h.items():
            total = stats['wins'] + stats['losses']
            if total >= 30:
                win_pct = stats['wins'] / total
                h2h_list.append({
                    'driver_id': driver,
                    'teammate_wins': stats['wins'],
                    'teammate_losses': stats['losses'],
                    'total_matches': total,
                    'h2h_win_rate': win_pct
                })
                
        h2h_df = pd.DataFrame(h2h_list)
        # Add names
        names = df.groupby('driver_id')['driver_name'].first().reset_index()
        h2h_df = pd.merge(h2h_df, names, on='driver_id')
        
        print(f"\nTop 10 Teammate Dominators (min 30 H2H matches):")
        top_h2h = h2h_df.sort_values('h2h_win_rate', ascending=False).head(10)
        print(top_h2h[['driver_name', 'teammate_wins', 'teammate_losses', 'total_matches', 'h2h_win_rate']].to_string(index=False))
        
        # Save H2H chart
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=top_h2h, x='h2h_win_rate', y='driver_name', palette='flare', ax=ax)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        ax.set_xlabel('Teammate H2H Win Rate', fontsize=12)
        ax.set_ylabel('Driver', fontsize=12)
        ax.set_title('Teammate Dominators: Head-to-Head Win Rate (2000+)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '03_teammate_h2h.png'), dpi=150)
        plt.close()
        print("   Chart saved: outputs/03_teammate_h2h.png")
    
    # -------------------------------------------------------------------------
    # 4. ERA ADJUSTED RANKING
    # -------------------------------------------------------------------------
    print("\n4. Era-Adjusted Finish Positions...")
    # Calculate starters per race
    starters = df.groupby(['season', 'round']).size().reset_index().rename(columns={0: 'starters'})
    df_adjusted = pd.merge(df, starters, on=['season', 'round'])
    
    df_adjusted['norm_finish'] = (df_adjusted['position'] - 1) / (df_adjusted['starters'] - 1)
    
    adjusted_stats = df_adjusted.groupby(['driver_id', 'driver_name']).agg(
        races=('position', 'count'),
        avg_norm_finish=('norm_finish', 'mean')
    ).reset_index()
    
    print(f"\nTop 10 Drivers by Era-Adjusted Finish Score (min 50 races, lower is better):")
    top_adjusted = adjusted_stats[adjusted_stats['races'] >= 50].sort_values('avg_norm_finish').head(10)
    print(top_adjusted[['driver_name', 'races', 'avg_norm_finish']].to_string(index=False))
    
    print("\nDriver Performance Analysis complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
