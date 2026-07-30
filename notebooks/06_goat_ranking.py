"""
================================================================================
NOTEBOOK 06: OBJECTIVE GOAT INDEX & PCA RANKING
================================================================================

WHY ERA ADJUSTMENT MATTERS:
    In F1's early decades, cars were highly unreliable, calendars had only 7-10 races,
    and drivers points systems were totally different (e.g. 8 points for a win vs 25 today).
    Simply adding up total wins bias rankings towards modern drivers who enter 23 races a year.
    We normalize win rates, adjust scoring systems, and look at longevity to find the true GOAT.

RUN THIS SCRIPT:
    python notebooks/06_goat_ranking.py
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_all_data, build_master_df, get_driver_career_stats

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def calculate_championships(master_df):
    """
    Calculate the total driver championships won by each driver in F1 history.
    """
    df_grouped = master_df.groupby(['season', 'driver_id']).agg(
        points=('points', 'sum')
    ).reset_index()
    
    df_grouped['rank'] = df_grouped.groupby('season')['points'].rank(ascending=False, method='first')
    champs = df_grouped[df_grouped['rank'] == 1].groupby('driver_id').size().reset_index().rename(columns={0: 'championships'})
    return champs


def main():
    print("=" * 70)
    print("  NOTEBOOK 06: OBJECTIVE GOAT INDEX & PCA")
    print("=" * 70)
    
    # Load master data
    print("\nLoading data...")
    data = load_all_data()
    master = build_master_df(data)
    
    # Calculate championships
    champs = calculate_championships(master)
    
    # Get general driver career statistics
    stats = get_driver_career_stats(master)
    
    # Merge championships
    stats = pd.merge(stats, champs, on='driver_id', how='left')
    stats['championships'] = stats['championships'].fillna(0).astype(int)
    
    min_races = 30
    stats = stats[stats['total_races'] >= min_races].copy()
    
    # -------------------------------------------------------------------------
    # 1. DESIGN THE GOAT SCORE
    # -------------------------------------------------------------------------
    print("\n1. Computing GOAT Index...")
    
    def min_max_normalize(col):
        if col.max() - col.min() == 0:
            return col * 0.0
        return (col - col.min()) / (col.max() - col.min())
        
    stats['norm_win_rate'] = min_max_normalize(stats['win_rate'])
    stats['norm_champs'] = min_max_normalize(stats['championships'])
    stats['norm_podiums'] = min_max_normalize(stats['podium_rate'])
    stats['norm_finish'] = min_max_normalize(1.0 / (stats['avg_position'] + 1.0))
    stats['norm_races'] = min_max_normalize(stats['total_races'])
    stats['norm_gain'] = min_max_normalize(stats['avg_position_delta'])
    stats['norm_reliability'] = min_max_normalize(1.0 - stats['dnf_rate'])
    
    stats['GOAT_Score'] = (
        0.25 * stats['norm_win_rate'] +
        0.20 * stats['norm_champs'] +
        0.15 * stats['norm_podiums'] +
        0.15 * stats['norm_finish'] +
        0.10 * stats['norm_races'] +
        0.10 * stats['norm_gain'] +
        0.05 * stats['norm_reliability']
    )
    
    stats['GOAT_Score'] = stats['GOAT_Score'] * 100
    
    goat_ranking = stats.sort_values('GOAT_Score', ascending=False)
    
    print(f"\nTop 15 Drivers by Cumulative GOAT Index (min {min_races} races):")
    print(goat_ranking[['driver_name', 'total_races', 'total_wins', 'championships', 'win_rate', 'GOAT_Score']].head(15).to_string(index=False, formatters={'win_rate': '{:.1%}'.format, 'GOAT_Score': '{:.2f}'.format}))
    
    # Save the GOAT plot
    fig, ax = plt.subplots(figsize=(12, 7))
    top15 = goat_ranking.head(15)
    sns.barplot(data=top15, x='GOAT_Score', y='driver_name', palette='autumn', ax=ax)
    ax.set_xlabel('GOAT Score (0-100 Scale)', fontsize=12)
    ax.set_ylabel('Driver', fontsize=12)
    ax.set_title('All-Time F1 GOAT Index Ranking', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '06_goat_ranking.png'), dpi=150)
    plt.close()
    print("   Chart saved: outputs/06_goat_ranking.png")
    
    # -------------------------------------------------------------------------
    # 2. PCA RANKING (Principal Component Analysis)
    # -------------------------------------------------------------------------
    print("\n2. Performing PCA on Driver Statistics...")
    
    pca_features = [
        'win_rate', 'podium_rate', 'points_per_race', 
        'avg_position', 'avg_grid', 'avg_position_delta', 'dnf_rate'
    ]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(stats[pca_features])
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    stats['PC1'] = X_pca[:, 0]
    stats['PC2'] = X_pca[:, 1]
    
    loadings = pd.DataFrame(
        pca.components_[0], 
        index=pca_features, 
        columns=['PC1_Weight']
    ).sort_values('PC1_Weight', ascending=False)
    
    print("\nPCA Component 1 Loadings (What PC1 represents):")
    print(loadings)
    
    win_rate_loading = loadings.loc['win_rate', 'PC1_Weight']
    if win_rate_loading < 0:
        stats['PC1'] = -stats['PC1']
        
    pca_ranking = stats.sort_values('PC1', ascending=False)
    
    print("\nTop 10 Drivers by PCA Component 1 Score:")
    print(pca_ranking[['driver_name', 'total_wins', 'win_rate', 'PC1']].head(10).to_string(index=False, formatters={'win_rate': '{:.1%}'.format, 'PC1': '{:.3f}'.format}))
    
    print("\nGOAT Index ranking complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
