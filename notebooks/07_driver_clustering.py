"""
================================================================================
NOTEBOOK 07: DRIVER CLUSTERING & ARCHETYPES
================================================================================

WHY CLUSTERING?
    Instead of ranking drivers from 1 to N, we want to see if drivers naturally fall into
    different "types" or "archetypes". For example, is there a cluster of highly consistent point scorers?
    Or a cluster of fast but unreliable drivers? Clustering finds these patterns automatically.

RUN THIS SCRIPT:
    python notebooks/07_driver_clustering.py
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

from src.data_loader import load_all_data, build_master_df, get_driver_career_stats

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print("  NOTEBOOK 07: DRIVER CLUSTERING & ARCHETYPES")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    data = load_all_data()
    master = build_master_df(data)
    
    # Career stats
    stats = get_driver_career_stats(master)
    
    stats = stats[stats['total_races'] >= 20].copy()
    
    # -------------------------------------------------------------------------
    # 1. SELECT AND SCALE FEATURES
    # -------------------------------------------------------------------------
    features = ['win_rate', 'podium_rate', 'dnf_rate', 'avg_position_delta', 'avg_position', 'points_per_race']
    
    print(f"\nScaling {len(features)} features for clustering...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(stats[features])
    
    # -------------------------------------------------------------------------
    # 2. RUN K-MEANS CLUSTERING
    # -------------------------------------------------------------------------
    print("\n2. Running K-Means clustering...")
    
    best_k = 4
    best_score = -1
    
    for k in range(2, 8):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"   K = {k} | Silhouette Score = {score:.4f}")
        if score > best_score:
            best_score = score
            
    print(f"\nUsing K = {best_k} for interpreting driver archetypes.")
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    stats['kmeans_cluster'] = kmeans.fit_predict(X_scaled)
    
    # -------------------------------------------------------------------------
    # 3. CLUSTER PROFILING & INTERPRETATION
    # -------------------------------------------------------------------------
    print("\n3. Profiling Clusters...")
    profile = stats.groupby('kmeans_cluster')[features + ['total_races']].mean().reset_index()
    print(profile)
    
    cluster_names = {}
    for cluster_id in range(best_k):
        c_data = profile[profile['kmeans_cluster'] == cluster_id]
        win_rate = c_data['win_rate'].values[0]
        dnf_rate = c_data['dnf_rate'].values[0]
        avg_pos = c_data['avg_position'].values[0]
        
        if win_rate > 0.08:
            cluster_names[cluster_id] = "Dominators and Champions"
        elif dnf_rate > 0.35:
            cluster_names[cluster_id] = "High-DNF Era / Unreliable"
        elif avg_pos < 11.5:
            cluster_names[cluster_id] = "Consistent Scorers"
        else:
            cluster_names[cluster_id] = "Backmarkers"
            
    stats['archetype'] = stats['kmeans_cluster'].map(cluster_names)
    
    print("\nDriver Archetype Distribution:")
    print(stats['archetype'].value_counts())
    
    for arch_name in stats['archetype'].unique():
        print(f"\n--- {arch_name} Samples ---")
        samples = stats[stats['archetype'] == arch_name].head(8)
        print(samples[['driver_name', 'total_races', 'total_wins', 'win_rate', 'dnf_rate', 'avg_position']].to_string(index=False, formatters={'win_rate': '{:.1%}'.format, 'dnf_rate': '{:.1%}'.format}))
        
    # -------------------------------------------------------------------------
    # 4. DIMENSIONALITY REDUCTION AND VISUALIZATION
    # -------------------------------------------------------------------------
    print("\n4. Dimensionality Reduction (PCA) for Visualization...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    stats['pca_x'] = X_pca[:, 0]
    stats['pca_y'] = X_pca[:, 1]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        data=stats, x='pca_x', y='pca_y', hue='archetype', 
        palette='Set1', s=80, alpha=0.8, edgecolor='w', ax=ax
    )
    
    famous_drivers = ['hamilton', 'michael_schumacher', 'senna', 'prost', 'vettel', 'verstappen', 
                      'alonso', 'raikkonen', 'ricciardo', 'perez', 'bottas', 'latifi', 'mazepin']
    for idx, row in stats.iterrows():
        if row['driver_id'] in famous_drivers:
            ax.text(row['pca_x']+0.05, row['pca_y']+0.05, row['driver_name'], fontsize=8, alpha=0.8, fontweight='bold')
            
    ax.set_xlabel('Principal Component 1 (Overall Capability)', fontsize=12)
    ax.set_ylabel('Principal Component 2 (Consistency and Era style)', fontsize=12)
    ax.set_title('Driver Archetype Clusters visualized via PCA', fontsize=14, fontweight='bold')
    ax.legend(title='Archetype', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '07_driver_clustering.png'), dpi=150)
    plt.close()
    print("   Chart saved: outputs/07_driver_clustering.png")
    
    stats.to_csv(os.path.join(OUTPUT_DIR, 'driver_clusters.csv'), index=False)
    print("   Clustering data saved: outputs/driver_clusters.csv")
    
    print("\nDriver Clustering complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
