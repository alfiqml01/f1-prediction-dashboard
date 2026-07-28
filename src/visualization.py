"""
visualization.py — Reusable Plotting Functions
================================================

PURPOSE:
    Common charting functions used across all notebooks.
    This avoids copy-pasting the same plotting code everywhere.

LIBRARIES USED:
    - matplotlib: The classic Python plotting library (static images)
    - seaborn: Built on matplotlib, makes statistical plots prettier
    - plotly: Interactive charts you can hover over and zoom into

STUDY RESOURCES:
    - Matplotlib tutorial: https://matplotlib.org/stable/tutorials/index.html
    - Seaborn tutorial: https://seaborn.pydata.org/tutorial.html
    - Plotly tutorial: https://plotly.com/python/
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set a consistent style for all plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# F1-themed colors
F1_COLORS = {
    'red': '#E10600',
    'dark': '#15151E',
    'white': '#FFFFFF',
    'silver': '#C0C0C0',
    'gold': '#FFD700',
    'blue': '#0090D0',
    'green': '#00D2BE',
}

# Team colors for visualization
TEAM_COLORS = {
    'red_bull': '#3671C6',
    'mercedes': '#27F4D2',
    'ferrari': '#E8002D',
    'mclaren': '#FF8000',
    'aston_martin': '#229971',
    'alpine': '#FF87BC',
    'williams': '#64C4FF',
    'haas': '#B6BABD',
    'sauber': '#52E252',
    'rb': '#6692FF',
}


def plot_model_comparison(results_df, metric='ROC-AUC', save_path=None):
    """
    Bar chart comparing all models on a given metric.
    
    WHAT THIS SHOWS:
        A horizontal bar chart where each bar is a model.
        Longer bars = better performance. Easy to see which model wins!
    
    Args:
        results_df: DataFrame from train_and_evaluate()
        metric: Which metric to compare (default 'ROC-AUC')
        save_path: Optional path to save the chart as an image
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sorted_df = results_df.sort_values(metric, ascending=True)
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_df)))
    
    bars = ax.barh(sorted_df['Model'], sorted_df[metric], color=colors, edgecolor='white')
    
    # Add value labels on each bar
    for bar, val in zip(bars, sorted_df[metric]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontweight='bold', fontsize=10)
    
    ax.set_xlabel(metric, fontsize=12)
    ax.set_title(f'Model Comparison — {metric}', fontsize=14, fontweight='bold')
    ax.set_xlim(0, sorted_df[metric].max() * 1.15)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved to {save_path}")
    plt.show()


def plot_feature_importance(importance_df, top_n=15, save_path=None):
    """
    Bar chart showing which features the model found most useful.
    
    WHAT THIS SHOWS:
        Which pieces of information helped the model predict winners.
        For example, "grid position" might be 35% of the decision.
    
    Args:
        importance_df: DataFrame from get_feature_importance()
        top_n: How many features to show (default 15)
        save_path: Optional save path
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top = importance_df.head(top_n).sort_values('importance_pct')
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top)))
    
    ax.barh(top['feature'], top['importance_pct'], color=colors, edgecolor='white')
    
    for i, (val, name) in enumerate(zip(top['importance_pct'], top['feature'])):
        ax.text(val + 0.3, i, f'{val:.1f}%', va='center', fontsize=9)
    
    ax.set_xlabel('Importance (%)', fontsize=12)
    ax.set_title('Feature Importance — What Matters Most?', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None):
    """
    Plot a confusion matrix heatmap.
    
    WHAT IS A CONFUSION MATRIX?
        A table showing where the model got confused:
        
                          Predicted: NOT Winner | Predicted: Winner
        Actual: NOT Winner |      TN (correct)  |    FP (false alarm)
        Actual: Winner     |      FN (missed)   |    TP (correct!)
        
        - TN (True Negative): Correctly said "not a winner"
        - FP (False Positive): Said "winner" but was wrong (false alarm)
        - FN (False Negative): Missed an actual winner
        - TP (True Positive): Correctly identified a winner
    
    IDEAL: High numbers on the diagonal (TN and TP), low off-diagonal (FP and FN).
    """
    from sklearn.metrics import confusion_matrix as cm_func
    
    matrix = cm_func(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels or ['Not Winner', 'Winner'],
                yticklabels=labels or ['Not Winner', 'Winner'])
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_wins_by_driver(master_df, top_n=20, save_path=None):
    """
    Horizontal bar chart of all-time wins per driver.
    
    WHAT THIS SHOWS:
        The top 20 drivers by total race wins in F1 history.
        Great for understanding who the most successful drivers are.
    """
    wins = master_df[master_df['is_winner'] == 1].groupby('driver_name').size()
    top_winners = wins.nlargest(top_n).sort_values()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(top_winners)))
    
    bars = ax.barh(top_winners.index, top_winners.values, color=colors, edgecolor='white')
    
    for bar, val in zip(bars, top_winners.values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                str(int(val)), va='center', fontweight='bold')
    
    ax.set_xlabel('Total Wins', fontsize=12)
    ax.set_title(f'Top {top_n} F1 Drivers by Race Wins', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_grid_vs_finish(master_df, season=None, save_path=None):
    """
    Scatter plot: Starting grid position vs. finishing position.
    
    WHAT THIS SHOWS:
        Each dot is a race result. X-axis = where they started, Y-axis = where they finished.
        Points on the diagonal = finished where they started (no change).
        Points BELOW the diagonal = gained positions (good!).
        Points ABOVE the diagonal = lost positions (bad!).
    
    This helps us understand how much starting position matters in F1.
    """
    df = master_df.dropna(subset=['grid', 'position'])
    if season:
        df = df[df['season'] == season]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.scatter(df['grid'], df['position'], alpha=0.1, s=10, c=F1_COLORS['red'])
    
    # Draw diagonal line (start position = finish position)
    ax.plot([0, 25], [0, 25], 'k--', alpha=0.5, label='No change')
    
    ax.set_xlabel('Grid (Starting) Position', fontsize=12)
    ax.set_ylabel('Race (Finishing) Position', fontsize=12)
    
    title = f'Grid vs Finish Position'
    if season:
        title += f' — {season}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 25)
    ax.legend()
    ax.invert_yaxis()  # Position 1 at top
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_radar_chart(driver_stats, driver_name, save_path=None):
    """
    Radar (spider) chart showing a driver's strengths across multiple dimensions.
    
    WHAT THIS SHOWS:
        A multi-axis chart where each axis is a different stat (wins, poles,
        consistency, etc.). The area covered shows overall ability.
        A driver good at everything will have a large, round shape.
    
    Args:
        driver_stats: Dict with stat names as keys, normalized values (0-1) as values
        driver_name: Name to display on the chart
    """
    categories = list(driver_stats.keys())
    values = list(driver_stats.values())
    
    # Close the radar chart (connect last point to first)
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    ax.fill(angles, values, color=F1_COLORS['red'], alpha=0.25)
    ax.plot(angles, values, color=F1_COLORS['red'], linewidth=2)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_title(f'{driver_name} — Performance Profile', fontsize=14, 
                 fontweight='bold', pad=20)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_season_points_progression(master_df, season, top_n=5, save_path=None):
    """
    Line chart showing cumulative points throughout a season.
    
    WHAT THIS SHOWS:
        How the championship battle unfolded race by race.
        Lines that diverge show dominance; close lines = tight battle.
    """
    season_data = master_df[master_df['season'] == season].copy()
    
    # Calculate cumulative points per driver per round
    cumulative = season_data.groupby(['driver_name', 'round'])['points'].sum()
    cumulative = cumulative.reset_index()
    cumulative['cum_points'] = cumulative.groupby('driver_name')['points'].cumsum()
    
    # Get top N drivers by total points
    total = cumulative.groupby('driver_name')['cum_points'].max()
    top_drivers = total.nlargest(top_n).index
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for driver in top_drivers:
        driver_data = cumulative[cumulative['driver_name'] == driver]
        ax.plot(driver_data['round'], driver_data['cum_points'], 
                marker='o', linewidth=2, markersize=4, label=driver)
    
    ax.set_xlabel('Race Round', fontsize=12)
    ax.set_ylabel('Cumulative Points', fontsize=12)
    ax.set_title(f'{season} Championship Points Progression', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_clustering_scatter(X_pca, labels, driver_names=None, save_path=None):
    """
    Scatter plot of driver clusters from PCA-reduced data.
    
    WHAT THIS SHOWS:
        Each dot is a driver, plotted in 2D space after PCA compression.
        Colors indicate which cluster (archetype) each driver belongs to.
        Drivers near each other have similar racing profiles.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='Set1',
                         s=80, alpha=0.7, edgecolors='white', linewidths=0.5)
    
    if driver_names is not None:
        # Label some notable drivers
        for i, name in enumerate(driver_names):
            if i < 30:  # Only label first 30 to avoid clutter
                ax.annotate(name, (X_pca[i, 0], X_pca[i, 1]),
                           fontsize=7, alpha=0.7,
                           xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    ax.set_title('Driver Archetypes — Clustering Results', fontsize=14, fontweight='bold')
    
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_simulation_results(sim_results, title='Race Simulation Results', save_path=None):
    """
    Bar chart showing win probabilities from Monte Carlo simulation.
    
    WHAT THIS SHOWS:
        After running 10,000 simulated races, this shows the probability
        of each driver winning. Higher bar = more likely to win.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sorted_results = dict(sorted(sim_results.items(), key=lambda x: x[1], reverse=True))
    top_results = dict(list(sorted_results.items())[:15])
    
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(top_results)))[::-1]
    
    bars = ax.bar(range(len(top_results)), list(top_results.values()), 
                  color=colors, edgecolor='white')
    ax.set_xticks(range(len(top_results)))
    ax.set_xticklabels(list(top_results.keys()), rotation=45, ha='right')
    
    for bar, val in zip(bars, top_results.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax.set_ylabel('Win Probability', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
