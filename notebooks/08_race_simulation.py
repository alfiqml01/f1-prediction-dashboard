"""
================================================================================
NOTEBOOK 08: DETAILED RACE SIMULATION ENGINE
================================================================================

WHY LAP-BY-LAP SIMULATION?
    While prediction models (like Notebook 02) give static win probabilities,
    simulation engines let us watch the race unfold, tracking live positions,
    crashes, and probability shifts. This is similar to how real F1 teams
    run thousands of simulations on the pit wall.

RUN THIS SCRIPT:
    python notebooks/08_race_simulation.py
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
from src.visualization import plot_simulation_results

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def simulate_lap_by_lap(grid_order, pace_ratings, dnf_probs, overtaking_factor, laps=50):
    """
    Simulate a single race lap-by-lap, returning position history.
    """
    drivers = list(grid_order.keys())
    current_order = sorted(drivers, key=lambda d: grid_order[d])
    
    history = [current_order.copy()]
    retired = []
    
    lap_dnf_probs = {d: 1.0 - (1.0 - dnf_probs.get(d, 0.08)) ** (1.0 / laps) for d in drivers}
    
    for lap in range(laps):
        still_active = []
        for d in current_order:
            if np.random.rand() < lap_dnf_probs[d]:
                retired.append(d)
            else:
                still_active.append(d)
        current_order = still_active
        
        i = 0
        while i < len(current_order) - 1:
            d_ahead = current_order[i]
            d_behind = current_order[i+1]
            
            pace_diff = pace_ratings[d_behind] - pace_ratings[d_ahead]
            
            if pace_diff < 0:
                base_prob = 0.05
                advantage_bonus = abs(pace_diff) * 0.02
                prob_overtake = (base_prob + advantage_bonus) * overtaking_factor
                
                if np.random.rand() < prob_overtake:
                    current_order[i], current_order[i+1] = d_behind, d_ahead
                    i += 2
                    continue
            i += 1
            
        full_standing = current_order + list(reversed(retired))
        history.append(full_standing)
        
    return history


def main():
    print("=" * 70)
    print("  NOTEBOOK 08: DETAILED RACE SIMULATION ENGINE")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    data = load_all_data()
    master = build_master_df(data)
    
    df = get_modern_era(master, start_year=2015)
    
    drivers_list = ['max_verstappen', 'norris', 'leclerc', 'piastri', 'sainz', 
                    'hamilton', 'russell', 'perez', 'alonso', 'stroll']
    
    names_dict = df.groupby('driver_id')['driver_name'].first().to_dict()
    
    grid_order = {d: i+1 for i, d in enumerate(drivers_list)}
    
    career = df.groupby('driver_id').agg(
        avg_pos=('position', 'mean'),
        dnf_rate=('is_dnf', 'mean')
    ).to_dict()
    
    pace_ratings = {d: career['avg_pos'].get(d, 10.0) for d in drivers_list}
    dnf_probs = {d: career['dnf_rate'].get(d, 0.1) for d in drivers_list}
    
    print("\nSimulation parameters:")
    for d in drivers_list:
        print(f"  {names_dict[d]:<25} | Start Grid: {grid_order[d]:<2} | Pace: {pace_ratings[d]:.1f} | DNF Rate: {dnf_probs[d]:.1%}")
        
    print("\nSimulating Monaco Grand Prix (Overtaking Factor = 0.05)...")
    monaco_wins = {d: 0 for d in drivers_list}
    for _ in range(2000):
        hist = simulate_lap_by_lap(grid_order, pace_ratings, dnf_probs, overtaking_factor=0.05, laps=78)
        winner = hist[-1][0]
        monaco_wins[winner] += 1
        
    print("\nSimulating Belgian (Spa) Grand Prix (Overtaking Factor = 0.8)...")
    spa_wins = {d: 0 for d in drivers_list}
    for _ in range(2000):
        hist = simulate_lap_by_lap(grid_order, pace_ratings, dnf_probs, overtaking_factor=0.8, laps=44)
        winner = hist[-1][0]
        spa_wins[winner] += 1
        
    print(f"\n{'Driver':<25} | {'Monaco Win %':<15} | {'Spa Win %':<15}")
    print("-" * 65)
    for d in drivers_list:
        name = names_dict[d]
        m_pct = monaco_wins[d] / 2000
        s_pct = spa_wins[d] / 2000
        print(f"{name:<25} | {m_pct:>13.1%} | {s_pct:>12.1%}")
        
    m_probs = {names_dict[d]: monaco_wins[d] / 2000 for d in drivers_list}
    s_probs = {names_dict[d]: spa_wins[d] / 2000 for d in drivers_list}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    m_sorted = dict(sorted(m_probs.items(), key=lambda x: x[1]))
    s_sorted = dict(sorted(s_probs.items(), key=lambda x: x[1]))
    
    ax1.barh(list(m_sorted.keys()), list(m_sorted.values()), color='red', alpha=0.7)
    ax1.set_title('Monaco GP Win Probabilities (Overtaking is Hard)')
    ax1.set_xlabel('Probability')
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    
    ax2.barh(list(s_sorted.keys()), list(s_sorted.values()), color='blue', alpha=0.7)
    ax2.set_title('Spa-Francorchamps Win Probabilities (Overtaking is Easier)')
    ax2.set_xlabel('Probability')
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '08_race_comparison.png'), dpi=150)
    plt.close()
    print("\n   Chart saved: outputs/08_race_comparison.png")
    
    print("\nDetailed Race Simulation complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
