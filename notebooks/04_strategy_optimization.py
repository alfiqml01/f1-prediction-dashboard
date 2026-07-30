"""
================================================================================
NOTEBOOK 04: STRATEGY OPTIMIZATION & MONTE CARLO SIMULATION
================================================================================

WHY MONTE CARLO?
    A single F1 race has too many random factors: crashes, mechanical failures,
    safety cars, overtaking. A deterministic formula can't capture this.
    Instead, we model the probability of each event, roll random "dice" for each lap,
    simulate the race 10,000 times, and look at the distribution of outcomes.

RUN THIS SCRIPT:
    python notebooks/04_strategy_optimization.py
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


def run_monte_carlo_race(grid_positions, dnf_rates, pace_ratings, num_simulations=1000, laps=50):
    """
    Run a Monte Carlo simulation of a single race.
    """
    drivers = list(grid_positions.keys())
    results = {d: [] for d in drivers}
    
    lap_dnf_probs = {}
    for d in drivers:
        race_rate = dnf_rates.get(d, 0.1)
        race_rate = max(0.01, min(0.99, race_rate))
        lap_dnf_probs[d] = 1.0 - (1.0 - race_rate) ** (1.0 / laps)
        
    for sim in range(num_simulations):
        positions = list(grid_positions.keys())
        positions = sorted(positions, key=lambda d: grid_positions[d])
        
        active_drivers = positions.copy()
        retired_drivers = []
        
        for lap in range(laps):
            if len(active_drivers) <= 1:
                break
                
            still_active = []
            for d in active_drivers:
                if np.random.rand() < lap_dnf_probs[d]:
                    retired_drivers.append(d)
                else:
                    still_active.append(d)
            active_drivers = still_active
            
            for i in range(len(active_drivers) - 1):
                d1 = active_drivers[i]
                d2 = active_drivers[i+1]
                
                pace_diff = pace_ratings.get(d1, 10) - pace_ratings.get(d2, 10)
                swap_chance = 0.05 + max(-0.04, min(0.15, pace_diff * 0.01))
                
                if np.random.rand() < swap_chance:
                    active_drivers[i], active_drivers[i+1] = d2, d1
                    
        final_order = active_drivers + list(reversed(retired_drivers))
        
        for rank, d in enumerate(final_order):
            results[d].append(rank + 1)
            
    return results


def main():
    print("=" * 70)
    print("  NOTEBOOK 04: STRATEGY OPTIMIZATION & SIMULATION")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    data = load_all_data()
    master = build_master_df(data)
    
    df = get_modern_era(master, start_year=2015)
    
    # -------------------------------------------------------------------------
    # 1. DNF & RELIABILITY ANALYSIS BY CONSTRUCTOR
    # -------------------------------------------------------------------------
    print("\n1. Analyzing Constructor Reliability (2015+)...")
    constructor_dnf = df.groupby('constructor_name').agg(
        races=('is_dnf', 'count'),
        dnfs=('is_dnf', 'sum')
    ).reset_index()
    constructor_dnf['dnf_rate'] = constructor_dnf['dnfs'] / constructor_dnf['races']
    
    print("\nConstructor Reliability Ranking (Sorted by DNF Rate, Lower = More Reliable):")
    ranking = constructor_dnf[constructor_dnf['races'] >= 50].sort_values('dnf_rate')
    print(ranking[['constructor_name', 'races', 'dnfs', 'dnf_rate']].to_string(index=False))
    
    # -------------------------------------------------------------------------
    # 2. OPTIMAL GRID POSITION ANALYSIS
    # -------------------------------------------------------------------------
    print("\n2. Grid Position Finish Probability & Points Return...")
    grid_stats = df.groupby('grid').agg(
        entries=('position', 'count'),
        wins=('is_winner', 'sum'),
        podiums=('is_podium', 'sum'),
        avg_finish=('position', 'mean'),
        avg_points=('points', 'mean')
    ).reset_index()
    
    grid_stats = grid_stats[(grid_stats['grid'] >= 1) & (grid_stats['grid'] <= 20)]
    grid_stats['win_rate'] = grid_stats['wins'] / grid_stats['entries']
    grid_stats['podium_rate'] = grid_stats['podiums'] / grid_stats['entries']
    
    print("\nGrid Position Conversion Rates:")
    print(grid_stats[['grid', 'entries', 'avg_finish', 'avg_points', 'win_rate', 'podium_rate']].to_string(index=False))
    
    # Plot expected points by grid
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=grid_stats, x='grid', y='avg_points', color='#E10600', ax=ax)
    ax.set_xlabel('Starting Grid Position', fontsize=12)
    ax.set_ylabel('Expected Points Return', fontsize=12)
    ax.set_title('Expected Points Return by Starting Grid Position (2015+)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_grid_points_expectation.png'), dpi=150)
    plt.close()
    print("   Chart saved: outputs/04_grid_points_expectation.png")
    
    # -------------------------------------------------------------------------
    # 3. MONTE CARLO RACE SIMULATION
    # -------------------------------------------------------------------------
    print("\n3. Setting up Monte Carlo Race Simulation...")
    
    df_2024 = df[df['season'] == 2024]
    top_drivers = df_2024['driver_id'].value_counts().head(10).index
    mock_grid = {driver: i+1 for i, driver in enumerate(top_drivers)}
    
    career = df.groupby('driver_id').agg(
        dnfs=('is_dnf', 'sum'),
        races=('is_dnf', 'count'),
        avg_pos=('position', 'mean')
    ).reset_index()
    
    career['dnf_rate'] = career['dnfs'] / career['races']
    
    dnf_rates = dict(zip(career['driver_id'], career['dnf_rate']))
    pace_ratings = dict(zip(career['driver_id'], career['avg_pos']))
    
    print("\nSimulation Inputs:")
    print(f"{'Driver':<20} {'Grid Start':<12} {'Career DNF Rate':<18} {'Career Avg Position':<20}")
    print("-" * 75)
    for d in top_drivers:
        name = df[df['driver_id'] == d]['driver_name'].iloc[0]
        print(f"{name:<20} {mock_grid[d]:<12} {dnf_rates.get(d, 0.1):.1%}              {pace_ratings.get(d, 10.0):.1f}")
        
    print("\nRunning 1,000 simulated races...")
    sim_raw = run_monte_carlo_race(
        grid_positions=mock_grid,
        dnf_rates=dnf_rates,
        pace_ratings=pace_ratings,
        num_simulations=1000,
        laps=50
    )
    
    win_probabilities = {}
    for d, positions in sim_raw.items():
        wins = sum(1 for pos in positions if pos == 1)
        name = df[df['driver_id'] == d]['driver_name'].iloc[0]
        win_probabilities[name] = wins / len(positions)
        
    print("\nSimulation Output (Win Probabilities):")
    sorted_probs = sorted(win_probabilities.items(), key=lambda x: x[1], reverse=True)
    for name, prob in sorted_probs:
        print(f"   {name:<25} -> {prob:.1%}")
        
    plot_simulation_results(win_probabilities, title='Monte Carlo Simulated Win Probabilities', save_path=os.path.join(OUTPUT_DIR, '04_simulation_results.png'))
    print("   Chart saved: outputs/04_simulation_results.png")
    
    print("\nStrategy Optimization and Simulation complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
