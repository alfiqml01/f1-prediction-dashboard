import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px

st.title("Race Simulation & Strategy")
st.write("""
This tool uses a Monte Carlo simulation engine to simulate an F1 race lap-by-lap.
You can adjust the overtaking difficulty (e.g. Monaco vs. Spa) and DNF rates to see how they impact win probabilities.
""")

# Simplified simulation engine for dashboard use
def run_simulation(drivers, grid, pace, dnf_rates, laps=50, overtaking_factor=0.5, sims=500):
    results = {d: 0 for d in drivers}
    
    # Prob DNF per lap
    lap_dnf_probs = {d: 1.0 - (1.0 - dnf_rates.get(d, 0.1)) ** (1.0 / laps) for d in drivers}
    
    for _ in range(sims):
        # Start in grid order
        active = sorted(drivers, key=lambda d: grid[d])
        retired = []
        
        for lap in range(laps):
            if len(active) <= 1:
                break
                
            # DNF check
            still_active = []
            for d in active:
                if np.random.rand() < lap_dnf_probs[d]:
                    retired.append(d)
                else:
                    still_active.append(d)
            active = still_active
            
            # Overtaking
            i = 0
            while i < len(active) - 1:
                d1 = active[i]
                d2 = active[i+1]
                
                # Pace difference (lower is faster)
                pace_diff = pace[d2] - pace[d1]
                
                if pace_diff < 0: # behind is faster
                    swap_prob = (0.05 + abs(pace_diff) * 0.02) * overtaking_factor
                    if np.random.rand() < swap_prob:
                        active[i], active[i+1] = d2, d1
                        i += 2
                        continue
                i += 1
                
        winner = active[0] if len(active) > 0 else retired[-1]
        results[winner] += 1
        
    return {d: count / sims for d, count in results.items()}

# Configuration sidebar
st.sidebar.header("Simulation Settings")
laps = st.sidebar.slider("Number of Laps", 10, 80, 50)
overtaking = st.sidebar.slider("Overtaking Difficulty (0 = Monaco, 1 = Spa)", 0.05, 1.0, 0.5)
sims = st.sidebar.slider("Number of Simulations", 100, 2000, 500)

drivers = ["Verstappen", "Norris", "Leclerc", "Piastri", "Sainz", "Hamilton"]
grid = {"Verstappen": 1, "Norris": 2, "Leclerc": 3, "Piastri": 4, "Sainz": 5, "Hamilton": 6}

st.subheader("🏎️ Edit Driver Attributes")
pace = {}
dnfs = {}

cols = st.columns(6)
for i, d in enumerate(drivers):
    with cols[i]:
        st.markdown(f"**{d} (Grid: {grid[d]})**")
        # Pace rating: lower is better (represents average finishing pos)
        pace[d] = st.slider(f"Pace Rating for {d}", 1.0, 15.0, float(grid[d]), key=f"pace_{d}")
        dnfs[d] = st.slider(f"DNF Rate for {d}", 0.0, 0.5, 0.08, key=f"dnf_{d}")

if st.button("🎲 Run Monte Carlo Simulation"):
    with st.spinner("Simulating..."):
        win_probs = run_simulation(drivers, grid, pace, dnfs, laps, overtaking, sims)
        
        probs_df = pd.DataFrame(list(win_probs.items()), columns=['Driver', 'Probability'])
        probs_df = probs_df.sort_values('Probability', ascending=False)
        
        st.success("Simulation Complete!")
        
        fig = px.bar(
            probs_df, x='Probability', y='Driver', 
            orientation='h', 
            title="Win Probabilities from Monte Carlo Simulation",
            labels={'Probability': 'Win Chance', 'Driver': 'Driver'},
            color='Probability',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
