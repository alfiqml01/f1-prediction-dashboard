import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px

st.title("Race Simulation & Strategy")
st.write("""
This tool uses a Monte Carlo simulation engine to simulate an F1 race lap-by-lap.
Select a circuit preset to load realistic overtaking values, or adjust the sliders manually to run your custom scenarios!
""")

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

# Load circuits data
@st.cache_data
def load_circuits():
    return pd.read_csv(os.path.join(DATA_DIR, 'f1_circuits.csv'))

circuits_df = load_circuits()
circuits_dict = dict(zip(circuits_df['circuit_name'], circuits_df['circuit_id']))

# Pre-defined Overtaking Factors
overtaking_presets = {
    'monaco': 0.1,
    'hungaroring': 0.25,
    'marina_bay': 0.3,
    'albert_park': 0.4,
    'catalunya': 0.45,
    'yas_marina': 0.5,
    'silverstone': 0.65,
    'monza': 0.7,
    'bahrain': 0.7,
    'interlagos': 0.75,
    'baku': 0.75,
    'spa': 0.8,
    'red_bull_ring': 0.85
}

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

# Circuit selection preset
default_index = 0
sorted_names = sorted(circuits_df['circuit_name'].unique())
for idx, name in enumerate(sorted_names):
    if "Spa-Francorchamps" in name or "Spa" in name:
        default_index = idx
        break

selected_circuit_name = st.sidebar.selectbox("Select Circuit Preset", sorted_names, index=default_index)
selected_circuit_id = circuits_dict[selected_circuit_name]

# Get preset or default to 0.5
default_overtaking = overtaking_presets.get(selected_circuit_id, 0.5)

# Detailed preset reasons/explanations
overtaking_reasons = {
    'monaco': "Tight street track with narrow lanes. Passing is virtually impossible.",
    'hungaroring': "Twisty, low-speed corners and a short straight. Often called 'Monaco without walls'.",
    'marina_bay': "Bumpy, slow street circuit with many corners, making passing quite difficult.",
    'albert_park': "A semi-street circuit. Overtaking is tricky, though layout changes have helped.",
    'catalunya': "Historically hard to overtake due to dirty air, but a long pit straight offers DRS passing.",
    'yas_marina': "Long straights help passing under DRS, but technical low-speed sections limit overtaking.",
    'silverstone': "Wide track with sweeping fast turns allowing drivers to take multiple lines and overtake.",
    'monza': "The Temple of Speed. Long straights and huge slipstreams make passing highly common.",
    'bahrain': "Multiple long straights with heavy braking zones make it a classic overtaking track.",
    'interlagos': "Short lap with a massive uphill main straight allowing dramatic slipstreams.",
    'baku': "Massive 2km main straight facilitates easy DRS drafting, despite tight castle section.",
    'spa': "Huge DRS zones at Kemmel Straight and Blanchimont allow high-speed overtaking.",
    'red_bull_ring': "Short track with three consecutive DRS zones and heavy braking zones."
}

reason = overtaking_reasons.get(selected_circuit_id, "Standard race circuit layout with average overtaking characteristics.")
st.sidebar.info(f"ℹ️ **Preset loaded**: {reason}")

st.sidebar.write("---")

laps = st.sidebar.slider("Number of Laps", 10, 80, 50)

# Overtaking slider (initialized to the preset default)
overtaking = st.sidebar.slider(
    "Ease of Overtaking", 
    0.05, 1.0, 
    value=default_overtaking,
    help="Controls how easily drivers can pass each other. Circuit presets set realistic defaults, but you can adjust this value to test custom scenarios."
)

# Visual category for current slider value
if overtaking <= 0.15:
    category = "🔴 Very Difficult"
elif overtaking <= 0.35:
    category = "🟠 Difficult"
elif overtaking <= 0.60:
    category = "🟡 Moderate"
elif overtaking <= 0.80:
    category = "🟢 Easy"
else:
    category = "🔵 Very Easy"

st.sidebar.markdown(f"### **`{overtaking:.2f}` {category}**")

st.sidebar.markdown("""
<div style="font-size: 0.85rem; color: #888;">
Controls how easily drivers can pass each other. Circuit presets set realistic defaults, but you can adjust this value to test custom scenarios.
</div>

**How it affects the simulation:**
* 📉 **Lower values** → Starting grid position matters more.
* 📈 **Higher values** → Fast drivers can recover from poor qualifying.
* 🔄 **Higher values** → More position changes during the race.
""", unsafe_allow_html=True)

st.sidebar.write("---")

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
            title=f"Win Probabilities from Monte Carlo Simulation ({selected_circuit_name})",
            labels={'Probability': 'Win Chance', 'Driver': 'Driver'},
            color='Probability',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
