import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

st.title("Driver Performance Analysis")
st.write("Explore driver career performance profiles, consistency indices, and teammate head-to-head records.")

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

# Load raw datasets
@st.cache_data
def load_driver_data():
    drivers = pd.read_csv(os.path.join(DATA_DIR, 'f1_drivers.csv'))
    results = pd.read_csv(os.path.join(DATA_DIR, 'f1_results.csv'))
    drivers['full_name'] = drivers['given_name'] + " " + drivers['family_name']
    
    # Calculate simple wins/entries per driver
    results['position_num'] = pd.to_numeric(results['position'], errors='coerce')
    results['is_winner'] = (results['position_num'] == 1).astype(int)
    results['is_podium'] = (results['position_num'] <= 3).astype(int)
    
    stats = results.groupby('driver_id').agg(
        races=('position', 'count'),
        wins=('is_winner', 'sum'),
        podiums=('is_podium', 'sum'),
        avg_finish=('position_num', 'mean')
    ).reset_index()
    
    stats = pd.merge(stats, drivers[['driver_id', 'full_name', 'nationality']], on='driver_id')
    return stats, results

stats, results = load_driver_data()

# Select a driver
selected_name = st.selectbox("Select Driver", stats.sort_values('full_name')['full_name'])
driver_id = stats[stats['full_name'] == selected_name]['driver_id'].values[0]
driver_profile = stats[stats['driver_id'] == driver_id].iloc[0]

# Display profile card
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Races Started", f"{int(driver_profile['races'])}")
with col2:
    st.metric("Total Wins", f"{int(driver_profile['wins'])}")
with col3:
    st.metric("Podium Finishes", f"{int(driver_profile['podiums'])}")
with col4:
    st.metric("Average Finish Position", f"{driver_profile['avg_finish']:.1f}")

# Career History Plot
st.subheader("🏁 Career Finish Position Distribution")
d_results = results[results['driver_id'] == driver_id].dropna(subset=['position_num'])

if len(d_results) > 0:
    fig = px.histogram(
        d_results, x='position_num', 
        nbins=20, 
        title=f"Finishing Positions for {selected_name}",
        labels={'position_num': 'Finishing Position', 'count': 'Number of Races'},
        color_discrete_sequence=['#E10600']
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No finishing position data available for this driver.")
