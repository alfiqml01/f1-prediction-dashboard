import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px

st.title("Driver Style Clustering")
st.write("""
This tool groups Formula 1 drivers into different archetypes (e.g. Dominators vs Consistent Midfielders vs Backmarkers)
using unsupervised K-Means clustering. We project their multi-dimensional statistics into a 2D space using PCA.
""")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

cluster_file = os.path.join(OUTPUT_DIR, 'driver_clusters.csv')

if not os.path.exists(cluster_file):
    st.warning("⚠️ Driver clustering data has not been generated yet. Please run Notebook 07 first.")
else:
    @st.cache_data
    def load_clusters():
        return pd.read_csv(cluster_file)
        
    df_clusters = load_clusters()
    
    st.subheader("📊 Driver Archetypes Scatter Plot (PCA View)")
    st.write("""
    Each dot is a driver. Drivers positioned close to each other have similar career statistics
    (win rate, DNF rate, consistency, points per race, position changes).
    """)
    
    fig = px.scatter(
        df_clusters, x='pca_x', y='pca_y',
        color='archetype',
        hover_name='driver_name',
        hover_data=['total_races', 'total_wins', 'win_rate', 'dnf_rate', 'avg_position'],
        title="K-Means Driver Clustering projected on 2D PCA Space",
        labels={'pca_x': 'PCA Component 1 (Capability)', 'pca_y': 'PCA Component 2 (Consistency/Era Style)'},
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
    st.plotly_chart(fig, use_container_width=True)
    
    # Archetype selector to show drivers in that archetype
    st.subheader("📋 Explore Drivers by Archetype")
    arch = st.selectbox("Select Archetype to Inspect", df_clusters['archetype'].unique())
    
    arch_df = df_clusters[df_clusters['archetype'] == arch].sort_values('total_races', ascending=False)
    
    st.write(f"Showing drivers classified as: **{arch}**")
    st.dataframe(
        arch_df[['driver_name', 'total_races', 'total_wins', 'win_rate', 'dnf_rate', 'avg_position']],
        column_config={
            'win_rate': st.column_config.NumberColumn("Win Rate", format="%.1%"),
            'dnf_rate': st.column_config.NumberColumn("DNF Rate", format="%.1%"),
            'avg_position': st.column_config.NumberColumn("Avg Finish Position", format="%.1f")
        },
        use_container_width=True
    )
