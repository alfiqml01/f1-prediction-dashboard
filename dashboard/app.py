import streamlit as st
import os
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Formula 1 Analytics & Prediction Platform",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling the dashboard
st.markdown("""
<style>
    .main-title {
        color: #E10600;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        color: #C0C0C0;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #15151E;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #E10600;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🏎️ Formula 1 Analytics Platform</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">A complete Machine Learning and Data Science dashboard built around Formula 1 historical data.</p>', unsafe_allow_html=True)

# Grid Layout for Overview Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color:#C0C0C0; margin:0; font-size: 0.9rem;">SEASONS COVERED</h3>
        <h1 style="color:#FFFFFF; margin:5px 0 0 0; font-size: 2.2rem;">1950 - 2026</h1>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color:#C0C0C0; margin:0; font-size: 0.9rem;">TOTAL RACE RESULTS</h3>
        <h1 style="color:#FFFFFF; margin:5px 0 0 0; font-size: 2.2rem;">25,939+</h1>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color:#C0C0C0; margin:0; font-size: 0.9rem;">DRIVERS PROFILED</h3>
        <h1 style="color:#FFFFFF; margin:5px 0 0 0; font-size: 2.2rem;">879+</h1>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color:#C0C0C0; margin:0; font-size: 0.9rem;">CONSTRUCTORS</h3>
        <h1 style="color:#FFFFFF; margin:5px 0 0 0; font-size: 2.2rem;">214+</h1>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.header("🏁 Welcome to the F1 Analytics Platform!")
st.write("""
This interactive platform lets you explore the data-driven side of Formula 1. Navigate through the sidebar pages to:
* **🔮 Race Prediction**: Predict who will win upcoming races using our machine learning models.
* **📊 Driver Analysis**: Compare driver performance, teammates, and career profiles.
* **🎮 Race Simulation**: Run lap-by-lap Monte Carlo race simulations under different track profiles.
* **🏆 Championship Predictor**: See how the championship odds evolve race-by-race.
* **🐐 GOAT Index**: Explore the Greatest of All Time rankings with customizable weights.
* **🧬 Driver Clustering**: Discover driver style archetypes using unsupervised machine learning.
""")

st.info("💡 **Tip for beginners**: Each page contains simplified definitions and visual guides explaining how the underlying data science algorithms work!")
