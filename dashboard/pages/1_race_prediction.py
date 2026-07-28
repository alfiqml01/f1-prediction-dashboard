import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

st.title("Race Winner Prediction")
st.write("""
This tool uses historical driver and constructor statistics to predict the winner of a race.
Select a model, input starting grid details, and see prediction odds!
""")

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

# Check if model files exist
required_models = ['xgboost.pkl', 'imputer.pkl', 'scaler.pkl']
models_exist = all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in required_models)

if not models_exist:
    st.warning("⚠️ Machine Learning models have not been trained yet. Please run the training notebooks/scripts first or check back later.")
else:
    # Load model, imputer, scaler
    @st.cache_resource
    def load_prediction_assets():
        model = joblib.load(os.path.join(MODELS_DIR, 'xgboost.pkl'))
        imputer = joblib.load(os.path.join(MODELS_DIR, 'imputer.pkl'))
        scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
        return model, imputer, scaler

    model, imputer, scaler = load_prediction_assets()
    
    # Load driver options
    drivers_df = pd.read_csv(os.path.join(DATA_DIR, 'f1_drivers.csv'))
    drivers_df['full_name'] = drivers_df['given_name'] + " " + drivers_df['family_name']
    drivers_dict = dict(zip(drivers_df['full_name'], drivers_df['driver_id']))
    
    # Select track context
    circuits_df = pd.read_csv(os.path.join(DATA_DIR, 'f1_circuits.csv'))
    track = st.selectbox("Select Circuit", circuits_df['circuit_name'])
    
    # Interactive Grid Input
    st.subheader("🏎️ Configure Starting Grid (Top 5 Drivers)")
    
    cols = st.columns(5)
    selected_drivers = []
    grid_positions = []
    
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"**Grid Position {i+1}**")
            d_name = st.selectbox(f"Driver for P{i+1}", list(drivers_dict.keys()), key=f"d_{i}", index=min(i, len(drivers_dict)-1))
            selected_drivers.append(drivers_dict[d_name])
            grid_positions.append(i+1)
            
    # Run Prediction Button
    if st.button("🔮 Calculate Odds"):
        # Let's create dummy feature values for prediction based on the selected grid and drivers
        # In a real pipeline, we'd pull their exact historical stats. Here we simulate them for the dashboard.
        features = [
            'grid', 'quali_position', 'driver_avg_pos_last3', 'driver_avg_pos_last5',
            'driver_avg_points_last5', 'driver_avg_delta_last5', 'cumulative_win_rate',
            'cumulative_podium_rate', 'cumulative_dnf_rate', 'constructor_avg_points_last5',
            'constructor_avg_pos_last5', 'constructor_win_rate', 'circuit_experience',
            'driver_circuit_avg_pos', 'season_momentum', 'prev_season_driver_pos',
            'prev_season_constructor_pos'
        ]
        
        preds_list = []
        for d_id, grid in zip(selected_drivers, grid_positions):
            # Mock historical features based on grid position for demonstration
            # In a production app, these would be loaded from our database/loader
            mock_data = {
                'grid': grid,
                'quali_position': grid,
                'driver_avg_pos_last3': grid + 1.0,
                'driver_avg_pos_last5': grid + 1.5,
                'driver_avg_points_last5': max(0.0, 25.0 - grid * 3.0),
                'driver_avg_delta_last5': 0.1,
                'cumulative_win_rate': 0.05 if grid > 1 else 0.25,
                'cumulative_podium_rate': 0.15 if grid > 3 else 0.50,
                'cumulative_dnf_rate': 0.08,
                'constructor_avg_points_last5': max(0.0, 40.0 - grid * 5.0),
                'constructor_avg_pos_last5': grid + 1.0,
                'constructor_win_rate': 0.15 if grid > 2 else 0.40,
                'circuit_experience': 5.0,
                'driver_circuit_avg_pos': grid + 1.2,
                'season_momentum': 0.5,
                'prev_season_driver_pos': grid + 1,
                'prev_season_constructor_pos': max(1, grid // 2)
            }
            
            # Predict
            df_pred = pd.DataFrame([mock_data])[features]
            df_imputed = pd.DataFrame(imputer.transform(df_pred), columns=features)
            
            prob = model.predict_proba(df_imputed)[:, 1][0]
            preds_list.append((d_id, prob))
            
        # Display Results
        probs_df = pd.DataFrame(preds_list, columns=['driver_id', 'prob'])
        # Normalize
        probs_df['normalized_prob'] = probs_df['prob'] / probs_df['prob'].sum()
        
        # Add names
        probs_df['Driver Name'] = probs_df['driver_id'].map(dict(zip(drivers_df['driver_id'], drivers_df['full_name'])))
        probs_df = probs_df.sort_values('normalized_prob', ascending=False)
        
        st.write("### Predicted Win Probabilities:")
        for _, row in probs_df.iterrows():
            st.metric(label=row['Driver Name'], value=f"{row['normalized_prob']:.1%}")
