"""
================================================================================
NOTEBOOK 02: RACE WINNER PREDICTION
================================================================================

WHY NOT JUST ACCURACY?
    In F1, only 1 driver wins out of 20. If a model always predicts "NOT WINNER",
    it has 95% accuracy! But it's 100% useless. We must look at ROC-AUC, Precision,
    Recall, and F1 Score to see if it actually finds the winner.

RUN THIS SCRIPT:
    python notebooks/02_race_winner_prediction.py
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
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.data_loader import load_all_data, build_master_df, get_modern_era
from src.feature_engineering import build_prediction_features, PREDICTION_FEATURES
from src.models import get_models, time_based_split, train_and_evaluate, get_feature_importance, save_model
from src.visualization import plot_model_comparison, plot_feature_importance, plot_confusion_matrix

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print("  NOTEBOOK 02: RACE WINNER PREDICTION")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # 1. LOAD AND PREPARE DATA
    # -------------------------------------------------------------------------
    print("\n1. Loading data...")
    data = load_all_data()
    master = build_master_df(data)
    
    # Filter to modern era (2000+) as old eras have different patterns
    print("Filtering to modern era (2000+)...")
    modern = get_modern_era(master, start_year=2000)
    print(f"   Modern era entries: {len(modern):,} rows")
    
    # Run the feature engineering pipeline
    print("\n2. Building features...")
    featured = build_prediction_features(modern, data)
    
    # -------------------------------------------------------------------------
    # 2. HANDLE MISSING VALUES AND CLEAN PREDICTION FEATURES
    # -------------------------------------------------------------------------
    print("\n3. Cleaning features and target variable...")
    
    # Features (X) and Target (y)
    X = featured[PREDICTION_FEATURES].copy()
    y = featured['is_winner'].copy()
    
    print(f"   Target variable class balance (0 = Lost, 1 = Won):")
    counts = y.value_counts()
    print(f"     Lost: {counts[0]:,} ({counts[0]/len(y)*100:.1f}%)")
    print(f"     Won:  {counts[1]:,} ({counts[1]/len(y)*100:.1f}%)")
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)
    
    # Save the imputer and scaler
    import joblib
    joblib.dump(imputer, os.path.join(MODELS_DIR, 'imputer.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    print("   Imputer and Scaler saved to models/")
    
    # -------------------------------------------------------------------------
    # 3. TIME-BASED SPLIT
    # -------------------------------------------------------------------------
    featured['row_idx'] = np.arange(len(featured))
    train_idx = featured[featured['season'] < 2020].index
    test_idx = featured[featured['season'] >= 2020].index
    
    X_train, X_test = X_imputed.loc[train_idx], X_imputed.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    
    X_train_scaled, X_test_scaled = X_scaled.loc[train_idx], X_scaled.loc[test_idx]
    
    print(f"   Train set size: {len(X_train):,} rows")
    print(f"   Test set size:  {len(X_test):,} rows")
    
    # -------------------------------------------------------------------------
    # 4. TRAIN AND COMPARE MODELS
    # -------------------------------------------------------------------------
    print("\n4. Training models...")
    models_dict = get_models(include_heavy=True)
    
    trained_models = {}
    results = []
    
    for name, model in models_dict.items():
        print(f"   Training {name}...")
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            probs = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]
            
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, preds),
            'Precision': precision_score(y_test, preds, zero_division=0),
            'Recall': recall_score(y_test, preds, zero_division=0),
            'F1 Score': f1_score(y_test, preds, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test, probs),
            'Log Loss': log_loss(y_test, probs),
        })
        trained_models[name] = model
        
        # Save model
        save_model(model, name.lower().replace(' ', '_'), MODELS_DIR)
        
    results_df = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)
    print("\nModel Evaluation Results (Sorted by ROC-AUC):")
    print(results_df.to_string(index=False))
    
    # Save comparison plot
    print("\n5. Saving comparison charts...")
    plot_model_comparison(results_df, metric='ROC-AUC', save_path=os.path.join(OUTPUT_DIR, '02_model_comparison_auc.png'))
    
    # Let's inspect the best model's feature importance
    best_model_name = results_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]
    print(f"\nBest model: {best_model_name}")
    
    imp_df = get_feature_importance(best_model, PREDICTION_FEATURES)
    plot_feature_importance(imp_df, top_n=15, save_path=os.path.join(OUTPUT_DIR, '02_feature_importance.png'))
    print("    Saved: outputs/02_feature_importance.png")
    
    # -------------------------------------------------------------------------
    # 5. SAMPLE RACE PREDICTION
    # -------------------------------------------------------------------------
    print("\n6. Testing predictions on a sample race (2024 Bahrain GP)...")
    sample_race_mask = (featured['season'] == 2024) & (featured['round'] == 1)
    sample_race = featured[sample_race_mask]
    
    if len(sample_race) > 0:
        sample_X = X_imputed.loc[sample_race.index]
        if best_model_name == 'Logistic Regression':
            sample_X_scaled = X_scaled.loc[sample_race.index]
            probs = best_model.predict_proba(sample_X_scaled)[:, 1]
        else:
            probs = best_model.predict_proba(sample_X)[:, 1]
            
        predictions = sample_race.copy()
        predictions['pred_prob'] = probs
        
        # Normalize probabilities so they sum to 100% across the grid
        predictions['pred_prob_normalized'] = predictions['pred_prob'] / predictions['pred_prob'].sum()
        
        # Sort by prediction
        predictions = predictions.sort_values('pred_prob_normalized', ascending=False)
        
        print("\nPredicted Win Probabilities (Bahrain 2024):")
        print(f"{'Driver':<25} {'Grid':<6} {'Actual Pos':<12} {'Raw Prob':<10} {'Normalized Prob':<15}")
        print("-" * 75)
        for _, row in predictions.head(10).iterrows():
            print(f"{row['driver_name']:<25} {int(row['grid']):<6} {int(row['position']):<12} "
                  f"{row['pred_prob']:.3f}     {row['pred_prob_normalized']:.1%}")
    else:
        print("Warning: 2024 Round 1 data not found. Skipping sample prediction.")
        
    print("\nRace Winner Prediction pipeline complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
