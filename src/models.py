"""
models.py — Model Training, Evaluation & Comparison
=====================================================

PURPOSE:
    Provides a standardized way to train, evaluate, and compare ML models.
    Instead of writing training code in every notebook, we use this module.

WHAT ARE ML MODELS?
    Machine Learning models are algorithms that learn patterns from data.
    You show them historical examples (training data), and they learn to
    make predictions on new, unseen data (test data).

STUDY RESOURCES FOR EACH MODEL:
    - Logistic Regression: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    - Random Forest: https://scikit-learn.org/stable/modules/ensemble.html#random-forests
    - XGBoost: https://xgboost.readthedocs.io/en/latest/tutorials/model.html
    - LightGBM: https://lightgbm.readthedocs.io/en/latest/
    - CatBoost: https://catboost.ai/docs/concepts/tutorials.html
    - Scikit-learn overview: https://scikit-learn.org/stable/tutorial/index.html

HOW TO CHANGE ALGORITHMS:
    1. Import the new model from its library
    2. Add it to the get_models() function below
    3. That's it! The training pipeline handles the rest.
    
    Example: Want to add a Support Vector Machine?
        from sklearn.svm import SVC
        models['SVM'] = SVC(probability=True)  # probability=True for ROC-AUC
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


def get_models(include_heavy=True):
    """
    Returns a dictionary of all models we want to compare.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    
    models = {}
    
    # === 1. LOGISTIC REGRESSION (Baseline) ===
    models['Logistic Regression'] = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )
    
    # === 2. RANDOM FOREST ===
    models['Random Forest'] = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    # === 3. GRADIENT BOOSTING ===
    models['Gradient Boosting'] = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    
    # === 4. XGBOOST ===
    try:
        from xgboost import XGBClassifier
        models['XGBoost'] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
    except ImportError:
        print("Warning: XGBoost not installed. Run: pip install xgboost")
    
    # === 5. LIGHTGBM ===
    if include_heavy:
        try:
            from lightgbm import LGBMClassifier
            models['LightGBM'] = LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
        except ImportError:
            print("Warning: LightGBM not installed. Run: pip install lightgbm")
    
    # === 6. CATBOOST ===
    if include_heavy:
        try:
            from catboost import CatBoostClassifier
            models['CatBoost'] = CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=0
            )
        except ImportError:
            print("Warning: CatBoost not installed. Run: pip install catboost")
    
    return models


def time_based_split(df, test_start_year=2020):
    """
    Split data into training and test sets based on TIME.
    """
    train = df[df['season'] < test_start_year].copy()
    test = df[df['season'] >= test_start_year].copy()
    
    print(f"Time-based split:")
    print(f"   Training: {train['season'].min()}-{train['season'].max()} "
          f"({len(train):,} rows)")
    print(f"   Testing:  {test['season'].min()}-{test['season'].max()} "
          f"({len(test):,} rows)")
    
    return train, test


def train_and_evaluate(models_dict, X_train, y_train, X_test, y_test):
    """
    Train all models and compare their performance.
    """
    results = []
    trained_models = {}
    
    for name, model in models_dict.items():
        print(f"\nTraining {name}...")
        
        try:
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = y_pred.astype(float)
            
            metrics = {
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, zero_division=0),
                'Recall': recall_score(y_test, y_pred, zero_division=0),
                'F1 Score': f1_score(y_test, y_pred, zero_division=0),
                'ROC-AUC': roc_auc_score(y_test, y_prob),
                'Log Loss': log_loss(y_test, y_prob),
            }
            
            results.append(metrics)
            trained_models[name] = model
            
            print(f"   ROC-AUC: {metrics['ROC-AUC']:.4f} | "
                  f"F1: {metrics['F1 Score']:.4f} | "
                  f"Accuracy: {metrics['Accuracy']:.4f}")
            
        except Exception as e:
            print(f"   {name} failed: {str(e)}")
    
    results_df = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)
    
    return results_df, trained_models


def cross_validate_model(model, X, y, cv=5):
    """
    Perform cross-validation to get a more reliable performance estimate.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
    
    return {
        'mean_auc': scores.mean(),
        'std_auc': scores.std(),
        'all_scores': scores
    }


def get_feature_importance(model, feature_names):
    """
    Extract feature importance from a trained model.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return pd.DataFrame({'feature': feature_names, 'importance': [0] * len(feature_names)})
    
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    df['importance_pct'] = df['importance'] / df['importance'].sum() * 100
    
    return df


def save_model(model, model_name, directory='models'):
    """
    Save a trained model to disk so we can reuse it later.
    """
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, f'{model_name}.pkl')
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")
    return filepath


def load_model(model_name, directory='models'):
    """Load a previously saved model from disk."""
    filepath = os.path.join(directory, f'{model_name}.pkl')
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model


if __name__ == '__main__':
    print("Available models:")
    models = get_models(include_heavy=False)
    for name in models:
        print(f"   {name}")
    print(f"\nTotal: {len(models)} models ready")
