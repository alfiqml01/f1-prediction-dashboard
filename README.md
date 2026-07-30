# 🏎️ Formula 1 Analytics Platform

A comprehensive end-to-end data science and machine learning project that analyzes historical Formula 1 data and predicts future race outcomes, championship standings, and driver archetypes. Built to celebrate the highly anticipated return of the **Sepang International Circuit** to the F1 calendar in October 2026.

This project processes F1 data from the inaugural 1950 season right up to the 2026 Japanese GP (sourced from Kaggle), turning raw statistics into interactive, model-driven insights.

## ✨ Features

The project is divided into analytical pipelines (notebooks) and an interactive web dashboard:

1. **🔮 Race Winner Prediction**: Uses XGBoost and Random Forest classifiers to predict the probability of a driver winning a specific race based on historical patterns, grid position, and track characteristics.
2. **📊 Driver Performance Analysis**: Compares teammates head-to-head, measures consistency, analyzes overtaking ability, and adjusts stats across different eras of the sport.
3. **🎮 Strategy Optimization & Monte Carlo Simulation**: Runs 10,000 probabilistic race scenarios to simulate lap-by-lap race outcomes, accounting for random events like safety cars and DNFs.
4. **🏆 Championship Forecasting**: Forecasts end-of-season championship probabilities based on mid-season standings using machine learning.
5. **🐐 Objective GOAT Index**: Uses Principal Component Analysis (PCA) to build an era-adjusted "Greatest Of All Time" ranking based on a composite score of multi-dimensional performance metrics.
6. **🧬 Driver Clustering**: Uses unsupervised K-Means clustering and DBSCAN to group drivers into behavioral archetypes (e.g., Dominators vs. Consistent Midfielders).

## 🛠️ Technology Stack

- **Data Processing**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn`, `xgboost`, `lightgbm`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Web App/Dashboard**: `streamlit`

## 📁 Project Structure

```
F1 prediction/
│
├── data/                  # Raw CSV files from Kaggle (results, drivers, constructors, etc.)
├── src/                   # Core Python modules
│   ├── data_loader.py         # Central data loading and merging logic
│   └── feature_engineering.py # Functions to create ML features
│
├── notebooks/             # ML Pipelines & Analysis
│   ├── 01_data_exploration.py
│   ├── 02_race_winner_prediction.py
│   ├── 03_driver_performance_analysis.py
│   ├── 04_strategy_optimization.py
│   ├── 05_championship_prediction.py
│   ├── 06_goat_ranking.py
│   ├── 07_driver_clustering.py
│   ├── 08_race_simulation.py
│   └── 09_constructor_prediction.py
│
├── models/                # Saved trained ML models (.joblib)
├── outputs/               # Generated CSVs and visualizations (Note: CSVs are tracked in git)
│
├── dashboard/             # Streamlit Application
│   ├── HOME.py            # Main entry point for the dashboard
│   └── pages/             # Individual interactive dashboard pages
│
├── requirements.txt       # Python dependencies
└── README.md              # You are here!
```

## 🚀 Getting Started

### 1. Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/f1-prediction.git
cd f1-prediction
pip install -r requirements.txt
```

### 2. Run the Analytical Pipeline

Before running the dashboard, ensure you have generated all the necessary models and output files by running the notebook scripts in order:

```bash
# Example: Run Data Exploration
python notebooks/01_data_exploration.py

# Example: Generate Driver Clusters (Required for the dashboard's Clustering page)
python notebooks/07_driver_clustering.py
```
*(Tip: Run all notebooks 01 through 09 to fully populate the `models/` and `outputs/` directories).*

### 3. Launch the Dashboard

Once the models and data are generated, start the Streamlit web application:

```bash
streamlit run dashboard/HOME.py
```

The application will be accessible at `http://localhost:8501`.

## 📈 Why This Project?

With Sepang returning to the F1 calendar in October 2026, fans deserve smarter, data-driven previews rather than just reading headlines. By grounding this project in the latest available context (as of the 2026 Japanese GP), we can preview exactly what this comeback could look like through the lens of data analytics. 

Let's see what the numbers say! 🇲🇾✨

---
*Disclaimer: This project is for educational and portfolio purposes. Formula 1 data is property of FOM (Formula One Management) and respective owners.*
