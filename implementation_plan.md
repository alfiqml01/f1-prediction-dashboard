# 🏎️ Complete F1 Analytics & Prediction Platform

A comprehensive data science portfolio project that combines historical analysis, machine learning prediction, unsupervised learning, simulation, and an interactive dashboard — all built around Formula 1 racing data.

## Your Data at a Glance

| Dataset | Rows | Columns | Coverage |
|---------|------|---------|----------|
| `f1_results.csv` | 25,939 | 15 | 1950–2026 race results |
| `f1_races.csv` | 1,171 | 7 | All race metadata |
| `f1_drivers.csv` | 879 | 8 | Driver profiles |
| `f1_qualifying.csv` | 11,036 | 8 | 1994–2026 qualifying |
| `f1_constructors.csv` | 214 | 4 | All constructors |
| `f1_constructor_standings.csv` | 924 | 6 | End-of-season standings |
| `f1_driver_standings.csv` | 3,128 | 6 | End-of-season standings |
| `f1_circuits.csv` | 78 | 7 | Circuit locations |

---

## Project Architecture

```
F1 prediction/
├── data/                          # Raw CSV files (already present)
├── notebooks/                     # Jupyter notebooks (main deliverable)
│   ├── 01_data_exploration.ipynb
│   ├── 02_race_winner_prediction.ipynb
│   ├── 03_driver_performance_analysis.ipynb
│   ├── 04_strategy_optimization.ipynb
│   ├── 05_championship_prediction.ipynb
│   ├── 06_goat_ranking.ipynb
│   ├── 07_driver_clustering.ipynb
│   ├── 08_race_simulation.ipynb
│   └── 09_constructor_prediction.ipynb
├── src/                           # Reusable Python modules
│   ├── __init__.py
│   ├── data_loader.py             # Central data loading & merging
│   ├── feature_engineering.py     # Shared feature pipelines
│   ├── models.py                  # Model wrappers & evaluation
│   └── visualization.py          # Common plotting functions
├── dashboard/                     # Streamlit interactive dashboard
│   ├── app.py                     # Main dashboard app
│   ├── pages/
│   │   ├── 1_race_prediction.py
│   │   ├── 2_driver_analysis.py
│   │   ├── 3_strategy.py
│   │   ├── 4_championship.py
│   │   ├── 5_goat_index.py
│   │   └── 6_clustering.py
│   └── assets/
├── models/                        # Saved trained models (.pkl)
├── outputs/                       # Generated charts, reports
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Beginner-Friendly Concept Explanations

> [!NOTE]
> Since you're a beginner, here's a quick glossary of every technique we'll use, explained simply:

### Machine Learning Models

| Model | What It Does (Simple) | Analogy |
|-------|----------------------|---------|
| **Random Forest** | Makes many "decision trees" and takes a vote | Asking 100 experts and going with the majority |
| **XGBoost** | Builds trees one-by-one, each fixing mistakes of the last | A student retaking a test, studying what they got wrong each time |
| **LightGBM** | Like XGBoost but faster — grows trees leaf-by-leaf | Same student, but with a smarter study method |
| **Logistic Regression** | Draws a line to separate "yes" from "no" | Deciding if an email is spam based on a score |
| **LSTM** | A neural network with "memory" for sequences | Reading a book and remembering earlier chapters |
| **ARIMA** | Predicts future values based on past patterns | Predicting tomorrow's temperature from last week's weather |
| **Prophet** | Facebook's tool for time-series with trends & seasons | A weather app that knows summer is always hotter |

### Unsupervised Learning

| Technique | What It Does | Analogy |
|-----------|-------------|---------|
| **K-Means** | Groups similar items into K clusters | Sorting laundry into piles by color |
| **DBSCAN** | Finds clusters of any shape, ignores outliers | Finding friend groups at a party — some people stand alone |
| **PCA** | Reduces many features to fewer important ones | Summarizing a 10-page essay into 3 key bullet points |

### Other Techniques

| Technique | What It Does | Analogy |
|-----------|-------------|---------|
| **Monte Carlo Simulation** | Runs a scenario 10,000 times with randomness | Rolling dice 10,000 times to figure out average outcomes |
| **Feature Engineering** | Creating new useful data from existing data | Calculating "position gained = start - finish" from raw data |
| **Cross-Validation** | Testing a model by hiding parts of the data | Studying with flash cards, but hiding some to test yourself |
| **Weighted Scoring** | Giving more importance to harder achievements | A university giving more credit hours to harder classes |

### Additional Models We'll Explore (Beyond Your List)

| Model | What It Does | Why Use It Here |
|-------|-------------|-----------------|
| **CatBoost** | Like XGBoost but handles text/categories natively | F1 data has many categories (driver names, teams, circuits) |
| **Gradient Boosting (sklearn)** | The "original" boosting — good baseline | Simpler to understand before jumping to XGBoost |
| **Neural Network (MLP)** | Layers of connected "neurons" | Can catch complex non-linear patterns in race data |
| **Stacking Ensemble** | Combines predictions from multiple models | Asking RF, XGBoost, AND LightGBM then combining their answers |
| **Support Vector Machine (SVM)** | Finds the best dividing boundary between classes | Drawing the widest possible line between winners and losers |
| **ElasticNet** | Regression with penalty to prevent overfitting | Keeps the model simple and generalizable |
| **Bayesian Optimization** | Smart hyperparameter tuning | Instead of trying every combination, intelligently picks the next best one to try |

---

## Proposed Changes — Module by Module

### Phase 0: Project Setup & Data Infrastructure

#### [NEW] [requirements.txt](file:///c:/Users/maiqmal/Desktop/proj/F1%20prediction/requirements.txt)
All Python dependencies: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `matplotlib`, `seaborn`, `plotly`, `streamlit`, `scipy`, `statsmodels`, `shap`, `joblib`, `jupyter`.

#### [NEW] [src/data_loader.py](file:///c:/Users/maiqmal/Desktop/proj/F1%20prediction/src/data_loader.py)
Central module to load all 8 CSVs, merge them into a master DataFrame, handle data types, and provide clean accessor functions. This avoids repeating `pd.read_csv()` in every notebook.

#### [NEW] [src/feature_engineering.py](file:///c:/Users/maiqmal/Desktop/proj/F1%20prediction/src/feature_engineering.py)
Shared feature pipeline:
- **Rolling averages**: Last 3/5/10 race results per driver
- **Position delta**: `grid - position` (how many positions gained/lost)
- **Win rate**: Career and recent (last N races) win percentages
- **Constructor strength**: Average constructor points per season
- **Circuit familiarity**: Number of previous races at a circuit
- **Qualifying gap**: Time difference to pole position
- **Season momentum**: Points trend over recent races
- **DNF rate**: Historical reliability per driver/constructor
- **Head-to-head**: Teammate comparison stats

#### [NEW] [src/models.py](file:///c:/Users/maiqmal/Desktop/proj/F1%20prediction/src/models.py)
Model wrappers with standardized `train()`, `evaluate()`, `predict()` interfaces. Includes cross-validation, hyperparameter grid definitions, and model comparison utilities.

#### [NEW] [src/visualization.py](file:///c:/Users/maiqmal/Desktop/proj/F1%20prediction/src/visualization.py)
Reusable plotting: radar charts, heatmaps, position gain charts, probability distributions, era timelines.

---

### Phase 1: Data Exploration & Cleaning (Notebook 01)

#### [NEW] [notebooks/01_data_exploration.ipynb](file:///c:/Users/maiqmal/Desktop/proj/F1%20prediction/notebooks/01_data_exploration.ipynb)

**What you'll learn**: How to inspect, clean, and understand raw data before modeling.

- Load all 8 datasets, display shapes and dtypes
- Handle missing values:
  - `time`, `fastest_lap`, `fastest_lap_rank` — 66% null (expected: only finishers have times)
  - `qualifying.q2`, `q3` — expected nulls (drivers eliminated in Q1/Q2)
  - `driver_standings.position` — 47% null (edge case handling)
- Merge datasets into a master race-level DataFrame
- Exploratory visualizations:
  - Wins by driver (all-time top 20)
  - Wins by constructor
  - Grid position vs. finish position scatter
  - DNF causes over decades
  - Races per season trend

---

### Phase 2: Race Winner Prediction (Notebook 02) ⭐ Core Project

#### [NEW] [notebooks/02_race_winner_prediction.ipynb](file:///c:/Users/maiqmal/Desktop/proj/F1%20prediction/notebooks/02_race_winner_prediction.ipynb)

**Objective**: Predict the winner of a race before it starts.

**Target variable**: `is_winner` (binary: 1 if position == 1, else 0)

**Features** (engineered from raw data):
| Feature | Source | Description |
|---------|--------|-------------|
| `grid` | results | Starting grid position |
| `quali_position` | qualifying | Qualifying result |
| `driver_championship_pos` | driver_standings | Current championship position |
| `constructor_championship_pos` | constructor_standings | Constructor standing |
| `driver_recent_avg_pos` | results (rolling) | Avg. finish last 5 races |
| `driver_win_rate` | results (cumulative) | Career win % up to that race |
| `constructor_avg_points` | results (rolling) | Constructor's avg points last 5 races |
| `circuit_experience` | results (count) | No. of times driver raced here |
| `driver_circuit_avg_pos` | results (grouped) | Avg. finish at this specific circuit |
| `position_delta_avg` | results (rolling) | Avg. positions gained last 5 races |
| `dnf_rate` | results (cumulative) | % of races ending in DNF |
| `season_momentum` | results (rolling) | Points trend (going up or down) |

**Models to compare**:
1. Logistic Regression (baseline)
2. Random Forest
3. Gradient Boosting (sklearn)
4. XGBoost
5. LightGBM
6. CatBoost
7. Stacking Ensemble (combines the best 3)

**Evaluation**:
- Train/test split: Pre-2020 train, 2020+ test (time-based split — important!)
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Log Loss
- Feature importance (SHAP values — visual explanations)
- Confusion matrix

**Output**: For any race, output probability of each driver winning + predicted top 3.

---

### Phase 3: Driver Performance Analysis (Notebook 03)

#### [NEW] [notebooks/03_driver_performance_analysis.ipynb](file:///c:/Users/maiqmal/Desktop/proj/F1%20prediction/notebooks/03_driver_performance_analysis.ipynb)

**Objective**: Determine which drivers outperform their car.

**Analyses**:
1. **Position gain per race**: `grid - position` averaged per driver
2. **Teammate head-to-head**: Compare drivers in the same car (same constructor, same season)
3. **Consistency index**: Standard deviation of finishing positions (lower = more consistent)
4. **Performance radar charts**: Multi-axis chart per driver (wins, poles, podiums, consistency, position gain)
5. **Era-adjusted ranking**: Normalize performance relative to the grid size and competition level
6. **Position gain heatmaps**: Driver × Circuit heatmap showing avg. positions gained

**Visualizations**: Plotly interactive charts + Matplotlib static for portfolio.

---

### Phase 4: Strategy Optimization (Notebook 04)

#### [NEW] [notebooks/04_strategy_optimization.ipynb](file:///c:/Users/maiqmal/Desktop/proj/F1%20prediction/notebooks/04_strategy_optimization.ipynb)

**Objective**: Analyze pit stop strategies and optimal race strategies.

> [!IMPORTANT]
> Our dataset doesn't include direct pit stop data (lap-by-lap). We'll derive strategy insights from:
> - `laps` completed vs. race distance
> - `status` column (pit-stop-related retirements)
> - Position changes during races (approximated)
> - Time gaps between drivers

**Analyses**:
1. **Laps completed analysis**: Distribution of laps for different race outcomes
2. **Status-based strategy insights**: Correlating mechanical failures with race length
3. **Monte Carlo race simulation**: Simulate 10,000 race outcomes varying:
   - Grid position advantage (from historical data)
   - DNF probability per lap
   - Position-change probability
4. **Optimal starting position analysis**: What grid position gives the best expected finish?
5. **Statistical analysis**: Correlation between grid, points, and race duration

**Output**: Strategy recommendation summary with confidence intervals.

---

### Phase 5: Championship Prediction (Notebook 05)

#### [NEW] [notebooks/05_championship_prediction.ipynb](file:///c:/Users/maiqmal/Desktop/proj/F1%20prediction/notebooks/05_championship_prediction.ipynb)

**Objective**: After race X, predict final championship standings.

**Approach**:
1. Build cumulative points after each race in a season
2. For each race midpoint, predict the champion
3. **Methods**:
   - **XGBoost classifier**: Features = current points, gap to leader, avg points/race, races remaining
   - **ARIMA**: Time-series of cumulative points → forecast to season end
   - **LSTM**: Sequence of race results → predict remaining races
   - **Prophet**: Detect trend + seasonality in points accumulation
4. **Output**: After Race X → probability of each driver becoming champion + predicted final standings

**Evaluation**: Backtest on 2015–2025 seasons (predict mid-season, check against actual result).

---

### Phase 6: GOAT Ranking (Notebook 06)

#### [NEW] [notebooks/06_goat_ranking.ipynb](file:///c:/Users/maiqmal/Desktop/proj/F1%20prediction/notebooks/06_goat_ranking.ipynb)

**Objective**: Create an objective "Greatest of All Time" index.

**Raw metrics** (per driver):
- Total wins, poles, podiums, championships
- Win rate (wins / races entered)
- Podium rate, points per race

**Era adjustment**:
- Normalize by grid size (winning against 20 cars ≠ winning against 30)
- Weight championships by era competitiveness (how close was the title fight?)
- Adjust points to a common system (pre-2010 vs post-2010 scoring)

**GOAT Index formula** (weighted composite):
```
GOAT Score = 0.25 × Adjusted_Win_Rate 
           + 0.20 × Adjusted_Championship_Score
           + 0.15 × Adjusted_Podium_Rate 
           + 0.15 × Consistency_Index
           + 0.10 × Longevity_Factor
           + 0.10 × Position_Gain_Ability
           + 0.05 × Pole_Rate
```

**Techniques**: PCA to validate feature weights, sensitivity analysis.

---

### Phase 7: Driver Clustering (Notebook 07)

#### [NEW] [notebooks/07_driver_clustering.ipynb](file:///c:/Users/maiqmal/Desktop/proj/F1%20prediction/notebooks/07_driver_clustering.ipynb)

**Objective**: Identify driving archetypes using unsupervised learning.

**Features per driver** (career aggregates):
- Win rate, podium rate, DNF rate
- Avg. position gain (start to finish)
- Consistency (std. dev of positions)
- Qualifying strength (avg. grid position)
- Points per race

**Methods**:
1. **PCA**: Reduce to 2-3 components for visualization
2. **K-Means**: Test K=3 to K=8, use Elbow method + Silhouette score to pick best K
3. **DBSCAN**: Density-based clustering to find natural groups
4. **Hierarchical Clustering**: Dendrogram showing driver relationships

**Expected clusters** (to be validated by data):
- 🏆 **Dominators**: High win rate, low DNF, strong qualifying
- 📊 **Consistent Scorers**: Moderate wins, very low position variance
- ⚡ **Aggressive Racers**: High position gain, higher DNF rate
- 🐢 **Backmarkers**: Low points, high position numbers
- 🌟 **One-Hit Wonders**: Few races, occasional podiums

---

### Phase 8: Race Simulation Engine (Notebook 08)

#### [NEW] [notebooks/08_race_simulation.ipynb](file:///c:/Users/maiqmal/Desktop/proj/F1%20prediction/notebooks/08_race_simulation.ipynb)

**Objective**: Simulate an entire race using Monte Carlo methods.

**Simulation parameters** (derived from historical data):
- **Driver skill rating**: Based on career stats
- **Car performance**: Constructor's recent average points
- **Position change probability**: Per-lap probability of overtaking/being overtaken
- **DNF probability**: Historical reliability
- **Track-specific factors**: Some circuits allow more overtaking

**Process**:
1. Start with qualifying grid order
2. Simulate each lap:
   - Calculate position swap probabilities between adjacent drivers
   - Roll random numbers to determine if position changes occur
   - Check for DNFs (random with historical probability)
3. Run 10,000 simulations
4. Aggregate results → probability of each driver winning

**Output**: "Verstappen: 42%, Norris: 21%, Leclerc: 17%, ..."

---

### Phase 9: Constructor Success Prediction (Notebook 09)

#### [NEW] [notebooks/09_constructor_prediction.ipynb](file:///c:/Users/maiqmal/Desktop/proj/F1%20prediction/notebooks/09_constructor_prediction.ipynb)

**Objective**: Predict if a constructor finishes in top 3 of the championship.

**Target**: Binary — 1 if final position ≤ 3, else 0.

**Features**:
- Previous season's finishing position
- Number of wins in last 3 seasons
- Driver quality score (avg of their two drivers' stats)
- Historical success at upcoming circuits
- Budget proxy (derived from points/wins trends)

**Models**: Logistic Regression, Random Forest, XGBoost — compared on AUC.

---

### Phase 10: Interactive Streamlit Dashboard

#### [NEW] [dashboard/app.py](file:///c:/Users/maiqmal/Desktop/proj/F1%20prediction/dashboard/app.py)
Main multipage Streamlit app with dark F1-themed styling. Landing page with key stats.

#### [NEW] Dashboard pages (6 pages under `dashboard/pages/`)
Interactive versions of the notebook analyses:
1. **Race Prediction**: Select a race → see win probabilities
2. **Driver Analysis**: Pick a driver → see radar chart + stats
3. **Strategy**: Race simulation controls + results
4. **Championship**: Select a season + race round → see predicted standings
5. **GOAT Index**: Interactive ranking table with adjustable weights
6. **Clustering**: PCA scatter plot with hover-over driver info

---

## User Review Required

> [!IMPORTANT]
> **Pit stop data**: Your dataset does not include lap-by-lap or pit stop timing data. The Strategy Optimization module (Phase 4) will work with approximations from the available columns (`laps`, `status`, `time`). If you want full pit stop analysis, we can fetch additional data from the [Ergast API](http://ergast.com/mrd/) or [OpenF1 API](https://openf1.org/). **Should we add this?**

> [!IMPORTANT]
> **Weather data**: Your original spec mentioned weather as optional. We don't have weather data in the CSVs. We could scrape historical weather for race dates + locations from a weather API. This would significantly boost the Race Winner Prediction model. **Do you want to include this?**

> [!WARNING]
> **Python environment**: Python 3.13 is installed but `python` isn't on your PATH (we have to use the full path). Before we start, we should set up a virtual environment and install dependencies. I'll guide you through this.

## Open Questions

1. **Scope priority**: This is a large project (~9 notebooks + dashboard). Would you like me to build it **all at once** or in **phases** (e.g., start with Race Winner Prediction + Dashboard, then add modules)?

2. **Streamlit dashboard vs. static notebooks**: Do you want the Streamlit dashboard as a key deliverable, or are the Jupyter notebooks sufficient for your portfolio?

3. **Deep learning (LSTM)**: For Championship Prediction, LSTM requires TensorFlow/PyTorch (appears installed on your system). Do you want to include deep learning models, or keep it simpler with XGBoost + ARIMA?

4. **Additional data sources**: Beyond the Ergast/OpenF1 pit stop and weather questions above, would you like to incorporate:
   - Sprint race data
   - Fastest laps per lap (not just overall)
   - Free practice session data

## Verification Plan

### Automated Tests
- Each notebook will be executable end-to-end without errors
- Model performance benchmarks: ROC-AUC > 0.75 for race winner prediction
- Unit tests for `src/` modules: `python -m pytest tests/`

### Manual Verification
- Run `streamlit run dashboard/app.py` and verify all pages load
- Visually inspect all charts for correctness
- Compare GOAT rankings and cluster assignments against F1 domain knowledge
- Cross-validate predictions against known 2024–2025 race results
