# README — DSAI3202 Football Player Performance and Market Value project phase 1.

## Bronze → Silver → Gold + Feature Engineering
This project implements a complete data engineering and feature engineering workflow for football player performance and market value analysis. Our group followed the Lakehouse architecture model and produced three data layers: Bronze, Silver, and Gold, along with a full feature engineering stage to prepare the dataset for machine learning.

Below we describe, step by step, what we built and why.


1. Bronze Layer — Raw Data

The Bronze layer stores the raw data exactly as received, without any transformation.  
We uploaded all source datasets into Azure Data Lake Storage, including:

- Players  
- Games  
- Appearances  
- Player_valuations  
- Clubs
- Club_games
- game_events
- game_lineups
- Competitions
- transfers

The purpose of this layer is to preserve the original data, maintain traceability, and allow reproducibility.  
No cleaning or processing occurs at this stage.

2. Silver Layer — Cleaned Parquet Data

In the Silver layer, we cleaned and standardized the raw data. This included:

- Casting numeric fields to the correct types  
- Converting date fields to proper date formats  
- Removing rows with missing or invalid IDs  
- Dropping corrupted or duplicated rows  
- Enforcing consistent schema and column types  
- Standardizing text formats  

All Silver datasets were saved in Parquet format inside the `/processed/` directory.  
This layer represents validated, structured data that is ready for analytical processing.


3. Gold Layer (Part 1) — Curated Player-Season Table

Using the Silver data, we created a curated analytical dataset where each row represents a player's performance for a specific season.

The table integrates information from:

- Appearances  
- Games  
- Player   
- Club   
- Seasonal market value  
- Performance metrics  

This curated dataset is named:

`player_season_value_features`

It is the foundation for the feature engineering stage.

4. Gold Layer (Part 2) — Feature Engineering

Because our dataset is structured rather than text-based, we adapted the feature extraction methodology to the football domain.  
We engineered a comprehensive set of features that capture performance, age dynamics, positional characteristics, and market value behavior.

4.1 Performance-Based Features

We created normalized and derived performance metrics including:

- `goals_per90`  
- `assists_per90`  
- `cards_per90`  
- `goal_contribution_per90`  
- `goals_per_match`  
- `assists_per_match`  
- `cards_per_match`  
- `minutes_per_match`  
- `discipline_index`

These offer a standardized representation of performance rather than relying only on raw totals.

4.2 Age-Related Features

Since age significantly influences player value, we engineered:

- `age_squared`  
- `is_under_21`  
- `is_over_30`  

These help capture non-linear age effects and identify youth/late-career categories.

4.3 Market Value Transformations

Market value is highly skewed, so we derived:

- `market_value_millions`  
- `log_market_value`  

The log transformation reduces skew and helps models learn value patterns more effectively.

4.4 Categorical Encoding

We encoded the following fields using StringIndexer and OneHotEncoder:

- `position`  
- `sub_position`

This converts categorical attributes into numerical vectors required by ML algorithms.

4.5 Scaling

All engineered numeric features were assembled into a vector and scaled using StandardScaler.  
This normalizes ranges and ensures compatibility with models sensitive to scale.

The final engineered dataset is produced as:

`player_season_features`

## Why We Performed Numerical Feature Engineering Instead of Text-Based Extraction

The feature extraction methods (such as TF-IDF, n-grams, and transformer-based embeddings) are designed for unstructured text data. Our dataset, however, consists of structured football performance tables containing numerical statistics (minutes played, goals, assists, cards, age, club information, etc.) instead of textual reviews. Because of this, traditional text-focused techniques are not applicable.

Instead, we implemented domain-specific numerical feature engineering tailored to football analytics. This approach allowed us to derive meaningful performance indicators—such as per-90 metrics, age-based transformations, positional encodings, and market value adjustments—that are more relevant and appropriate for modeling player performance and valuation.

5. Validation and Analysis

To validate the engineered features, we performed correlation analysis between goal contribution metrics and market value.

The overall global correlation was low, which is expected because market value depends on many external factors.  
To investigate further, we calculated correlations by position:

- Defenders: near zero  
- Goalkeepers: near zero  
- Midfielders: low  
- Attackers: highest correlation relative to others  
- Missing categories: higher values due to data artifacts  

These results match real football trends and confirm that the engineered features behave realistically.

6. Final Outputs

The project produces the following:

1. **Bronze Layer** — Raw football datasets  
2. **Silver Layer** — Cleaned Parquet data stored under `/processed/`  
3. **Gold Layer: Curated** — `player_season_value_features`  
4. **Gold Layer: Engineered** — `player_season_features`  

All final outputs are saved as Delta tables and ready for machine learning.

7. Summary of Work Completed

As a group, we accomplished:

- Raw ingestion into the Bronze layer  
- Full cleaning and standardization for the Silver layer  
- Construction of a curated analytical table in the Gold layer  
- Comprehensive feature engineering tailored to the football domain  
- Detailed validation through correlation analysis  
- Creation of a complete machine-learning-ready dataset  

# README — DSAI3202 Football Player Performance and Market Value project phase 2.
Phase 2 — MLOps Pipeline (Feature Store + Model Training Pipeline)
In Phase 2, our group extended the project into an MLOps workflow following Lab 5 requirements.
The focus was on automation, modularization, and operationalization of the machine learning lifecycle.

Below is what we implemented.

1. Feature Store Integration (Gold Layer Source)
We reused Phase 1 engineered datasets (player_season_features and player_season_value_features) as inputs for automated ML processing.

These were registered as a reusable Data Asset in Azure ML:

✔ playeranalytics_DataAsset
— this became the pipeline input source.

2. Component-Based Architecture
We developed an MLOps pipeline based on three independent modular components, each packaged as:

✔ Python script
✔ YAML specification
✔ Container environment + inputs/outputs definition

This allows traceability, reuse, version control and reproducibility.

Component A — Feature Retrieval
📌 Purpose
Pull curated feature datasets from Gold Layer, merge them, split train/test and save outputs.

📌 Outputs

train.parquet

test.parquet

📌 What it does

✔ Reads Gold data
✔ Joins performance + value tables
✔ Performs 80/20 train–test split
✔ Saves outputs for downstream stages

Component B — Feature Selection (Baseline Method)
📌 Purpose
Automatically select informative features prior to training.

📌 What it does

✔ Reads training parquet
✔ Applies VarianceThreshold feature filtering
✔ Stores list of selected features in .json output

📌 Output

selected_features.json

This selection is reused by component C to train using only relevant features.

Component C — Model Training + Evaluation
📌 Purpose
Train a supervised ML model inventory and evaluate it.

📌 What it does

✔ Loads train/test data
✔ Reads selected feature list from Component B
✔ Trains a Random Forest model
✔ Predicts on test data
✔ Calculates RMSE
✔ Saves model artifact + metrics

📌 Outputs

model.pkl

metrics.json

We later registered this model in Azure ML.

3. Azure ML Pipeline Assembly
All components were orchestrated using an Azure ML pipeline script:

➡ pipeline_job.py

The pipeline:

Calls feature retrieval

Passes outputs to feature selection

Feeds selected features + datasets into training component

Returns the trained model and evaluation metrics

✔ We set default compute = compute1
✔ We executed the pipeline through Azure ML Job submission

Output appeared in Azure ML Studio under Jobs.

4. Model Registration
Once training completed successfully:

✔ Model artifact was stored
✔ Metadata included feature selection list + version reference

This enables future deployment or retraining using lineage traceability.

5. Repository & Folder Structure
We created a production-style project structure:

player_mlops/
│
├── components/
│   ├── feature_retrieval.py
│   ├── feature_retrieval.yml
│   ├── feature_selection.py
│   ├── feature_selection.yml
│   ├── train_eval.py
│   └── train_eval.yml
│
├── pipeline_job.py   ← Pipeline definition & submission
└── README.md         ← Documentation
This follows MLOps best practices for modularity and maintainability.

6. What We Achieved in Phase 2
✔ Built automated model workflow
✔ Applied structured feature selection
✔ Enabled reproducible ML training
✔ Registered model artifacts for deployment
✔ Used Azure ML Pipelines and Compute Cluster execution
✔ Created reusable components for retraining or scheduling
