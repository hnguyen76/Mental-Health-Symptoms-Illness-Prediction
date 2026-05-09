# Mental Health Symptoms & Illness Prediction Dashboard

Created by Hieu Nguyen

This repository contains a professional Streamlit dashboard for exploring a mental-health symptom dataset with 8,304 records, 22 condition labels, and 185 binary symptom features.

The dashboard is built as a recruiter-ready analytics portfolio project. It focuses on clean presentation, interactive exploration, condition-level profiling, symptom relationship analysis, and an explainable similarity-based prediction sandbox.

## Highlights

- Executive overview of condition distribution, symptom prevalence, and symptom load
- Condition profile deep dives with distinctive symptoms and co-occurrence heatmaps
- Symptom explorer for condition association and related symptom lift
- Prediction sandbox that ranks condition profiles from selected symptoms
- Standalone `clean_up.py` script for repeatable dataset cleanup and quality reporting
- Data quality page covering missing values, duplicates, balance, and schema readiness
- Clear responsible-use note: this is an analytics demo, not a clinical diagnosis tool

## Tech Stack

- Python
- Streamlit
- pandas
- Plotly
- NumPy

## Project Structure

```text
.
+-- app.py
+-- clean_up.py
+-- illness_dataset.csv
+-- requirements.txt
+-- .streamlit/
|   +-- config.toml
+-- README.md
```

## Data Cleanup

Run the cleanup script to normalize column names, coerce symptom features to binary values, add an active symptom count, and export a lightweight quality report.

```bash
python clean_up.py
```

Generated files:

- `data/processed/illness_dataset_clean.csv`
- `data/processed/data_quality_report.csv`

Optional duplicate removal:

```bash
python clean_up.py --drop-duplicates
```

## Run Locally

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py --server.port 8502
```

Then open the local Streamlit URL shown in the terminal.

## Responsible Use

This project is for analytics, education, and portfolio demonstration only. It is not designed to diagnose, treat, or replace care from a qualified medical or mental-health professional.
