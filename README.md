# Mental Health Symptoms & Illness Prediction Dashboard

Created by Hieu Nguyen

This repository contains a professional Streamlit dashboard for exploring a mental-health symptom dataset with 8,304 records, 22 condition labels, and 185 binary symptom features.

The dashboard is built as a recruiter-ready analytics portfolio project. It focuses on clean presentation, interactive exploration, condition-level profiling, symptom relationship analysis, and an explainable similarity-based prediction sandbox.

## Highlights

- Executive overview of condition distribution, symptom prevalence, and symptom load
- Condition profile deep dives with distinctive symptoms and co-occurrence heatmaps
- Symptom explorer for condition association and related symptom lift
- Prediction sandbox that ranks condition profiles from selected symptoms
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
├── app.py
├── illness_dataset.csv
├── requirements.txt
└── README.md
```

## Run Locally

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Responsible Use

This project is for analytics, education, and portfolio demonstration only. It is not designed to diagnose, treat, or replace care from a qualified medical or mental-health professional.
