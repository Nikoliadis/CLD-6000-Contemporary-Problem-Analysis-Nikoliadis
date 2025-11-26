# Employee Attrition Prediction System

This repository contains the experiments and implementation for predicting employee attrition using machine learning (Decision Trees and other models).

## Project Structure

- `data/`: HR dataset (e.g. hr_data.csv)
- `src/`:
  - `data_preprocessing.py`: functions for loading and preprocessing the dataset
  - `train_models.py`: train multiple ML models
  - `evaluate_models.py`: evaluation utilities
- `notebooks/`: exploratory analysis and visualisations
- `models/`: trained model files (`.pkl`)
- `diagrams/`: design diagrams used in the report
- `requirements.txt`: Python dependencies

## How to run

```bash
pip install -r requirements.txt
python src/pipeline.py
