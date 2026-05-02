# Sales Prediction System

## Overview

The Sales Prediction System is a machine learning-based application
designed to predict profit based on business expenditures such as R&D
Spend, Administration, and Marketing Spend. The system provides both
real-time and batch predictions through an interactive web interface
built with Streamlit.

This project demonstrates a complete end-to-end machine learning
pipeline, including data validation, feature engineering, model
training, evaluation, and deployment.

------------------------------------------------------------------------

## Features

-   Real-time profit prediction
-   Batch prediction using CSV upload
-   Model performance visualization
-   What-if analysis using interactive charts
-   Automatic model training on first run
-   Model versioning and logging

------------------------------------------------------------------------

## Project Structure

    Sales Prediction System/
    │
    ├── data/                # Dataset files
    ├── models/              # Saved trained models
    ├── logs/                # Application logs
    ├── src/                 # Core logic
    │   ├── model_trainer.py
    │   ├── predictor.py
    │   ├── feature_engineering.py
    │   ├── data_validator.py
    │   ├── sales_prediction.py
    │
    ├── ui/                  # Streamlit UI
    ├── config.yaml          # Configuration file
    ├── requirements.txt     # Dependencies
    ├── execute.txt          # Execution instructions
    └── README.md            # Project documentation

------------------------------------------------------------------------

## Requirements

-   Python 3.8 or higher
-   pip (Python package manager)

------------------------------------------------------------------------

## Libraries Used

-   pandas
-   numpy
-   scikit-learn
-   xgboost
-   joblib
-   streamlit
-   plotly
-   pyyaml

------------------------------------------------------------------------

## Installation and Setup

### 1. Clone or Download the Project

Download the project and extract it.

------------------------------------------------------------------------

### 2. Open Terminal

Navigate to the project folder:

    cd path/to/Sales Prediction System

------------------------------------------------------------------------

### 3. Create Virtual Environment (Optional)

    python -m venv venv

Activate it:

Windows:

    venv\Scripts\activate

Mac/Linux:

    source venv/bin/activate

------------------------------------------------------------------------

### 4. Install Dependencies

    pip install -r requirements.txt

------------------------------------------------------------------------

## Running the Application

Run the following command:

    streamlit run ui/sales_prediction_ui.py

------------------------------------------------------------------------

## Access the Application

Open your browser and go to:

http://localhost:8501

------------------------------------------------------------------------

## How to Use

### 1. Single Prediction

Enter values for: - R&D Spend - Administration Spend - Marketing Spend

Click predict to get profit estimation.

------------------------------------------------------------------------

### 2. Batch Prediction

Upload a CSV file with required columns to predict multiple records.

------------------------------------------------------------------------

### 3. Model Performance

View evaluation metrics such as R2 Score, MAE, and RMSE.

------------------------------------------------------------------------

### 4. What-If Analysis

Adjust values dynamically and observe prediction changes using charts.

------------------------------------------------------------------------

## Model Training

-   Automatically triggered if no saved model exists
-   Uses XGBoost Regressor
-   Applies feature engineering
-   Evaluates model performance
-   Saves model to models/ directory

------------------------------------------------------------------------

## Logs

All logs are stored in:

    logs/sales_prediction.log

------------------------------------------------------------------------

## Models

Saved in:

    models/

Includes: - trained_model.pkl - versioned models with timestamp

------------------------------------------------------------------------

## Troubleshooting

### Streamlit not found

    pip install streamlit

### Port already in use

    streamlit run ui/sales_prediction_ui.py --server.port 8502

### Dependency issues

    pip install --upgrade pip
    pip install -r requirements.txt

------------------------------------------------------------------------

## Notes

-   Ensure dataset is available in the data folder
-   Do not include virtual environment in project distribution
-   Recommended to retrain model with larger datasets for better
    performance

------------------------------------------------------------------------

## License

This project is for educational purposes.
