# Thyroid Disease Predictor

A machine learning web application that predicts whether a patient may have thyroid disease using clinical input features.  
The app compares predictions from two models:

- Random Forest Classifier
- XGBoost Classifier

## Live Demo

Try the deployed app here:

https://thyroid-predictor-chtqqyovezoxk3atnlb38s.streamlit.app/

## GitHub Repository

https://github.com/AbdullahKhalid166/Thyroid-Predictor

## Project Overview

This project uses thyroid-related medical data to train machine learning models for disease prediction.  
The trained models are deployed using Streamlit, allowing users to enter patient information and get real-time prediction results with probability scores.

## Features

- User-friendly Streamlit web interface
- Random Forest prediction
- XGBoost prediction
- Disease probability output
- Handles missing numerical values
- Uses preprocessing pipeline with imputation and encoding
- Deployed online using Streamlit Cloud

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Joblib
- Matplotlib
- Seaborn

## Project Structure

```text
Thyroid-Predictor/
│
├── app.py
├── requirements.txt
├── runtime.txt
│
├── models/
│   ├── thyroid_model.pkl
│   └── xgb_model.pkl
│
├── data/
│   └── thyroidDF.csv
│
├── notebooks/
│   └── thyroidproject.ipynb
│
└── README.md