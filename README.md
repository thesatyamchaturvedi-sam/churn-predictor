# Customer Churn Predictor

A web app that predicts whether a telecom customer will churn, built with a scikit-learn pipeline (imputation, scaling, one-hot encoding, Logistic Regression) and deployed with Streamlit.

**Live demo:** <add your Streamlit link here>

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Model
- Dataset: Telco Customer Churn
- Pipeline: ColumnTransformer + Logistic Regression
- Metrics: <add your accuracy / precision / recall / ROC-AUC here>

## Files
- `app.py` Streamlit app
- `churn_model.pkl` trained pipeline
- `requirements.txt` pinned dependencies (scikit-learn 1.6.1 matches the training version)
