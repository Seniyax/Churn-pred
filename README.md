# Churn Prediction API
This project implements a customer churn prediction API using a LightGBM model, built with FastAPI and containerized with Docker. It predicts whether a customer will churn based on the Telco Customer Churn dataset, achieving a ROC-AUC of 0.860 and recall of 0.531. The project includes preprocessing, model training, API serving, and unit tests, with MLOps integration via MLflow for logging and DVC for versioning.
## Project Structure
````
Churn-pred/
|
