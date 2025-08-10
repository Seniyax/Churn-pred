# Churn Prediction API
This project implements a customer churn prediction API using a LightGBM model, built with FastAPI and containerized with Docker. It predicts whether a customer will churn based on the Telco Customer Churn dataset, achieving a ROC-AUC of 0.860 and recall of 0.531. The project includes preprocessing, model training, API serving, and unit tests, with MLOps integration via MLflow for logging and DVC for versioning.
## Project Structure
````
Churn-pred/
├──data/
│    └──WA_Fn-UseC_-Telco-Customer-Churn.csv  #data
├──models/
│     └──lightgbm.pkl.dvc                      # versioned model
├──notebooks/
│      └──eda.ipynb     # EDA
├──src/
│     ├──mlruns   #MLflow loggings
│     ├──Data-pipeline.py # Data preprocessing
│     ├──api.py  # Fast API
│     ├──train.py # Model training script
├──tests/
│     └── api_test.py # Unit test    
├──Dockerfile # Docker configuration for the api
├──docker-compose.yml # Orchestration for MLflow and API
├──requirements.txt # project dependencies
````
## Features
 - **Model** - Lightgbm Classifier (ROC-AUC:0.860,Precision:0.678)
 - **API Endpoints** - GET / : Health check, POST/Predict : prediction for a single person
 - **MLOps** - MLflow for logging predictions and errors, DVC for model versioning
 - **Docker** - Containerized API for easy deployment.
## Setup & Installation
### 1. Clone the Repository
```` bash
git clone <Seniyax/Churn-pred>
cd Churn pred
````
### 2. Setup Conda enviroment
```` bash
conda create -n Churn-pred python=3.10
conda activate Churn-pred
pip install -r requirements.txt
````
### 3. Create MLflow directory
```` bash
mkdir mlflow
````
### 4. Build and run the service
```` bash
docker-compose up -build
````
### 5. Access APIs
API:http://localhost:8000/

MLflow UI:http:/localhost:5000/
### 6. Test API
- Run the Unit test
## Future Improvements
- Evaluate alternative models eg: Xgboost
  
       
    
