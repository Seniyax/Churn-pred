import pickle
from contextlib import asynccontextmanager
from typing import cast
from sklearn.compose import ColumnTransformer

from fastapi import FastAPI,HTTPException
import pandas as pd
from mlflow.utils.cli_args import MODEL_PATH



from src.Data_pipeline import  feature_engineering,get_feature_columns,preprocessing_pipeline
from pydantic import BaseModel
import joblib
import mlflow
import os
from typing import List





ASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(ASE_DIR,".." ,"models","lightgbm.pkl")
with open(model_path,"rb") as f:
    model = joblib.load(f)

preprocessor = None

class CustomerData(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str

class BatchCustomerData(BaseModel):
    customers: List[CustomerData]

@asynccontextmanager
async def lifespan(app : FastAPI):
    global preprocessor,model


    #mlflow.set_experiment("Customer Churn Prediction API")
    data_path = os.path.join(ASE_DIR, "..", "data", "X_train.csv")
    sample_data = pd.read_csv(data_path)
    preprocessor = None
    app.state.encoded_columns = sample_data.columns.tolist()
    print("[INFO] Data already encoded — skipping preprocessor fitting")

    #categorical_features,numerical_features = get_feature_columns(sample_data)
    #preprocessor = preprocessing_pipeline(categorical_features,numerical_features)
    #preprocessor.fit(sample_data)
    yield


app = FastAPI(
    title="Churn Prediction API",
    version="1.0",
    lifespan=lifespan
)


@app.get("/")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(data:CustomerData):
    if preprocessor is None:
        raise HTTPException(status_code=500, detail="Preprocessor not initialized")
    try:



        with mlflow.start_run(run_name="Customer Churn Prediction API"):
            input_df = pd.DataFrame([data.model_dump()])

            mlflow.log_param("customerID",data.customerID)

            input_df = feature_engineering(input_df)

            #X = input_df.drop(columns=["customerID","gender","PhoneService"],errors="ignore")

            #X_processed = cast(ColumnTransformer, preprocessor).transform(X)
            encoded_cols = app.state.encoded_columns
            X_processed = input_df.reindex(columns=encoded_cols, fill_value=0)

            proba = model.predict_proba(X_processed)[:, 1][0]
            prediction =int (model.predict(X_processed)[0])

            mlflow.log_metric("prediction",prediction)

            return {
                "customerID": data.customerID,
                "churn_prediction": prediction,
                "churn_probability": float(proba)
              }






    except Exception as e:
        if mlflow.active_run():
            mlflow.log_param("error", str(e))

        raise HTTPException(status_code=500, detail=f"prediction error:{str(e)}")





