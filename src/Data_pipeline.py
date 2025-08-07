import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

def load_data(file_path):
    return pd.read_csv(file_path)

def feature_engineering(df):
    df["SeniorCitizen"] = df["SeniorCitizen"].astype("category")

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors = 'coerce')
    df["ChargeTenure"] = df["MonthlyCharges"] * df["tenure"]

    return df

def get_feature_columns(df):
    categorical_features = ["SeniorCitizen","Partner","Dependents",
                            "MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
                            "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
                            "Contract","PaperlessBilling","PaymentMethod"
]
    numerical_features = ["MonthlyCharges","TotalCharges","ChargeTenure"]
    return categorical_features, numerical_features

def preprocessing_pipeline(categorical_features, numerical_features):
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore',sparse_output=False)),
    ])
    numerical_transformer = Pipeline(steps=[
        ('impute',SimpleImputer(strategy='median'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor



def savefile (X_train,y_train,X_test,y_test,output_path):
    os.makedirs(output_path,exist_ok=True)
    pd.DataFrame(X_train).to_csv(os.path.join(output_path, 'X_train.csv'),index=False)
    pd.DataFrame(y_train).to_csv(os.path.join(output_path, 'y_train.csv'), index=False)
    pd.DataFrame(X_test).to_csv(os.path.join(output_path, 'X_test.csv'), index=False)
    pd.DataFrame(y_test).to_csv(os.path.join(output_path, 'y_test.csv'), index=False)

def main(file_path = "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv",output_path="../data",test_size=0.2,random_state=42):
    df = load_data(file_path)
    df = feature_engineering(df)
    X=df.drop(["customerID","Churn","gender","PhoneService"],axis=1)
    y = df["Churn"].map({'Yes':1,'No':0})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    categorical_features, numerical_features = get_feature_columns(X_train)
    preprocessor = preprocessing_pipeline(categorical_features, numerical_features)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    cat_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
    feature_names = numerical_features +list(cat_feature_names)

    X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names)
    y_train_processed = pd.Series(y_train,name="Churn")
    y_test_processed = pd.Series(y_test,name="Churn")

    savefile(X_train_processed,y_train_processed,X_test_processed,y_test_processed,output_path)

    return X_train_processed, y_train_processed, X_test_processed, y_test_processed

if __name__ == "__main__":
    X_train_processed, y_train_processed, X_test_processed, y_test_processed = main()
    print("preprocessing Complete,data is saved")


