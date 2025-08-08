import pandas as pd
import numpy as np
import os

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
import optuna
import joblib
import mlflow
import mlflow.sklearn



# -------------------------
# Data Loader
# -------------------------
def load_processed_data(input_dir="../data"):
    X_train = pd.read_csv(os.path.join(input_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv")).squeeze()
    X_test = pd.read_csv(os.path.join(input_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(input_dir, "y_test.csv")).squeeze()
    return X_train, y_train, X_test, y_test


# -------------------------
# Optuna Objective
# -------------------------
def objective_initialization(model_type, X_train, y_train, X_val, y_val):
    def objective(trial):
        if model_type == "xgboost":
            params = {
                'max_depth': trial.suggest_int("max_depth", 2, 6),
                'learning_rate': trial.suggest_float("learning_rate", 0.01, 1.0),
                'n_estimators': trial.suggest_int("n_estimators", 50, 300),
                'subsample': trial.suggest_float("subsample", 0.5, 0.9),
                'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 0.9),
                'min_child_weight': trial.suggest_float("min_child_weight", 1, 10),
                'alpha': trial.suggest_float("alpha", 0.01, 1.0),
                'lambda': trial.suggest_float("lambda", 0.01, 1.0),
                'random_state': 42,
                'eval_metric': 'auc'
            }
            model = xgb.XGBClassifier(**params)



        elif model_type == "lightgbm":
            params = {
                'max_depth': trial.suggest_int("max_depth", 2, 6),
                'learning_rate': trial.suggest_float("learning_rate", 0.01, 1.0),
                'n_estimators': trial.suggest_int("n_estimators", 50, 300),
                'subsample': trial.suggest_float("subsample", 0.5, 0.9),
                'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 0.9),
                'min_child_weight': trial.suggest_float("min_child_weight", 1, 10),
                'reg_alpha': trial.suggest_float("reg_alpha", 0, 1.0),
                'reg_lambda': trial.suggest_float("reg_lambda", 0, 1.0),
                'random_state': 42,
                'metric': 'auc'
            }
            model = lgb.LGBMClassifier(**params)

        else:
            raise ValueError("Invalid model type")

        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        auks = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model.fit(X_tr, y_tr)
            y_pred_proba = model.predict_proba(X_va)[:, 1]
            auc = roc_auc_score(y_va, y_pred_proba)
            auks.append(auc)

        return np.mean(auks)

    return objective


# -------------------------
# Hyperparameter Tuning
# -------------------------
def tune_model(model_type, X_train, y_train, n_trials=10):
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective_initialization(model_type, X_train_sub, y_train_sub, X_val, y_val),
        n_trials=n_trials
    )
    return study.best_params


# -------------------------
# Model Training
# -------------------------
def train_model(model_type, X_train, y_train, params, X_val, y_val):
    if model_type == "xgboost":
        model = xgb.XGBClassifier(**params, eval_metric='auc')
        fit_kwargs = {
            'eval_set': [(X_val, y_val)] if X_val is not None else None,
            'verbose': False
        }
    elif model_type == "lightgbm":
        model = lgb.LGBMClassifier(**params, metric='auc')
        fit_kwargs = {
            'eval_set': [(X_val, y_val)] if X_val is not None else None,
            'callbacks': [lgb.early_stopping(stopping_rounds=10, verbose=False)]
        }
    else:
        raise ValueError("Invalid model type")

    model.fit(X_train, y_train, **{k: v for k, v in fit_kwargs.items() if v is not None})
    return model


# -------------------------
# Evaluation
# -------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
    }
    return metrics


# -------------------------
# Save Model
# -------------------------
def save_model(model_type, model, output_dir="../models"):
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model_type}.pkl")
    joblib.dump(model, model_path)
    return model_path


# -------------------------
# Main Pipeline
# -------------------------
def main(data_dir="../data", model_dir="../models", n_trials=10, random_state=42):
    np.random.seed(random_state)

  
    X_train, y_train, X_test, y_test = load_processed_data(input_dir=data_dir)
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state
    )

    model_types = ["xgboost", "lightgbm"]
    mlflow.set_experiment("Churn_Prediction")

    for model_type in model_types:
        with mlflow.start_run(run_name=f"{model_type}_run"):
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("n_trials", n_trials)

            
            best_params = tune_model(model_type, X_train, y_train, n_trials=n_trials)
            mlflow.log_params(best_params)

           
            best_params['random_state'] = random_state

         
            model = train_model(model_type, X_train, y_train, best_params, X_val, y_val)

           
            metrics = evaluate_model(model, X_test, y_test)
            mlflow.log_metrics(metrics)

            model_path = save_model(model_type, model, model_dir)
            mlflow.log_artifact(model_path)

            print(f"{model_type.capitalize()} - Metrics: {metrics}")
            print(f"Model saved to: {model_path}")

    return model_types


# -------------------------
# Run Script
# -------------------------
if __name__ == "__main__":
    model_types = main()
    print(f"Training completed: {model_types}")
