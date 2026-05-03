import warnings
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

warnings.filterwarnings("ignore")
np.random.seed(42)


# Evaluation Metrics Function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


# Load Dataset
data_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
data = pd.read_csv(data_url, sep=";")

train, test = train_test_split(data, test_size=0.25, random_state=42)

train_x = train.drop("quality", axis=1)
test_x = test.drop("quality", axis=1)
train_y = train["quality"]
test_y = test["quality"]


# Hyperparameters
alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5



# MLflow Experiment
mlflow.set_experiment("Wine-Quality-Experiment")

with mlflow.start_run():

    # Model
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    model.fit(train_x, train_y)

    predictions = model.predict(test_x)

    # Metrics
    rmse, mae, r2 = eval_metrics(test_y, predictions)

    print(f"ElasticNet(alpha={alpha}, l1_ratio={l1_ratio})")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2: {r2}")

    # Log params + metrics
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # Signature (important for deployment)
    signature = infer_signature(train_x, predictions)

    # Log model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=train_x.head(2)
    )