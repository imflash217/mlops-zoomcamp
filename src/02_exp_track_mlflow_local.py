from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import os
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

load_dotenv("../.env")  # TODO: fix this

###################################################################

TRACKING_URI = os.getenv("TRACKING_URI")
DATA_ROOT_PATH = os.getenv("DATA_ROOT_PATH")
ARTIFACT_PATH = os.getenv("ARTIFACT_PATH")
EXPERIMENT_NAME = "zoomcamp_mlflow_exp_01"
###################################################################

# print(f"Tracking URI: [{mlflow.get_tracking_uri()}]")
# print(DATA_ROOT_PATH)

print(mlflow.set_tracking_uri(TRACKING_URI))  # TODO: fix this
print(f"Tracking URI: [{mlflow.get_tracking_uri()}]")

# experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
# if not experiment:
#     experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
#     print(experiment_id)
# else:
#     print(experiment.experiment_id)

###################################################################
experiment = mlflow.set_experiment(EXPERIMENT_NAME)
with mlflow.start_run():
    X, y = load_iris(return_X_y=True)
    params = {
        "C": 0.1,
        "random_state": 217,
    }
    mlflow.log_params(params)

    log_reg = LogisticRegression(**params).fit(X, y)
    y_pred = log_reg.predict(X)

    metrics = {"accuracy": accuracy_score(y_true=y, y_pred=y_pred)}

    mlflow.log_metrics(metrics)

    mlflow.sklearn.log_model(log_reg, artifact_path=ARTIFACT_PATH)
    print(f"Default artifacts_uri = [{mlflow.get_artifact_uri()}]")

client = MlflowClient()

try:
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    run_id = runs[0].info.run_id
    print(f"run_id = {run_id}")

    mlflow.register_model(model_uri=f"runs:/{run_id}/models", name="iris-classifier")
except MlflowException:
    print("Not possible to access mlflow registry")
