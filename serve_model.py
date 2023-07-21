import os

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput
from mlflow import MlflowClient

os.environ["DATABRICKS_TOKEN"] = dbutils.secrets.get(
    scope="jtisbell", key="db-token"
)
os.environ["DATABRICKS_HOST"] = "https://e2-demo-field-eng.cloud.databricks.com"

w = WorkspaceClient()
mlflow_client = MlflowClient()

# TODO: make dynamic
model_name = "llamav2_7b_chat_model"
endpoint_name = "llama2-7b-chat"

model_version = mlflow_client.get_latest_versions(
    name=model_name, stages=["Staging"]
)[0]

endpoint_config = (
    {
        "name": endpoint_name,
        "served_models": [
            {
                "name": f'{model_version.name.replace(".", "_")}_{model_version.version}',
                "model_name": model_version.name,
                "model_version": model_version.version,
                "workload_type": "GPU_MEDIUM",
                "workload_size": "Small",
                "scale_to_zero_enabled": "True",
            }
        ],
    },
)

w.serving_endpoints.create(
    name=endpoint_name,
    config=EndpointCoreConfigInput.from_dict(endpoint_config),
)
