# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Model Serving Endpoint
# MAGIC Once the model is registered, we can use API to create a Databricks GPU Model Serving Endpoint that serves the `LLaMAV2-7b` model.
# MAGIC

# COMMAND ----------

# Provide a name to the serving endpoint
endpoint_name = "llama2-7b-chat"

# COMMAND ----------

databricks_url = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .apiUrl()
    .getOrElse(None)
)
token = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .apiToken()
    .getOrElse(None)
)

# COMMAND ----------

import json

import requests

deploy_headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}
deploy_url = f"{databricks_url}/api/2.0/serving-endpoints"

model_version = result  # the returned result of mlflow.register_model
endpoint_config = {
    "name": endpoint_name,
    "config": {
        "served_models": [
            {
                "name": f'{model_version.name.replace(".", "_")}_{model_version.version}',
                "model_name": model_version.name,
                "model_version": model_version.version,
                "workload_type": "GPU_MEDIUM",
                "workload_size": "Small",
                "scale_to_zero_enabled": "False",
            }
        ]
    },
}
endpoint_json = json.dumps(endpoint_config, indent="  ")

# Send a POST request to the API
deploy_response = requests.request(
    method="POST", headers=deploy_headers, url=deploy_url, data=endpoint_json
)

if deploy_response.status_code != 200:
    raise Exception(
        f"Request failed with status {deploy_response.status_code}, {deploy_response.text}"
    )

# Show the response of the POST request
# When first creating the serving endpoint, it should show that the state 'ready' is 'NOT_READY'
# You can check the status on the Databricks model serving endpoint page, it is expected to take ~35 min for the serving endpoint to become ready
print(deploy_response.json())

# COMMAND ----------

# MAGIC %md
# MAGIC Once the model serving endpoint is ready, you can query it easily with LangChain (see `04_langchain` for example code) running in the same workspace.
