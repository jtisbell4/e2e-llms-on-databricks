# Databricks notebook source
# MAGIC %md
# MAGIC # Optimized LLM Serving Example
# MAGIC
# MAGIC Optimized LLM Serving enables you to take state of the art OSS LLMs and deploy them on Databricks Model Serving with automatic optimizations for improved latency and throughput on GPUs. Currently, we support optimizing the Mosaic MPT-7B model and will continue introducing more models with optimization support.
# MAGIC
# MAGIC This example walks through:
# MAGIC
# MAGIC 1. Downloading the `mosaicml/mpt-7b` model from huggingface transformers
# MAGIC 2. Logging the model in an optimized serving supported format into the Databricks Model Registry
# MAGIC 3. Enabling optimized serving on the model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prerequisites
# MAGIC * Attach a cluster to the notebook with sufficient memory to load MPT-7B. We recommend a cluster with at least 32 GB of memory.
# MAGIC * (Optional) Install the latest transformers. MPT-7B native support in transformers was added on July 25, 2023. At the time of this notebook release, MPT-7B native support in transformers has not been officially released. For full compatibility of MPT-7B with mlflow, install the latest version from github. Optimized serving will work with older versions of transformers for MPT-7B, but there may be issues with loading the model locally.
# MAGIC
# MAGIC To install the latest version of transformers off github, run:
# MAGIC ```
# MAGIC %pip install git+https://github.com/huggingface/transformers@main
# MAGIC ```
# MAGIC
# MAGIC

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# %pip install -U transformers==4.29.1
%pip install git+https://github.com/huggingface/transformers@main
%pip install -U mlflow>=2.5.0
%pip install -U accelerate>=0.20.3
dbutils.library.restartPython()

# COMMAND ----------

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

name = 'mosaicml/mpt-7b-chat'
# If you are using the latest version of transformers that has native MPT support, replace the following line with:
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)

# model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)

# COMMAND ----------

type(model)

# COMMAND ----------

model.device

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained(name)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Logging required metadata for optimized serving
# MAGIC
# MAGIC To enable optimized serving, when logging the model, include the extra metadata dictionary when calling `mlflow.transformers.log_model` as shown below:
# MAGIC
# MAGIC ```
# MAGIC metadata = {"task": "llm/v1/completions"}
# MAGIC ```
# MAGIC This specifies the API signature used for the model serving endpoint.
# MAGIC

# COMMAND ----------

registered_model_name = "optimized-mpt-7b-chat"

# COMMAND ----------

import mlflow
import numpy as np

with mlflow.start_run():
    components = {
            "model": model,
            "tokenizer": tokenizer,
        }
    mlflow.transformers.log_model(
        transformers_model=components,
        artifact_path="mpt",
        registered_model_name=registered_model_name,
        input_example={
            "prompt": np.array(
                [
                    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat is Apache Spark?\n\n### Response:\n"
                ]
            ),
            "max_tokens": np.array([75]),
            "temperature": np.array([0.0]),
        },
        metadata={"task": "llm/v1/completions"},
        await_registration_for=480,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure and create your model serving GPU endpoint
# MAGIC
# MAGIC Modify the cell below to change the endpoint name. After calling the create endpoint API, the logged MPT-7B model will automatically be deployed with optimized LLM Serving!

# COMMAND ----------

model_version = 5
served_model_workload_size = "Medium"
served_model_scale_to_zero = False

API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

import json
import requests

data = {
    "name": registered_model_name,
    "config": {
        "served_models": [
            {
                "model_name": registered_model_name,
                "model_version": model_version,
                "workload_size": served_model_workload_size,
                "scale_to_zero_enabled": False,
                "workload_type": "GPU_MEDIUM"
            }
        ]
    }
}

headers = {
    "Context-Type": "text/json",
    "Authorization": f"Bearer {API_TOKEN}"
}

response = requests.post(
    url=f"{API_ROOT}/api/2.0/preview/serving-endpoints",
    json=data,
    headers=headers
)

print(json.dumps(response.json(), indent=4))


# COMMAND ----------

# MAGIC %md
# MAGIC ## View your endpoint!
# MAGIC To see your more information about your endpoint, go to the "Serving" section on the left navigation bar and search for your endpoint name.

# COMMAND ----------

# MAGIC %md
# MAGIC * MPT has Triton flash attention. When is it appropriate to use it?
# MAGIC * Why is model not being set to eval mode?
# MAGIC * Why is model not being sent to GPU?
# MAGIC * Got the following warning during model logging: `The model 'MPTForCausalLM' is not supported for text-generation. Supported models are ...`
# MAGIC * Input example didn't get logged due to invalid schema (using mlflow 2.4.2): 
# MAGIC   ```
# MAGIC   WARNING mlflow.transformers: Attempted to generate a signature for the saved model or pipeline but encountered an error: The input data is of an incorrect type. <class 'dict'> is invalid. Must be either string or List[str]
# MAGIC   ```
# MAGIC   (Side question: where are the docs for expected input schema? Huggingface? mlflow?)
# MAGIC * In the Google doc the model is logged like:
# MAGIC   ```
# MAGIC   mlflow.transformers.log_model(
# MAGIC         "model",
# MAGIC         transformers_model=MPT7BInstruct(),
# MAGIC         input_example={"prompt": np.array(["what is ML?"]), "temperature": np.array([0.5]),"max_tokens": np.array([100])},
# MAGIC         metadata={"task": "llm/v1/completions"},
# MAGIC         registered_model_name='mpt'
# MAGIC   )
# MAGIC   ```
# MAGIC   This is different than what is shown above.
# MAGIC * Error when trying to load model from registry: `AttributeError: module transformers has no attribute MPTForCausalLM` (tried transformers 4.32.0.dev0 and transformers 4.29.1)

# COMMAND ----------


