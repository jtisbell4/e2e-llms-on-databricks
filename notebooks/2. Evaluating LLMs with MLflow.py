# Databricks notebook source
# MAGIC %md
# MAGIC # Evaluating LLMs with MLflow
# MAGIC
# MAGIC The goal of this demo is to showcase MLflow's LLM evaluation capabilities, which is a new feature that was released in version 2.4. For more information please see this [blog post](https://www.databricks.com/blog/announcing-mlflow-24-llmops-tools-robust-model-evaluation) that highlights this release.
# MAGIC
# MAGIC In the previous notebook we registered a Llama2 model to the Model Registry, but before we deploy this model we need to evaluate it's performance. Questions like "Is this model returning hallucinations?", "Is it returning toxic responses?", or "What's the optimal max_tokens or temperature?" all need to be explored to make sure the model is meeting the customers' needs. To do this, we will use MLflow's new LLM evaluation API!
# MAGIC
# MAGIC **Cluster Specs:** g4dn.2xlarge (32GB GPU memory) with DBR 13.3 LTS ML (**`mlflow` >= 2.4 is required for this demo**)
# MAGIC

# COMMAND ----------

!pip install --upgrade transformers
!pip install --upgrade accelerate
!pip install --upgrade mlflow
dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Mlflow 2.4 introduced the new `mlflow.evaluate()` integration with language modeling tasks. For this demo we will be utilizing the "question-answering" evaluation task to evaluate the Llama2 model we created in xxx. When prompting the model, this task automatically calculates the following metrics for each response: forexact_match, mean_perplexity (requires `evaluate`, `pytorch`, `transformers`), toxicity_ratio (requires `evaluate`, `pytorch`, `transformers`), mean_ari_grade_level (requires `textstat`), mean_flesch_kincaid_grade_level (requires `textstat`).
# MAGIC
# MAGIC Please refer to the docs [here](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.evaluate) for more information on `mlflow.evaluate`.

# COMMAND ----------

def evaluate_model(model, model_input):
  evaluation_results = mlflow.evaluate(
    model,
    data=model_input,
    model_type="question-answering",
  )
  return evaluation_results


# COMMAND ----------

from mlflow.client import MlflowClient

client = MlflowClient()
model_details = [x for x in client.get_registered_model(model_name).latest_versions if x.current_stage == 'None'][0]

# COMMAND ----------

prompt = "What is Spark?"
temperatures = [0.60, 0.80, 0.95]

with mlflow.start_run() as run:

  model = mlflow.pyfunc.load_model(model_uri="models:/llamav2_7b_chat_model/5")

  for temperature in temperatures:
      evaluate_model(
          model=model,
          model_input=pd.DataFrame(
              {"prompt": [prompt], "max_new_tokens": [100], "temperature": [temperature]}
          )
      )

# COMMAND ----------

evaluation_results = mlflow.evaluate(
    model,
    data=test_data,
    model_type="question-answering",
)

# COMMAND ----------


