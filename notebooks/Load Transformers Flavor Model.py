# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Load Transformers Flavor Model

# COMMAND ----------

# MAGIC %pip install git+https://github.com/huggingface/transformers@main
# MAGIC %pip install -U mlflow>=2.5.0
# MAGIC %pip install -U accelerate>=0.20.3
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow

model = mlflow.transformers.load_model(model_uri='models:/optimized-mpt-7b-chat/latest', device=0)

# COMMAND ----------

import transformers
transformers.__version__

# COMMAND ----------


