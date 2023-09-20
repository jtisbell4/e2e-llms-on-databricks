# Databricks notebook source
# MAGIC %md
# MAGIC # Developing LLMs with MLflow
# MAGIC
# MAGIC The goal of this demo is to showcase how Databricks and MLflow make developing, tracking, and sharing LLM models easy. The basic workflow is as follows:
# MAGIC
# MAGIC 1. Download model repo from Hugging Face
# MAGIC 2. Create `mlflow.pyfunc` model object.
# MAGIC 3. Log model in an MLflow run
# MAGIC 4. Register model to MLflow Model Registry
# MAGIC 5. BONUS: deploy model to serving endpoint
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
import numpy as np
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizerFast
import mlflow
import torch

# COMMAND ----------

# MAGIC %md
# MAGIC ## Downloading Model From Hugging Face
# MAGIC
# MAGIC For this demo we will be using [Llama2 7B Chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), so we'll need to authenticate with Hugging Face first before downloading the model repo.

# COMMAND ----------

import huggingface_hub
#skip this if you are already logged in to hugging face
huggingface_hub.login()

# COMMAND ----------

model = "meta-llama/Llama-2-7b-chat-hf"
repository = huggingface_hub.snapshot_download(repo_id=model, ignore_patterns="*.safetensors*")

# COMMAND ----------

# MAGIC %md-sandbox <i18n value="fa7a0aaf-1961-4ee2-8e27-c14c3273bb5c"/>
# MAGIC
# MAGIC ## Create `mlflow.pyfunc` Model
# MAGIC
# MAGIC **A quick rundown on MLflow models...**
# MAGIC
# MAGIC MLflow models is a convention for packaging machine learning models that offers self-contained code, environments, and models.<br>
# MAGIC
# MAGIC * The main abstraction in this package is the concept of **flavors**
# MAGIC   - A flavor is a different ways the model can be used
# MAGIC   - For instance, a TensorFlow model can be loaded as a TensorFlow DAG or as a Python function
# MAGIC   - Using an MLflow model convention allows for both of these flavors
# MAGIC * The `python_function` or `pyfunc` flavor of models gives a generic way of bundling models
# MAGIC * We can thereby deploy a python function without worrying about the underlying format of the model
# MAGIC
# MAGIC **MLflow therefore maps any training framework to any deployment** using these platform-agnostic representations, massively reducing the complexity of inference.
# MAGIC
# MAGIC Arbitrary logic including pre and post-processing steps, arbitrary code executed when loading the model, side artifacts, and more can be included in the pipeline to customize it as needed.  This means that the full pipeline, not just the model, can be preserved as a single object that works with the rest of the MLflow ecosystem.
# MAGIC
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/mlflow-models-enviornments.png" style="height: 400px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC For this demo we will be utilizing the `pyfunc` model flavor because it is highly customizable, but MLflow offers several other relevant flavors for LLMs such as:
# MAGIC * Transformers
# MAGIC * Langchain
# MAGIC * PyTorch
# MAGIC
# MAGIC In order to create a `pyfunc` model, you simply create a subclass of `mlflow.pyfunc.PythonModel`. Two methods are required when doing this: `load_context()` (for loading artifacts and configuring the model) and `predict()` (for doing inference).

# COMMAND ----------

# Define prompt template to get the expected features and performance for the chat versions. See our reference code in github for details: https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L212

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

# Define PythonModel to log with mlflow.pyfunc.log_model

class Llama2(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        This method initializes the tokenizer and language model
        using the specified model repository.
        """
        # Initialize tokenizer and language model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            context.artifacts['repository'], padding_side="left")
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            context.artifacts['repository'], 
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, 
            trust_remote_code=True,
            device_map="auto",
            pad_token_id=self.tokenizer.eos_token_id)
        self.model.eval()

    def _build_prompt(self, instruction):
        """
        This method generates the prompt for the model.
        """
        return f"""<s>[INST]<<SYS>>\n{DEFAULT_SYSTEM_PROMPT}\n<</SYS>>\n\n\n{instruction}[/INST]\n"""

    def _generate_response(self, prompt, temperature, max_new_tokens):
        """
        This method generates prediction for a single input.
        """
        # Build the prompt
        prompt = self._build_prompt(prompt)

        # Encode the input and generate prediction
        encoded_input = self.tokenizer.encode(prompt, return_tensors='pt').to('cuda')
        output = self.model.generate(encoded_input, do_sample=True, temperature=temperature, max_new_tokens=max_new_tokens)

        # Removing the prompt from the generated text
        prompt_length = len(self.tokenizer.encode(prompt, return_tensors='pt')[0])
        generated_response = self.tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)

        return generated_response
      
    def predict(self, context, model_input):
        """
        This method generates prediction for the given input.
        """

        outputs = []

        for i in range(len(model_input["prompt"])):
          prompt = model_input["prompt"][i]
          temperature = model_input.get("temperature", [1.0])[i]
          max_new_tokens = model_input.get("max_new_tokens", [100])[i]

          outputs.append(self._generate_response(prompt, temperature, max_new_tokens))
      
        return outputs

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Log Model to MLflow
# MAGIC
# MAGIC MLflow is automatically integrated with Databricks Notebooks (no setup required). To log the model, simply start an MLflow run and call the `mlflow.pyfunc.log_model()` function. It's important to note that is just a basic example. You could also do something more advanced here like fine-tuning, and you could log training metrics, figures, etc.

# COMMAND ----------

with mlflow.start_run() as run:  
    mlflow.pyfunc.log_model(
        "model",
        python_model=Llama2(),
        artifacts={'repository' : repository},
        pip_requirements=["torch", "transformers", "accelerate"],
        input_example=
          {
              "prompt": ["what is ML?"],
              "max_new_tokens": [80],
              "temperature": [0.7]
          },
    )
    run_id = run.info.run_id

# COMMAND ----------

# MAGIC %md-sandbox <i18n value="c4f47d56-1cc8-4b97-b89f-63257dbb3e31"/>
# MAGIC
# MAGIC ## Register Model to MLflow Model Registry
# MAGIC
# MAGIC
# MAGIC **A quick rundown on the MLflow Model Registry...**
# MAGIC
# MAGIC The MLflow Model Registry component is a centralized model store, set of APIs, and UI, to collaboratively manage the full lifecycle of an MLflow Model. It provides model lineage (which MLflow Experiment and Run produced the model), model versioning, stage transitions (e.g. from staging to production), annotations (e.g. with comments, tags), and deployment management (e.g. which production jobs have requested a specific model version).
# MAGIC
# MAGIC Model registry has the following features:<br>
# MAGIC
# MAGIC * **Central Repository:** Register MLflow models with the MLflow Model Registry. A registered model has a unique name, version, stage, and other metadata.
# MAGIC * **Model Versioning:** Automatically keep track of versions for registered models when updated.
# MAGIC * **Model Stage:** Assigned preset or custom stages to each model version, like “Staging” and “Production” to represent the lifecycle of a model.
# MAGIC * **Model Stage Transitions:** Record new registration events or changes as activities that automatically log users, changes, and additional metadata such as comments.
# MAGIC * **CI/CD Workflow Integration:** Record stage transitions, request, review and approve changes as part of CI/CD pipelines for better control and governance.
# MAGIC
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/model-registry.png" style="height: 400px; margin: 20px"/></div>
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> See <a href="https://mlflow.org/docs/latest/registry.html" target="_blank">the MLflow docs</a> for more details on the model registry.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Our model has now been logged to MLflow, if we are happy with our model we can now register it to the Model Registry so that we can version and deploy it.

# COMMAND ----------

model_uri = f"runs:/{run_id}/model"
model_name = 'llamav2_7b_chat_model'

model_details = mlflow.register_model(model_uri=model_uri, name=model_name, await_registration_for=600)

# COMMAND ----------

from mlflow.client import MlflowClient

client = MlflowClient()
model_details = [x for x in client.get_registered_model(model_name).latest_versions if x.current_stage == 'None'][0]

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## BONUS: Deploy Model to Serving Endpoint
# MAGIC
# MAGIC Databricks now offers the ability to easily deploy models from the Model Registry to a REST API endpoint. This offers several key benefits:
# MAGIC
# MAGIC 1. Host AI models privately with ease: Deploy any type - open source, custom-built, or fine-tuned on your data, without worrying about complex infrastructure.
# MAGIC 2. Reduce TCO with Serverless Serving: Highly available and scalable serving with LLM-specific optimization that reduces latency and cost
# MAGIC 3. Accelerate deployments with Lakehouse Integrations: Automatic feature/vector lookups, monitoring and unified governance that automates deployment and reduce errors
# MAGIC
# MAGIC The deployment process can be done entirely in the UI, but here I will do it programmatically via the Databricks REST API (NOTE: you can also do this programmatically with the [Databricks SDK](https://docs.databricks.com/en/dev-tools/sdk-python.html)!)

# COMMAND ----------

# Provide a name to the serving endpoint
endpoint_name = 'llama2-7b-chat-taylor'

# COMMAND ----------

databricks_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

import requests
import json

deploy_headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
deploy_url = f'{databricks_url}/api/2.0/serving-endpoints'

model_version = model_details # 1  # the returned result of mlflow.register_model
endpoint_config = {
  "name": endpoint_name,
  "config": {
    "served_models": [{
      "name": f'{model_version.name.replace(".", "_")}_{model_version.version}',
      "model_name": model_version.name,
      "model_version": model_version.version,
      "workload_type": "GPU_MEDIUM",
      "workload_size": "Small",
      "scale_to_zero_enabled": "False"
    }]
  }
}
endpoint_json = json.dumps(endpoint_config, indent='  ')

# Send a POST request to the API
deploy_response = requests.request(method='POST', headers=deploy_headers, url=deploy_url, data=endpoint_json)

if deploy_response.status_code != 200:
  raise Exception(f'Request failed with status {deploy_response.status_code}, {deploy_response.text}')

# Show the response of the POST request
# When first creating the serving endpoint, it should show that the state 'ready' is 'NOT_READY'
# You can check the status on the Databricks model serving endpoint page, it is expected to take ~35 min for the serving endpoint to become ready
print(deploy_response.json())

# COMMAND ----------

status_url = f'{databricks_url}/api/2.0/serving-endpoints/{endpoint_name}'
status_response = requests.request(method='GET', headers=deploy_headers, url=status_url)

if status_response.status_code != 200:
  raise Exception(f'Request failed with status {status_response.status_code}, {status_response.text}')

print(status_response.json())

# COMMAND ----------


