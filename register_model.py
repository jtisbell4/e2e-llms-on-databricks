import mlflow
import pandas as pd
import torch
import transformers
from huggingface_hub import login, snapshot_download
from mlflow.models.signature import ModelSignature
from mlflow.types import ColSpec, DataType, Schema

# Login to Huggingface to get access to the model
login(token=dbutils.secrets.get(scope="jtisbell", key="hf-key"))

# it is suggested to pin the revision commit hash and not change it for
# reproducibility because the uploader might change the model afterwards; you
# can find the commmit history of llamav2-7b-chat in
# https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/commits/main
model = "meta-llama/Llama-2-7b-chat-hf"
revision = "0ede8dd71e923db6258295621d817ca8714516d4"

# If the model has been downloaded in previous cells, this will not
# repetitively download large model files, but only the remaining files in the
# repo
snapshot_location = snapshot_download(
    repo_id=model, revision=revision, ignore_patterns="*.safetensors"
)


class Llama2(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        This method initializes the tokenizer and language model
        using the specified model repository.
        """
        # Initialize tokenizer and language model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            context.artifacts["repository"], padding_side="left"
        )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            context.artifacts["repository"],
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
            pad_token_id=self.tokenizer.eos_token_id,
        )
        self.model.eval()

    def _build_prompt(self, instruction):
        """
        This method generates the prompt for the model.
        """
        INSTRUCTION_KEY = "### Instruction:"
        RESPONSE_KEY = "### Response:"
        INTRO_BLURB = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request."
        )

        return f"""{INTRO_BLURB}
        {INSTRUCTION_KEY}
        {instruction}
        {RESPONSE_KEY}
        """

    def _generate_response(self, prompt, temperature, max_new_tokens):
        """
        This method generates prediction for a single input.
        """
        # Build the prompt
        prompt = self._build_prompt(prompt)

        # Encode the input and generate prediction
        encoded_input = self.tokenizer.encode(prompt, return_tensors="pt").to(
            "cuda"
        )
        output = self.model.generate(
            encoded_input,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        # Removing the prompt from the generated text
        prompt_length = len(
            self.tokenizer.encode(prompt, return_tensors="pt")[0]
        )
        generated_response = self.tokenizer.decode(
            output[0][prompt_length:], skip_special_tokens=True
        )

        return generated_response

    def predict(self, context, model_input):
        """
        This method generates prediction for the given input.
        """

        outputs = []

        for i in range(len(model_input)):
            prompt = model_input["prompt"][i]
            temperature = model_input.get("temperature", [1.0])[i]
            max_new_tokens = model_input.get("max_new_tokens", [100])[i]

            outputs.append(
                self._generate_response(prompt, temperature, max_new_tokens)
            )

        return outputs


# Define input and output schema
input_schema = Schema(
    [
        ColSpec(DataType.string, "prompt"),
        ColSpec(DataType.double, "temperature"),
        ColSpec(DataType.long, "max_new_tokens"),
    ]
)
output_schema = Schema([ColSpec(DataType.string)])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
input_example = pd.DataFrame(
    {"prompt": ["what is ML?"], "temperature": [0.5], "max_new_tokens": [100]}
)

# Log the model with its details such as artifacts, pip requirements and input
# example
# This may take about 1.7 minutes to complete

mlflow.set_experiment("/Users/taylor.isbell@databricks.com/llm-experiment")

with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        "model",
        python_model=Llama2(),
        artifacts={"repository": snapshot_location},
        pip_requirements=["torch", "transformers", "accelerate"],
        input_example=input_example,
        signature=signature,
    )

# TODO: make dynamic
registered_name = "llamav2_7b_chat_model"

result = mlflow.register_model(
    model_uri="runs:/" + run.info.run_id + "/model",
    name=registered_name,
    await_registration_for=600,
)
