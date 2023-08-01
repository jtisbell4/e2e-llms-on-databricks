from langchain.llms import Databricks

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


class Chatbot:
    def __init__(self):
        self.conversation_history = ""
        self.model = Databricks(
            endpoint_name="llama2-7b-chat",
            model_kwargs={"temperature": 0.1, "max_new_tokens": 500},
        )

    def dummy_response(self):
        return "I'm just a dumb chatbot right now."

    def chat(self, instruction):
        self.conversation_history += instruction + " [/INST] "
        if len(self.conversation_history) > 4000:
            cropped_history = ...
        response = self.model(self.conversation_history)
        self.conversation_history += response + " </s><s> "
        return response


bot = Chatbot()
print(bot.chat("Hi there!"))
