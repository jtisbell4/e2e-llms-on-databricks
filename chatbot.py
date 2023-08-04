from langchain.llms import Databricks

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


class Chatbot:
    def __init__(self):
        self.conversation_list = [
            f"<s>[INST]<<SYS>>\n{DEFAULT_SYSTEM_PROMPT}\n<</SYS>>\n\n"
        ]
        self.model = Databricks(
            endpoint_name="optimized-mpt-7b-chat",
            model_kwargs={"temperature": 0.1, "max_new_tokens": 4000},
        )

    def chat(self, instruction):
        # Truncate converation history to roughly 2500 words
        while sum([len(x.split(" ")) for x in self.conversation_list]) > 2500:
            self.conversation_list.pop(1)

        conversation = "".join(self.conversation_list)
        conversation += instruction + "[/INST]"

        response = self.model(conversation)

        self.conversation_list.append(
            instruction + "[/INST]" + response + "</s><s>"
        )

        return response


bot = Chatbot()
print(bot.chat("Hi there!"))
