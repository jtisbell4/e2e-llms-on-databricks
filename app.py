import time
import os
import gradio as gr
from langchain.chat_models import ChatOpenAI
from langchain.llms import Databricks

from my_llm import get_llm_chain
from environment import load_environment_variables

# *** This is where you would instantiate your Databricks model endpoint ***
# We can use the langchain Databricks integration (see link below)
# https://python.langchain.com/docs/integrations/llms/databricks

LLM = Databricks(endpoint_name="databricks-llama-2-70b-chat")
# LLM = ChatOpenAI()

llm_chain = get_llm_chain(llm=LLM)

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    state = gr.State([])

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history, messages_history):
        user_message = history[-1][0]

        bot_message, messages_history = get_response(
            user_message, messages_history
        )

        messages_history += [{"role": "assistant", "content": bot_message}]

        history[-1][1] = bot_message

        time.sleep(1)
        return history, messages_history

    def get_response(message, messages_history):
        messages_history += [{"role": "user", "content": message}]

        response = llm_chain.predict(input=message)

        return response, messages_history

    def init_history(messages_history):
        messages_history = []
        return messages_history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [chatbot, state], [chatbot, state]
    )

    clear.click(lambda: None, None, chatbot, queue=False).success(
        init_history, [state], [state]
    )

load_environment_variables()

demo.launch()
