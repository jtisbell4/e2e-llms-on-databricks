import time

import gradio as gr
from langchain.llms import Databricks

system_message = {"role": "system", "content": "You are a helpful assistant."}
llm = Databricks(endpoint_name="llama2-7b-chat")

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    state = gr.State([])

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history, messages_history):
        user_message = history[-1][0]
        bot_message, messages_history = ask_gpt(user_message, messages_history)
        messages_history += [{"role": "assistant", "content": bot_message}]
        history[-1][1] = bot_message
        time.sleep(1)
        return history, messages_history

    def ask_gpt(message, messages_history):
        messages_history += [{"role": "user", "content": message}]
        # response = openai.ChatCompletion.create(
        #     model="gpt-3.5-turbo", messages=messages_history
        # )
        response = llm(message, temperature=0.1, max_new_tokens=200)
        return response["choices"][0]["message"]["content"], messages_history

    def init_history(messages_history):
        messages_history = []
        messages_history += [system_message]
        return messages_history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [chatbot, state], [chatbot, state]
    )

    clear.click(lambda: None, None, chatbot, queue=False).success(
        init_history, [state], [state]
    )

demo.launch()
