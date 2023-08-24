import time

import gradio as gr

from langchain_bot import chat_llm_chain

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

        response = chat_llm_chain.predict(human_input=message)

        return response, messages_history

    def init_history(messages_history):
        messages_history = []
        # messages_history += [system_message]
        return messages_history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [chatbot, state], [chatbot, state]
    )

    clear.click(lambda: None, None, chatbot, queue=False).success(
        init_history, [state], [state]
    )

demo.launch()
