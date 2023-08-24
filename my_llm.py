from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage


def get_llm_chain(llm) -> LLMChain:
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="You are a chatbot having a conversation with a human."
            ),  # The persistent system prompt
            MessagesPlaceholder(
                variable_name="chat_history"
            ),  # Where the memory will be stored.
            HumanMessagePromptTemplate.from_template(
                "{human_input}"
            ),  # Where the human input will injectd
        ]
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

    return LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=False,
        memory=memory,
    )
