from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage


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

# *** This is where you would instantiate your Databricks model endpoint ***
# We can use the langchain Databricks integration (see link below)
# https://python.langchain.com/docs/integrations/llms/databricks

# llm = Databricks(endpoint_name="llama2-7b-chat")
llm = ChatOpenAI()

chat_llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=False,
    memory=memory,
)

