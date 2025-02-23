import json
from typing import Dict, Any, List
from uuid import uuid4

from langchain.chains.question_answering.map_reduce_prompt import messages
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph
from langchain_gigachat import GigaChat, GigaChatEmbeddings
from langchain_core.tools import tool

import pandas as pd

import os
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

class State(MessagesState):
    is_reasoning: bool


llm = GigaChat(credentials=os.getenv("GIGACHAT_API_KEY"),
                   scope="GIGACHAT_API_PERS",
                   model="GigaChat-Max",
                       verify_ssl_certs=False,
                 profanity_check=False,
                 top_p=0)

# =============== RAG Set UP [Start] ====================

json_path = "data/recommendations.json"
with open(json_path, "r") as f:
    data = json.load(f)

print(len(data))

# Создаём эмбеддинги с использованием OpenAIEmbeddings
embeddings = GigaChatEmbeddings(credentials=os.getenv("GIGACHAT_API_KEY"),
                                scope="GIGACHAT_API_PERS", verify_ssl_certs=False)


documents = [Document(page_content=f'Наблюдение: {t["Наблюдение"]}\nУточняющий вопрос: {t["Уточняющий вопрос"]}\nРекомендация: {t["Рекомендация"]}') for t in data]

vectorstore = FAISS.from_documents(documents, embeddings)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# =============== RAG Set UP [End] ====================

# =============== TOOLS [Start] ====================

data = pd.read_excel('data/client_data.xlsx')

@tool
def get_client_data() -> Dict[str, Any]:
    """Используйте этот инструмент для получения информации о клиенте"""
    #TODO тут надо понять откуда id быдем брать
    client_id = 0
    s = data.iloc[client_id]
    print(s.to_dict())
    return s.to_json()


@tool
def recommendations_on_client_data(client_situation: str) -> str:
    """
    Используйте этот инструмент для получения рекомендаций для пользователя
    args:
        client_situation: причина отказа клиенту в кредите на основе данных о нём
    """

    results = retriever.get_relevant_documents(client_situation)

    # Для примера объединяем найденные тексты в один вывод
    combined_text = "\n---\n".join([doc.page_content for doc in results])

    return combined_text

# =============== TOOLS [End] ====================

tools = [get_client_data, recommendations_on_client_data]
llm_with_tools = llm.bind_tools(tools)


# =============== NODES [Start] ====================

sys_msg = """
Отвечай строго в формате вывода!

Формат вывода:
Рассуждение:
<Тут рассуждение по поводу того что ответить пользователю думаешь про себя>

Ответ пользователю:
<Тут уже обдуманный финальный лаконичны ответ, обращаешься к клиенту>



Ты работник СберБанка ты помогаешь дать рекомендации в случае отказа клиентам по займам.
- получи данные клиента через инструмент проанализируй их
- получи возможные рекомнедации описав ситуацию клиента через инструмент
- задай уточняющие вопросы и потом предложи свои рекомендации
Но не раскрывай полученные данные и фичи клиенту


"""


def assistant(state: State):
   return {"messages": [llm_with_tools.invoke([SystemMessage(sys_msg)] + state['messages'])], "is_reasoning": False}




builder = StateGraph(State)

builder.add_node('assistant', assistant)
builder.add_node('tools', ToolNode(tools))

builder.add_edge(START, 'assistant')
builder.add_conditional_edges('assistant', tools_condition)
builder.add_edge('tools', 'assistant')

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

def stream_graph_updates(graph, user_input, config=None):
    if user_input:
        data = {"messages": ("user", user_input)}
    else:
        data = None

    for event in graph.stream(
            data,
            stream_mode="updates",
            config=config
    ):
        node, data = list(event.items())[0]

        if "messages" in data and len(data['messages']) > 0:
            data["messages"][-1].pretty_print()
            # print(data["messages"][-1])


if __name__ == '__main__':

    config = {"configurable": {"thread_id": str(uuid4())}}

    while True:
        print("___________________________________________")
        user_input = input("User: ")

        if user_input.lower() in ["quit", "exit", "q"]:
            break

        stream_graph_updates(graph, user_input, config=config)

    print("HISTORY")
    snapshot = graph.get_state(config)

    for message in snapshot.values["messages"]:
        message.pretty_print()





