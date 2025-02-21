from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os
from langchain_gigachat.chat_models import GigaChat
from typing import Annotated, Dict, Any, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
import inspect
from uuid import uuid4

load_dotenv()

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


class State(TypedDict):
    messages: Annotated[list, add_messages]

#=================TOOLS=========================
@tool
def get_winner_euro_football(year: Literal["2024", "2025"]):
    """Используйте этот инструмент для получения информации о победителе Евро по футболу."""
    if year == "2024":
        return "Евро по футболу 2024 выиграла Испания."
    elif year == "2025":
        return "Евро 2025 еще не состоялся."
    else:
        raise ValueError("Неизвестный год")


@tool
def get_client_data(client_id: str) -> Dict[str, Any]:
    """Используйте этот инструмент для получения информации о клиенте"""
    example_data = {
        "client_id": client_id,
        "income": 80000,
        "existing_loans": [
            {"loan_type": "credit_card", "balance": 60000, "limit": 120000},
            {"loan_type": "consumer_loan", "balance": 200000, "limit": 300000}
        ],
        "credit_score": 650,
        "recommendations_completed": []
    }
    return example_data

def get_recommendation():
    pass

def get_probability_of_default():
    pass


#=================TOOLS=========================



llm = GigaChat(credentials=os.getenv("GIGACHAT_API_KEY"),
                   scope="GIGACHAT_API_PERS",
                   model="GigaChat-Max",
                       verify_ssl_certs=False,
                 profanity_check=False,
                 top_p=0)





def chatbot(state: State):
    SYSTEM_PROMPT = ("Ты работник СберБанка ты помогаешь дать рекомендации в случае отказа клиентам по займам. "
                     "Сначала получи данные клиента по id=25, далее предложи свои рекомендации "
                     "Но не раскрывай полученные данные и фичи клиенту")
    return {"messages": [llm_with_tools.invoke([("system", SYSTEM_PROMPT)] + state["messages"])]}

memory = MemorySaver()
graph_builder = StateGraph(State)



graph_builder.add_node("chatbot", chatbot)

tools = [get_winner_euro_football, get_client_data]
llm_with_tools = llm.bind_tools(tools)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile(checkpointer=memory)


if __name__ == '__main__':
    config = {"configurable": {"thread_id": str(uuid4())}}

    while True:
        print("___________________________________________")
        user_input = input("User:")

        if user_input.lower() in ["quit", "exit", "q"]:
            break
        stream_graph_updates(graph, user_input, config=config)

    print("HISTORY")
    snapshot = graph.get_state(config)

    for message in snapshot.values["messages"]:
        message.pretty_print()

