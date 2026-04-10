from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from qwen_model.qwen_model import get_llm
from langchain_core.messages import ToolMessage,AIMessageChunk,HumanMessage,AIMessage,BaseMessage


emergency_llm = get_llm(base_url='http://0.0.0.0:8502/v1', model_name = 'Qwen2.5-7B-Instruct')

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

async def chatbot(state: State):
    return {"messages": [await emergency_llm.ainvoke(state["messages"])]}
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
Emergency_agent = graph_builder.compile()
