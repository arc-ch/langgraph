from typing import List, Sequence
from dotenv import load_dotenv
import os
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph
from chains import generation_chain, reflection_chain

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
print(google_api_key)
import getpass
import os

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
REFLECT = "reflect"
GENERATE = "generate"
graph = MessageGraph()

def generate_node(state):
    return generation_chain.invoke({
        "messages": state
    })


def reflect_node(messages):
    response = reflection_chain.invoke({
        "messages": messages
    })
    return [HumanMessage(content=response.content)]


graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)
graph.set_entry_point(GENERATE)


def should_continue(state):
    if (len(state) > 4):
        return END 
    return REFLECT


graph.add_conditional_edges(GENERATE, should_continue)
graph.add_edge(REFLECT, GENERATE)

app = graph.compile()

print(app.get_graph().draw_mermaid())
app.get_graph().print_ascii()

response = app.invoke(HumanMessage(content="AI Agents taking over content creation"))

print(response)
