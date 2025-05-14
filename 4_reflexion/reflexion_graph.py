from typing import List
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessageGraph
from chains import revisor_chain, first_responder_chain
from execute_tools import execute_tools

MAX_ITERATIONS = 3
loop_counter = 0   # ← a true global integer

def event_loop(state):
    global loop_counter
    print(f"[DEBUG] event_loop called, loop_counter = {loop_counter}")
    if loop_counter < MAX_ITERATIONS:
        loop_counter += 1
        return "execute_tools"   # must exactly match your node name
    return END                  # will map to your __end__ node

# … then build your graph as before …
graph = MessageGraph()

graph.add_node("draft", first_responder_chain)
graph.add_node("execute_tools", execute_tools)
graph.add_node("revisor", revisor_chain)

graph.add_edge("draft", "execute_tools")
graph.add_edge("execute_tools", "revisor")

# def event_loop(state: List[BaseMessage]) -> str:
#     count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
#     num_iterations = count_tool_visits
#     if num_iterations > MAX_ITERATIONS:
#         return END
#     return "execute_tools"

graph.add_conditional_edges("revisor", event_loop)
graph.set_entry_point("draft")

app = graph.compile()

print(app.get_graph().draw_mermaid())

response = app.invoke(
    "Write about how small business can leverage AI to grow"
)

# print(response[-1].tool_calls[0]["args"]["answer"])

# #print(response, "response")

answer = response[-1].tool_calls[0]["args"]["answer"]
references = response[-1].tool_calls[0]["args"].get("references", [])
print(answer)
print("\nReferences:", references)

print("===========================================================================")

print(response, "response")