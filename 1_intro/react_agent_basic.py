from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import initialize_agent, tool
from langchain_community.tools import TavilySearchResults
import datetime

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

search_tool = TavilySearchResults(search_depth="basic")

# adding @tool decorator to enable langchain to recognize my own function as a tool 
@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """ Returns the current date and time in the specified format """

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

# List of tools available to the agent, which includes both the search tool and the time tool
tools = [search_tool, get_system_time]

# Initialize the agent, specifying the tools available, the language model, and the agent type.
# The agent is set to "zero-shot-react-description", which means it will dynamically figure out how to use
# the available tools based on the input question. The verbose flag is set to True to print debug output.
agent = initialize_agent(tools=tools, llm=llm, agent="zero-shot-react-description", verbose=True)

agent.invoke("When was SpaceX's last launch and how many days ago was that from this instant")
