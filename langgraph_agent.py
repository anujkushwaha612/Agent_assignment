from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
import os
import subprocess
from langchain.tools import tool
from typing import TypedDict, Annotated, Sequence
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage , BaseMessage , ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage] , add_messages]
    
@tool
def create_directory(dir_name : str) -> str:
    """Creates a directory with the given name"""

    try:
        dir_name = dir_name.strip()
        dir_name = dir_name.strip('"').strip("'")
        os.makedirs(dir_name , exist_ok = True)
        return f"Successfully created {dir_name} directory"
    except Exception as e:
        return f"Error creating directory : {e}"
    
@tool
def create_file(file_name : str) -> str:
    """Creates an empty file if it doesn't already exist.""" 
    
    try:
        file_name = file_name.strip()
        file_name = file_name.strip('"').strip("'")
        if os.path.exists(file_name):
            return f"{file_name} already exists"
        else:
            open(file_name , "w").close() # "w" -> overwrites the content of the file but since here we use close so the file create will always be empty
            return f"Successfully create {file_name} file."
    
    except Exception as e:
        return f"Error creting file {file_name}"
    
@tool
def write_in_file(input : str) -> str:
    # """Write content inside the existing file (or create if it doesn't exist and then write content inside it)"""
    # currently our function is not passing input in the correct format. 
    # previously we were using (file_name : str , content : str ) -> str because of which the model was not able to separate the file name and content separately
    
    
    """Write content inside a file. Input should be in 'file_name, content' format."""
    try:
        file_name, content = input.split(",", 1)
        file_name = file_name.strip().strip('"').strip("'")
        content = content.strip().strip('"').strip("'")
        with open(file_name , "a") as f: # "a" -> append the content inside the file
            f.write(content)
            
            return f"Successfully written content inside {file_name}"
        
    except Exception as e:
        return f"Error in writing content inside {file_name} : {e}"
    
@tool
def process_info(dummy_input : str = "") -> str:
    """This function gives information about current running processes and their memory usage"""
    
    try:
        result = subprocess.check_output("tasklist" , shell = True , text = True)
        
        # subprocess -> Standard python module that runs shell commands and return its output
        # "tasklist" -> it is the windows OS command that lists all currently running processees
        # text = True -> the obtained output is text rather than bytes
        
        # This function is internally using process_info() to get all the processes and then finds relevant process based on input

        return result
    except Exception as e:
        return f"Error fetching processes : {e}"

def call_model(state : AgentState) -> AgentState:
    system_prompt = SystemMessage(content = "you are my AI assistant, please answer my query to the best of your ability.")
    
    response = llm.invoke([system_prompt] + state["messages"])
    
    return {"messages" : [response]}

def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, AIMessage):
        for item in last_message.content:
            if isinstance(item, dict) and item.get("type") == "tool_use":
                return "continue"
    return "end"

# custom tool_node function
def run_tools(state: AgentState) -> AgentState:
    tools_map = {tool.name: tool for tool in tools}
    messages = state["messages"]
    last_message = messages[-1]

    tool_messages = []

    if isinstance(last_message, AIMessage):
        # each AI message is a list of dict so first we find which dict contain the "type" == "tool_use"
        for part in last_message.content:
            if part.get("type") == "tool_use":
                tool_name = part["name"]
                tool_input = part["input"]
                tool_id = part["id"]

                if tool_name in tools_map:
                    result = tools_map[tool_name].invoke(tool_input)
                    tool_messages.append(
                        ToolMessage(
                            tool_call_id=tool_id,
                            content=str(result)
                        )
                    )
                else:
                    tool_messages.append(
                        ToolMessage(
                            tool_call_id=tool_id,
                            content=f"Tool '{tool_name}' not found."
                        )
                    )

    return {"messages": messages + tool_messages}


tools = [create_directory , create_file , write_in_file , process_info]
llm = ChatAnthropic(model = "claude-3-5-sonnet-20241022").bind_tools(tools)

graph = StateGraph(AgentState)
graph.add_node("agent" , call_model)

# tool_node = ToolNode(tools=tools)
# graph.add_node("tools" , tool_node)

# using custom tool_node ----->
graph.add_node("tools" , run_tools)

graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue" : "tools",
        "end" : END
    }
)

graph.add_edge("tools", "agent")

app = graph.compile()

state = {"messages": []}

user_input = input("User : ")
while user_input != "exit":
    state["messages"].append(HumanMessage(content=user_input))

    state = app.invoke(state)

    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            print(f"Tool : {msg.content}")
        elif isinstance(msg, AIMessage):
            print(f"AI : {msg.content}")

    user_input = input("User : ")