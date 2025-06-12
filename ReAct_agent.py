from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
import os
import subprocess
from langchain.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

load_dotenv()

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

llm = ChatAnthropic(model = "claude-3-5-sonnet-20241022")

tools = [create_directory , create_file , write_in_file , process_info]

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent = agent,
    tools = tools,
    verbose = True
)

while True:
    user_input = input("Enter your command (or type 'exit' to quit): ")

    if user_input.lower() in ["exit", "quit"]:
        print("Exiting agent.")
        break

    response = agent_executor.invoke({"input": user_input})
    print(f"\nAgent response:\n{response['output']}")