import nbformat as nbf
import os

# Create debug directory if it doesn't exist
os.makedirs('notebooks/debug', exist_ok=True)

# Create a new notebook
nb = nbf.v4.new_notebook()

# Add markdown cell for introduction
intro_cell = nbf.v4.new_markdown_cell("""# Debug Panel Demo

This notebook demonstrates how to use the debug panel in AgentCore. The debug panel provides detailed information about:
1. Step parsing process
2. LLM responses
3. Component interactions

## How to Use
1. Enable debug mode in the agent configuration
2. The debug panel will automatically appear when running tasks
3. Click on the panel to expand/collapse the debug information""")

# Add code cell for imports
imports_cell = nbf.v4.new_code_cell("""import sys
import asyncio
sys.path.append('../..')

from agents.planning_agent import PlanningAgent
from dotenv import load_dotenv
from IPython.display import display, HTML, Markdown
import ipywidgets as widgets

# Load environment variables
load_dotenv()""")

# Add code cell for agent initialization
init_cell = nbf.v4.new_code_cell("""# Initialize planning agent with debug mode enabled
config = {
    "model_name": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 1000,
    "debug": True  # Enable debug mode
}

planning_agent = PlanningAgent(config)

# Example task
task = "Create a simple to-do list for a software project"
print(f"Task: {task}")

# Generate plan using async/await
async def generate_plan():
    # Get the raw response from the LLM
    response = await planning_agent.llm.ainvoke(
        planning_agent.planning_prompt.format_messages(task=task)
    )
    
    print("\\nRaw LLM Response:")
    print(response.content)
    
    # Now try to parse it into steps
    print("\\nAttempting to parse into steps...")
    plan = await planning_agent.plan(task)
    
    print("\\nParsed Plan:")
    if not plan:
        print("No steps were parsed from the response.")
    else:
        for i, step in enumerate(plan, 1):
            print(f"{i}. {step['description']}")

# Run the async function in Jupyter
await generate_plan()""")

# Add markdown cell for user task section
user_task_cell = nbf.v4.new_markdown_cell("""## Try Your Own Task

Now try creating a plan for your own task. The debug panel will show you how the agent parses the steps from the LLM's response.""")

# Add code cell for user task
user_task_code_cell = nbf.v4.new_code_cell("""# Your task here
your_task = ""  # Replace with your task

if your_task:
    async def generate_your_plan():
        # Get the raw response from the LLM
        response = await planning_agent.llm.ainvoke(
            planning_agent.planning_prompt.format_messages(task=your_task)
        )
        
        print("\\nRaw LLM Response:")
        print(response.content)
        
        # Now try to parse it into steps
        print("\\nAttempting to parse into steps...")
        plan = await planning_agent.plan(your_task)
        
        print("\\nParsed Plan:")
        if not plan:
            print("No steps were parsed from the response.")
        else:
            for i, step in enumerate(plan, 1):
                print(f"{i}. {step['description']}")
    
    await generate_your_plan()""")

# Add cells to notebook
nb.cells = [
    intro_cell,
    imports_cell,
    init_cell,
    user_task_cell,
    user_task_code_cell
]

# Add metadata
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "codemirror_mode": {
            "name": "ipython",
            "version": 3
        },
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.11.12"
    }
}

# Write the notebook to a file
with open('notebooks/debug/01_debug_panel_demo.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Debug panel demo notebook created successfully!") 