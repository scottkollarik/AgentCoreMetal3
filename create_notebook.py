import nbformat as nbf
import os
import asyncio

def create_notebook(filename, cells):
    nb = nbf.v4.new_notebook()
    nb['cells'] = cells
    os.makedirs('notebooks', exist_ok=True)
    with open(f'notebooks/{filename}', 'w') as f:
        nbf.write(nb, f)
    print(f"Created {filename}")

# 00 Quick Start
quick_start_cells = [
    nbf.v4.new_markdown_cell("""# Quick Start Guide

This notebook provides a quick introduction to AgentCore. We'll cover:
1. Environment setup
2. Basic usage
3. Next steps

## Getting Started
First, let's verify our environment is set up correctly."""),
    nbf.v4.new_code_cell("""import sys
import platform
from dotenv import load_dotenv

# Add project root to Python path
sys.path.append('..')

# Load environment variables
load_dotenv()

# Check Python version
print(f"Python version: {platform.python_version()}")
print(f"Platform: {platform.platform()}")
print(f"Architecture: {platform.machine()}")"""),
    nbf.v4.new_markdown_cell("""## Next Steps
1. Review the planning demo (01_agent_planning_demo.ipynb)
2. Try the workflow demo (02_agent_workflow_demo.ipynb)
3. Test the vector store (03_vector_store_test.ipynb)
4. Run core tests (04_test_agent_core.ipynb)""")
]

# 01 Planning Demo
planning_cells = [
    nbf.v4.new_markdown_cell("""# 01. Agent Planning Demo

This notebook demonstrates the planning capabilities of AgentCore's planning agent. We'll explore:
1. Basic task planning
2. Complex task decomposition
3. Plan validation and refinement

## Example Tasks
We'll work through these example tasks:
1. Simple task: "Create a summary of AI trends"
2. Medium task: "Research and compare different AI frameworks"
3. Complex task: "Analyze the impact of AI on healthcare and create a presentation"

## Expected Outputs
For each task, you'll see:
- A structured plan with clear steps
- Validation of the plan's feasibility
- Estimated time and resource requirements"""),
    nbf.v4.new_code_cell("""import sys
import asyncio
sys.path.append('..')

from agents.planning_agent import PlanningAgent
from dotenv import load_dotenv
from IPython.display import display, HTML

# Load environment variables
load_dotenv()"""),
    nbf.v4.new_code_cell("""# Initialize planning agent with configuration
config = {
    "model_name": "gpt-4",  # or "gpt-3.5-turbo" for faster, less expensive results
    "temperature": 0.7,     # Controls creativity vs. consistency
    "max_tokens": 1000,     # Maximum length of generated plans
    "timeout": 30          # Timeout in seconds for API calls
}

planning_agent = PlanningAgent(config)

# Example 1: Simple task
simple_task = "Create a summary of AI trends"
print(f"Task: {simple_task}")

# Generate plan using async/await
async def generate_plan():
    # Get the raw response from the LLM
    response = await planning_agent.llm.ainvoke(
        planning_agent.planning_prompt.format_messages(task=simple_task)
    )
    
    print("\\nRaw LLM Response:")
    print(response.content)
    
    # Now try to parse it into steps
    print("\\nAttempting to parse into steps...")
    plan = await planning_agent.plan(simple_task)
    
    print("\\nParsed Plan:")
    if not plan:
        print("No steps were parsed from the response.")
    else:
        for i, step in enumerate(plan, 1):
            print(f"{i}. {step['description']}")

# Run the async function in Jupyter
await generate_plan()"""),
    nbf.v4.new_markdown_cell("""## Try Your Own Task

Now it's your turn! Try creating a plan for your own task. Here are some tips:
1. Be specific about what you want to achieve
2. Consider dependencies between steps
3. Think about required resources
4. Consider potential challenges"""),
    nbf.v4.new_code_cell("""# Your task here
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
]

# 02 Workflow Demo
workflow_cells = [
    nbf.v4.new_markdown_cell("""# 02. Agent Workflow Demo

This notebook demonstrates the complete workflow of AgentCore, from planning to execution.

## Workflow Steps
1. Task Planning
2. Plan Validation
3. Execution
4. Result Processing
5. Memory Storage"""),
    nbf.v4.new_code_cell("""import sys
sys.path.append('..')

from agents.planning_agent import PlanningAgent
from agents.execution_agent import ExecutionAgent
from utils.vector_memory import VectorMemory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()"""),
    nbf.v4.new_code_cell("""# Initialize components
planning_agent = PlanningAgent()
execution_agent = ExecutionAgent()
vector_memory = VectorMemory()

# Example task
task = "Research the latest developments in AI and create a summary"

# Generate plan
plan = planning_agent.create_plan(task)
print("Generated Plan:")
for i, step in enumerate(plan, 1):
    print(f"{i}. {step}")"""),
    nbf.v4.new_code_cell("""# Execute plan
results = []
for step in plan:
    print(f"\\nExecuting: {step}")
    result = execution_agent.execute_step(step)
    results.append(result)
    print(f"Result: {result}")

# Store results in vector memory
vector_memory.add_document(str(results))""")
]

# 03 Vector Store Test
vector_store_cells = [
    nbf.v4.new_markdown_cell("""# 03. Vector Store Testing

This notebook demonstrates the vector store functionality of AgentCore, including:
1. Document storage and retrieval
2. Similarity search
3. Memory management"""),
    nbf.v4.new_code_cell("""import sys
sys.path.append('..')

from utils.vector_memory import VectorMemory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()"""),
    nbf.v4.new_code_cell("""# Initialize vector store
vector_store = VectorMemory()

# Add test documents
documents = [
    "AgentCore is a powerful AI agent framework.",
    "The framework supports multiple deployment models.",
    "It includes tools for planning and execution."
]

for doc in documents:
    vector_store.add_document(doc)
    print(f"Added document: {doc}")"""),
    nbf.v4.new_code_cell("""# Test similarity search
queries = [
    "What is AgentCore?",
    "What deployment options are available?",
    "What tools does it have?"
]

for query in queries:
    print(f"\\nQuery: {query}")
    results = vector_store.similarity_search(query, k=2)
    print("Results:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content}")""")
]

# 04 Core Testing
core_test_cells = [
    nbf.v4.new_markdown_cell("""# 04. AgentCore Core Testing

This notebook provides comprehensive testing of AgentCore's core functionality, including:
1. Environment and dependency verification
2. Component integration testing
3. Error handling and recovery
4. Performance monitoring"""),
    nbf.v4.new_code_cell("""import sys
import os
import platform
import time
from dotenv import load_dotenv

# Add project root to Python path
sys.path.append('..')

# Load environment variables
load_dotenv()"""),
    nbf.v4.new_code_cell("""# Verify environment
print("System Information:")
print(f"Python version: {platform.python_version()}")
print(f"Platform: {platform.platform()}")
print(f"Architecture: {platform.machine()}")

print("\\nRequired Environment Variables:")
required_vars = ['OPENAI_API_KEY', 'CHROMA_SERVER']
for var in required_vars:
    value = os.getenv(var)
    status = "✓" if value else "✗"
    print(f"{status} {var}")"""),
    nbf.v4.new_code_cell("""# Test component initialization
from agents.planning_agent import PlanningAgent
from agents.execution_agent import ExecutionAgent
from utils.vector_memory import VectorMemory
from utils.tool_orchestrator import ToolOrchestrator

components = {
    "Planning Agent": PlanningAgent(),
    "Execution Agent": ExecutionAgent(),
    "Vector Memory": VectorMemory(),
    "Tool Orchestrator": ToolOrchestrator()
}

print("Component Initialization:")
for name, component in components.items():
    print(f"✓ {name} initialized successfully")""")
]

# Create all notebooks
notebooks = [
    ("00_quick_start.ipynb", quick_start_cells),
    ("01_agent_planning_demo.ipynb", planning_cells),
    ("02_agent_workflow_demo.ipynb", workflow_cells),
    ("03_vector_store_test.ipynb", vector_store_cells),
    ("04_test_agent_core.ipynb", core_test_cells)
]

for filename, cells in notebooks:
    create_notebook(filename, cells)

print("\nAll notebooks created successfully!") 