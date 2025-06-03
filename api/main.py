from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import yaml
import os
from dotenv import load_dotenv

from agents.planning_agent import PlanningAgent
from agents.execution_agent import ExecutionAgent
from tools.web_search import WebSearchTool
from tools.summarizer import SummarizerTool
from memory.vector_memory import VectorMemory

# Load environment variables
load_dotenv()

app = FastAPI(
    title="AgentCore API",
    description="API for the AgentCore system",
    version="1.0.0"
)

# Load configuration
with open('configs/agent_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize components
planner = PlanningAgent(config['planning_agent'])
web_search = WebSearchTool(max_results=5).create_tool()
summarizer = SummarizerTool(model_name="gpt-4", max_length=500).create_tool()
executor = ExecutionAgent(config['execution_agent'], tools=[web_search, summarizer])
memory = VectorMemory(config['memory'])

class TaskRequest(BaseModel):
    task: str
    use_memory: bool = True

class TaskResponse(BaseModel):
    plan: List[Dict[str, Any]]
    results: List[Dict[str, Any]]
    memory_entries: Optional[List[Dict[str, Any]]] = None

@app.post("/execute-task", response_model=TaskResponse)
async def execute_task(request: TaskRequest):
    try:
        # Step 1: Planning
        plan = await planner.plan(request.task)
        
        # Step 2: Execution
        results = []
        for step in plan:
            result = await executor.execute(step)
            results.append(result)
            
            # Store in memory if enabled
            if request.use_memory:
                memory.add_memory(
                    content=f"Step: {step['description']}\nResult: {result['result']}",
                    metadata={"step_number": len(results)}
                )
        
        # Step 3: Get relevant memories if enabled
        memory_entries = None
        if request.use_memory:
            relevant_memories = memory.search_memory(request.task)
            memory_entries = [
                {
                    "content": mem.page_content,
                    "metadata": mem.metadata
                }
                for mem in relevant_memories
            ]
        
        return TaskResponse(
            plan=plan,
            results=results,
            memory_entries=memory_entries
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 