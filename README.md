# AgentCore

# Agentic AI Starter Kit
**Build. Extend. Orchestrate.**  
The easiest way to kickstart modular, scalable Agentic AI projects.

---

## 📚 Attribution

This project is a fork and implementation of the architectural blueprint provided by [AgentCore](https://github.com/honestsoul/AgentCore). The original repository provided the foundational structure and vision for this project. This implementation adds:

- Vector memory system with Chroma integration
- Error tracking and analysis system
- Planning and execution agents
- RAG-enabled document processing
- Cross-platform support (Windows CUDA, Linux CUDA, MacOS Metal 3)
- Local-first approach using Ollama
- Native Metal 3 GPU acceleration support

Special thanks to the original author for the architectural vision and directory structure that made this implementation possible.

---

## 🚀 Introduction

Agentic AI is the future — autonomous agents that can plan, reason, act, and collaborate.

But getting started often feels overwhelming.

The **Agentic AI Starter Kit** gives you a clean, extensible foundation where:
- Agents, tools, protocols, memory, and workflows are **modular and pluggable**
- New capabilities can be added without rewriting the core
- Configurations are managed cleanly through YAML
- Industry patterns like MCP, A2A, LangGraph orchestration are supported

Whether you're a beginner experimenting or a builder preparing production-grade systems — this starter kit accelerates your journey.

---

## 🏗️ Project Structure

```
agentic-ai-starter-kit/
│
├── agents/           # Agents (Planning, Execution, Monitoring, Custom)
├── protocols/        # Communication protocols (MCP, A2A)
├── tools/            # External tools (Web search, Summarization, DB query)
├── memory/           # Memory modules (short-term, vector)
├── workflows/        # Agent workflows (sequential, parallel)
├── orchestrators/    # Orchestrators (LangGraph, CrewAI, custom)
├── configs/          # YAML configurations
├── services/         # LLM and VectorStore wrappers
├── utils/            # Logging, retries, telemetry
├── examples/         # Ready-to-run demos
├── tests/            # Unit and integration tests
└── README.md
```

---

## ⚙️ Quickstart

Clone the repository:

```bash
git clone https://github.com/brij-kishore-pandey/agentic-ai-starter-kit.git
cd agentic-ai-starter-kit
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run your first Agentic AI Workflow:

```bash
python examples/run_basic_workflow.py
```

---

## 🖥️ Deployment Configurations

AgentCore supports multiple deployment configurations optimized for different platforms:

### MacOS (Apple Silicon)
- **Metal 3 Acceleration (Recommended)**
  - Optimized for Apple Silicon (M1/M2/M3/M4)
  - Automatic Metal shader optimization
  - Efficient memory management
  - Fallback to CPU when needed
  ```bash
  # Enable Metal 3 acceleration
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
  ```

### Windows
- **CUDA Acceleration**
  - NVIDIA GPU support
  - CUDA-optimized operations
  - Automatic CPU fallback
  ```bash
  # Enable CUDA acceleration
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
  ```
- **CPU-Only Mode**
  - Optimized for systems without GPU
  - Multi-threading support
  - Memory-efficient operations

### Linux
- **CUDA Acceleration**
  - NVIDIA GPU support
  - CUDA-optimized operations
  - Automatic CPU fallback
  ```bash
  # Enable CUDA acceleration
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
  ```
- **CPU-Only Mode**
  - Optimized for systems without GPU
  - Multi-threading support
  - Memory-efficient operations

### Configuration Management
The deployment configuration is managed through `configs/deployment_config.yaml`:
```yaml
# Example configuration
platform:
  auto_detect: true
  force_platform: null  # Override if needed

hardware:
  macos:
    metal:
      enabled: true
      priority: 1
```

---

## 🔄 Cross-Container Communication

AgentCore supports distributed tool execution across containers through a robust communication system:

### Architecture
```
[Agent Container] <---> [Tool Orchestrator] <---> [Tool Containers]
      |                        |                        |
  FastAPI                 Redis/MQTT              Various Tools
  (Python)                (Message Queue)         (Python/.NET/etc)
```

### Key Features
1. **Tool Discovery**
   - Automatic tool registration
   - Health checks
   - Capability negotiation

2. **Message Protocol**
   - JSON-based communication
   - Async/await support
   - Error handling
   - Retry mechanisms

3. **Resource Management**
   - Load balancing
   - Resource allocation
   - GPU sharing

### Example Configuration
```yaml
tool_communication:
  orchestrator:
    type: "redis"  # or "mqtt"
    host: "localhost"
    port: 6379
  tools:
    - name: "chart_generator"
      container: "python"
      endpoint: "http://chart-generator:8000"
    - name: "data_processor"
      container: "dotnet"
      endpoint: "http://data-processor:5000"
```

### Usage
```python
from agentcore.tools import ToolOrchestrator

# Initialize tool orchestrator
orchestrator = ToolOrchestrator()

# Call tool across containers
result = await orchestrator.execute_tool(
    tool_name="chart_generator",
    parameters={"data": data, "type": "bar"}
)
```

---

## 🧩 Core Concepts

| Component | Description |
|:----------|:------------|
| **Agents** | Modular agents (PlanningAgent, ExecutionAgent, MonitoringAgent) |
| **Protocols** | Communication logic (MCP, A2A) |
| **Tools** | Web Search, Summarization, Database Query |
| **Memory** | Short-term memory, Vector memory |
| **Workflows** | Sequential and parallel task execution |
| **Orchestrators** | Flexible orchestration styles |
| **Config-Driven** | Easily swap agents, tools, protocols with no code change |

---

## 🛠️ How to Extend

- Add a new Agent → Create a class under `/agents/` inheriting from `BaseAgent`
- Add a new Tool → Drop a new file under `/tools/`
- Add a new Protocol → Extend `/protocols/`
- Create custom workflows → Use `/workflows/`
- Customize orchestration → Swap orchestrators in `/orchestrators/`

---

## 📈 Roadmap (Coming Soon)

- Multi-memory management
- Advanced agent planning modules
- Full GUI for workflow building
- Cloud-native deployment templates (AWS, GCP, Azure)

---

## 🤝 Contributing

We welcome contributions to extend agents, tools, protocols, and workflows.  
Stay tuned for contribution guidelines.

---

## 📄 License

This project is licensed under the MIT License.

---

## ✍️ Designed and maintained by Brij Kishore Pandey
