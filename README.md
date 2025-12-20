# 🔬 Scientific Agent System

A fully autonomous **LLM-driven agentic pipeline** that automatically analyzes scientific papers, maps concepts to code implementations, and generates executable test scripts.

## 🎯 Overview

This system uses Google's Gemini API to power a multi-agent architecture that:

1. **Parses Scientific Papers** - Downloads and analyzes papers from arXiv, PDFs, or URLs
2. **Analyzes GitHub Repositories** - Clones and understands codebase structure
3. **Maps Concepts to Code** - Connects paper concepts to their implementations
4. **Generates Test Scripts** - Creates executable code to demonstrate concepts
5. **Executes in Sandboxes** - Runs code safely in Docker containers
6. **Produces Reports** - Generates comprehensive HTML reports with visualizations

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Pipeline Orchestrator                        │
│                    (Coordinates all agents)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  Paper Parser │    │ Repo Analyzer │    │ Coding Agent  │
│     Agent     │    │     Agent     │    │               │
└───────┬───────┘    └───────┬───────┘    └───────┬───────┘
        │                    │                     │
        └─────────────┬──────┴─────────────┬──────┘
                      ▼                    ▼
              ┌───────────────┐    ┌───────────────┐
              │  Knowledge    │    │    Docker     │
              │    Graph      │    │   Sandbox     │
              └───────────────┘    └───────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| **Gemini Client** | Wrapper for Google's Gemini API with retry logic and structured outputs |
| **Knowledge Graph** | Shared memory system using NetworkX for inter-agent communication |
| **Paper Parser Agent** | Extracts content, concepts, and methodology from papers |
| **Repo Analyzer Agent** | Analyzes codebase structure, dependencies, and architecture |
| **Coding Agent** | Generates code, manages Docker sandboxes, executes tests |
| **Orchestrator** | Coordinates the pipeline and generates reports |

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional, for sandboxed execution)
- Google Gemini API key ([Get one here](https://aistudio.google.com/app/apikey))

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/scientific-agent-system.git
cd scientific-agent-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file and add your API key
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### Running the Web Interface

```bash
# Option 1: Quick start script (recommended)
python run.py

# Option 2: Using main.py
python main.py serve

# Option 3: Direct uvicorn
uvicorn api.server:app --host 127.0.0.1 --port 8000
```

**Open http://localhost:8000 in your browser** (NOT 0.0.0.0:8000)

### Command Line Usage

```bash
# Run analysis directly
python main.py analyze \
    --paper "https://arxiv.org/abs/2301.00001" \
    --repo "https://github.com/owner/repo"

# Interactive mode
python main.py interactive

# Demo mode
python main.py demo
```

### Docker Deployment

```bash
# Build and run with docker-compose
docker-compose up -d

# Or build manually
docker build -t scientific-agent-system .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key scientific-agent-system
```

## 📁 Project Structure

```
scientific-agent-system/
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container definition
├── docker-compose.yml     # Multi-container setup
├── .env.example           # Environment template
│
├── core/
│   ├── __init__.py
│   ├── gemini_client.py   # Gemini API wrapper
│   ├── knowledge_graph.py # Shared memory system
│   └── orchestrator.py    # Pipeline coordinator
│
├── agents/
│   ├── __init__.py
│   ├── paper_parser_agent.py    # Paper analysis
│   ├── repo_analyzer_agent.py   # Code analysis
│   └── coding_agent.py          # Code generation
│
├── api/
│   ├── __init__.py
│   └── server.py          # FastAPI backend
│
├── ui/
│   └── index.html         # Web interface
│
└── outputs/               # Generated reports
```

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key |
| `GITHUB_TOKEN` | No | GitHub token for private repos |
| `USE_DOCKER` | No | Enable Docker sandboxing (default: true) |
| `OUTPUT_DIR` | No | Report output directory |

### Gemini Model Selection

The system uses `gemini-2.0-flash-exp` by default. You can configure this in the code:

```python
from core.gemini_client import GeminiConfig, GeminiModel

config = GeminiConfig(
    model=GeminiModel.PRO,  # Use gemini-1.5-pro
    temperature=0.7,
    max_output_tokens=8192
)
```

## 📊 Features

### Paper Analysis
- Downloads from arXiv, PDF URLs, or local files
- Extracts sections, equations, figures, and citations
- Uses LLM to identify key concepts and methodology
- Builds knowledge graph nodes for concepts

### Repository Analysis
- Clones and analyzes GitHub repositories
- Parses Python, JavaScript, TypeScript, and more
- Extracts dependencies from requirements.txt, package.json, etc.
- Identifies entry points and main modules
- Estimates compute requirements

### Concept-to-Code Mapping
- Uses LLM to match paper concepts to code elements
- Creates relationships in the knowledge graph
- Provides confidence scores and reasoning

### Code Generation & Execution
- Generates test scripts based on concepts
- Creates visualizations with matplotlib/plotly
- Executes in Docker sandboxes for safety
- Automatically attempts to fix errors
- Captures outputs and visualizations

### Report Generation
- Comprehensive HTML reports
- Interactive knowledge graph visualization
- Code snippets with syntax highlighting
- Embedded visualizations
- Execution timeline

## 🛠️ API Reference

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/config` | Set API configuration |
| `POST` | `/api/pipeline/start` | Start new analysis |
| `GET` | `/api/pipeline/{id}/status` | Get pipeline status |
| `GET` | `/api/pipeline/{id}/result` | Get full results |
| `GET` | `/api/pipeline/{id}/report` | Get HTML report |
| `GET` | `/api/runs` | List all runs |
| `DELETE` | `/api/pipeline/{id}` | Delete a run |

### WebSocket

Connect to `/ws/{run_id}` for real-time progress updates.

## 🧪 Example Usage

### Python API

```python
import asyncio
from core.orchestrator import run_analysis

async def main():
    result = await run_analysis(
        paper_url="https://arxiv.org/abs/1706.03762",
        repo_url="https://github.com/tensorflow/tensor2tensor",
        gemini_api_key="your_key",
        output_dir="./my_outputs"
    )
    
    print(f"Success: {result.success}")
    print(f"Mappings: {len(result.concept_mappings)}")
    print(f"Report: {result.report_path}")

asyncio.run(main())
```

### With Knowledge Graph Access

```python
from core.knowledge_graph import get_global_graph, NodeType

# After running analysis
kg = get_global_graph()

# Get all concepts
concepts = kg.get_nodes_by_type(NodeType.CONCEPT)
for concept in concepts:
    print(f"{concept.name}: {concept.content}")

# Search for specific topics
results = kg.search("attention mechanism", limit=5)
for node, score in results:
    print(f"{node.name} (score: {score})")

# Get paper-code connections
connections = kg.get_paper_code_connections()
for conn in connections:
    print(f"{conn['paper_element']['name']} -> {conn['code_element']['name']}")
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini API for LLM capabilities
- NetworkX for graph operations
- FastAPI for the web framework
- The open source community

---

Built with ❤️ using LLM-first principles
