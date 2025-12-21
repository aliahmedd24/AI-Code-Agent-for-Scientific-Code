"""
Scientific Agent System - Core Package

This package contains the core components:
- GeminiClient: LLM API wrapper with retry logic
- KnowledgeGraph: Shared memory using NetworkX
- Orchestrator: Pipeline coordinator
- AgentPrompts: Specialized system prompts
"""

# Import from knowledge_graph
from core.knowledge_graph import (
    KnowledgeGraph,
    KnowledgeNode,
    KnowledgeEdge,
    NodeType,
    EdgeType,
    get_global_graph,
    set_global_graph,
    reset_global_graph,
)

# Import from gemini_client
try:
    from core.gemini_client import (
        AgentLLM,
        GeminiConfig,
        GeminiModel,
    )
except ImportError:
    AgentLLM = None
    GeminiConfig = None
    GeminiModel = None

# Import from orchestrator
try:
    from core.orchestrator import (
        PipelineOrchestrator,
        PipelineResult,
        PipelineEvent,
        PipelineStage,
    )
except ImportError:
    PipelineOrchestrator = None
    PipelineResult = None
    PipelineEvent = None
    PipelineStage = None

# Import from agent_prompts
try:
    from core.agent_prompts import (
        AgentType,
        get_agent_prompt,
        build_task_prompt,
    )
except ImportError:
    AgentType = None
    get_agent_prompt = None
    build_task_prompt = None

__all__ = [
    # Knowledge Graph
    "KnowledgeGraph",
    "KnowledgeNode",
    "KnowledgeEdge",
    "NodeType",
    "EdgeType",
    "get_global_graph",
    "set_global_graph",
    "reset_global_graph",
    
    # Gemini Client
    "AgentLLM",
    "GeminiConfig",
    "GeminiModel",
    
    # Orchestrator
    "PipelineOrchestrator",
    "PipelineResult",
    "PipelineEvent",
    "PipelineStage",
    
    # Agent Prompts
    "AgentType",
    "get_agent_prompt",
    "build_task_prompt",
]