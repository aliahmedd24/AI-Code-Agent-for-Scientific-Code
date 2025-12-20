"""
Scientific Agent System - Core Package

This package contains the core components of the LLM-driven agentic pipeline:
- Gemini API client for LLM interactions
- Knowledge graph for inter-agent communication
- Pipeline orchestrator for workflow coordination
- Specialized agent prompts for task-specific behavior
- API key manager for rate limit handling
"""

from core.gemini_client import GeminiClient, AgentLLM, GeminiConfig, GeminiModel
from core.knowledge_graph import KnowledgeGraph, NodeType, EdgeType
from core.orchestrator import PipelineOrchestrator, run_analysis
from core.agent_prompts import (
    AgentType,
    AgentPromptConfig,
    get_agent_prompt,
    get_agent_config,
    build_task_prompt,
    AGENT_PROMPTS
)
from core.api_key_manager import (
    APIKeyManager,
    get_key_manager,
    set_key_manager,
    add_api_key,
    get_available_key
)

__all__ = [
    "GeminiClient",
    "AgentLLM", 
    "GeminiConfig",
    "GeminiModel",
    "KnowledgeGraph",
    "NodeType",
    "EdgeType",
    "PipelineOrchestrator",
    "run_analysis",
    "AgentType",
    "AgentPromptConfig",
    "get_agent_prompt",
    "get_agent_config",
    "build_task_prompt",
    "AGENT_PROMPTS",
    "APIKeyManager",
    "get_key_manager",
    "set_key_manager",
    "add_api_key",
    "get_available_key"
]
