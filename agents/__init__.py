"""
Scientific Agent System - Agents Package

This package contains the specialized agents:
- PaperParserAgent: Analyzes scientific papers
- RepoAnalyzerAgent: Analyzes GitHub repositories
- CodingAgent: Generates and executes code
"""

from agents.paper_parser_agent import PaperParserAgent, ParsedPaper
from agents.repo_analyzer_agent import RepoAnalyzerAgent, AnalyzedRepository
from agents.coding_agent import CodingAgent, GeneratedCode, ExecutionResult

__all__ = [
    "PaperParserAgent",
    "ParsedPaper",
    "RepoAnalyzerAgent",
    "AnalyzedRepository",
    "CodingAgent",
    "GeneratedCode",
    "ExecutionResult"
]
