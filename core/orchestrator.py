"""
Pipeline Orchestrator - Coordinates the Entire Agentic System

This module orchestrates:
- Sequential and parallel agent execution
- Inter-agent communication via knowledge graph
- Pipeline state management
- Error handling and recovery
- Report generation
- Overall workflow coordination
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Optional, Any, Dict, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import traceback

from jinja2 import Template

from core.gemini_client import AgentLLM, GeminiConfig, GeminiModel
from core.knowledge_graph import (
    KnowledgeGraph, NodeType, EdgeType,
    get_global_graph, set_global_graph
)
from core.agent_prompts import get_agent_prompt, AgentType
from agents.paper_parser_agent import PaperParserAgent, ParsedPaper
from agents.repo_analyzer_agent import RepoAnalyzerAgent, AnalyzedRepository
from agents.coding_agent import CodingAgent, GeneratedCode, ExecutionResult


class PipelineStage(Enum):
    """Stages of the pipeline."""
    INITIALIZED = "initialized"
    PARSING_PAPER = "parsing_paper"
    ANALYZING_REPO = "analyzing_repo"
    MAPPING_CONCEPTS = "mapping_concepts"
    GENERATING_CODE = "generating_code"
    SETTING_UP_ENV = "setting_up_environment"
    EXECUTING_CODE = "executing_code"
    GENERATING_REPORT = "generating_report"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineEvent:
    """Event in the pipeline."""
    stage: PipelineStage
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    data: Dict[str, Any] = field(default_factory=dict)
    is_error: bool = False


@dataclass
class PipelineResult:
    """Result of a complete pipeline run."""
    success: bool
    paper_info: Dict[str, Any]
    repo_info: Dict[str, Any]
    concept_mappings: List[Dict[str, Any]]
    generated_code: List[Dict[str, Any]]
    execution_results: List[Dict[str, Any]]
    visualizations: List[Dict[str, Any]]
    knowledge_graph: Dict[str, Any]
    report_path: Optional[str]
    events: List[PipelineEvent]
    total_time: float
    error: Optional[str] = None


class PipelineOrchestrator:
    """
    Orchestrates the complete agentic pipeline.
    
    Workflow:
    1. Parse scientific paper
    2. Analyze GitHub repository
    3. Map paper concepts to code
    4. Generate test scripts
    5. Set up execution environment
    6. Execute and capture results
    7. Generate comprehensive report
    """
    
    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        github_token: Optional[str] = None,
        output_dir: str = "./outputs",
        use_docker: bool = True
    ):
        # Create shared knowledge graph
        self.knowledge_graph = KnowledgeGraph("pipeline_graph")
        set_global_graph(self.knowledge_graph)
        
        # Initialize agents
        self.paper_agent = PaperParserAgent(
            knowledge_graph=self.knowledge_graph,
            gemini_api_key=gemini_api_key
        )
        
        self.repo_agent = RepoAnalyzerAgent(
            knowledge_graph=self.knowledge_graph,
            gemini_api_key=gemini_api_key,
            github_token=github_token
        )
        
        self.coding_agent = CodingAgent(
            knowledge_graph=self.knowledge_graph,
            gemini_api_key=gemini_api_key,
            use_docker=use_docker
        )
        
        # LLM for orchestration decisions - with specialized prompt
        system_prompt = get_agent_prompt(AgentType.ORCHESTRATOR)
        
        self.llm = AgentLLM(
            agent_name="Orchestrator",
            agent_role="Pipeline coordination and decision making",
            api_key=gemini_api_key,
            config=GeminiConfig(
                model=GeminiModel.FLASH,
                temperature=0.5,
                max_output_tokens=4096,
                system_instruction=system_prompt  # Use specialized prompt
            )
        )
        
        # State
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_stage = PipelineStage.INITIALIZED
        self.events: List[PipelineEvent] = []
        self.callbacks: List[Callable] = []
        
        # Results storage
        self.parsed_paper: Optional[ParsedPaper] = None
        self.analyzed_repo: Optional[AnalyzedRepository] = None
        self.concept_mappings: List[Dict[str, Any]] = []
        self.generated_code: List[GeneratedCode] = []
        self.execution_results: List[ExecutionResult] = []
    
    def add_callback(self, callback: Callable):
        """Add a callback for pipeline events."""
        self.callbacks.append(callback)
    
    async def _emit_event(
        self,
        stage: PipelineStage,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        is_error: bool = False
    ):
        """Emit a pipeline event."""
        event = PipelineEvent(
            stage=stage,
            message=message,
            data=data or {},
            is_error=is_error
        )
        self.events.append(event)
        self.current_stage = stage
        
        # Call callbacks
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                print(f"Callback error: {e}")
    
    async def run_pipeline(
        self,
        paper_url: str,
        repo_url: str,
        auto_fix_errors: bool = True,
        max_retries: int = 3
    ) -> PipelineResult:
        """
        Run the complete pipeline.
        
        Args:
            paper_url: URL to scientific paper (arXiv, PDF, etc.)
            repo_url: URL to GitHub repository
            auto_fix_errors: Automatically attempt to fix code errors
            max_retries: Maximum retry attempts for failed steps
        
        Returns:
            PipelineResult with all outputs
        """
        start_time = datetime.now()
        error = None
        report_path = None
        
        try:
            # Stage 1: Parse Paper
            await self._emit_event(
                PipelineStage.PARSING_PAPER,
                f"Parsing scientific paper: {paper_url}"
            )
            
            self.parsed_paper = await self.paper_agent.parse_paper(paper_url)
            paper_summary = await self.paper_agent.generate_paper_summary(paper_url)
            paper_concepts = await self.paper_agent.get_concepts_for_code_mapping(paper_url)
            
            await self._emit_event(
                PipelineStage.PARSING_PAPER,
                f"Paper parsed: {self.parsed_paper.title}",
                data={
                    "title": self.parsed_paper.title,
                    "concepts": len(paper_concepts),
                    "sections": len(self.parsed_paper.sections)
                }
            )
            
            # Stage 2: Analyze Repository
            await self._emit_event(
                PipelineStage.ANALYZING_REPO,
                f"Analyzing repository: {repo_url}"
            )
            
            self.analyzed_repo = await self.repo_agent.analyze_repository(repo_url)
            
            await self._emit_event(
                PipelineStage.ANALYZING_REPO,
                f"Repository analyzed: {self.analyzed_repo.name}",
                data={
                    "name": self.analyzed_repo.name,
                    "language": self.analyzed_repo.get_main_language(),
                    "files": len(self.analyzed_repo.files),
                    "dependencies": len(self.analyzed_repo.dependencies)
                }
            )
            
            # Stage 3: Map Concepts to Code
            await self._emit_event(
                PipelineStage.MAPPING_CONCEPTS,
                "Mapping paper concepts to code implementations"
            )
            
            self.concept_mappings = await self.repo_agent.map_concepts_to_code(
                paper_concepts, repo_url
            )
            
            await self._emit_event(
                PipelineStage.MAPPING_CONCEPTS,
                f"Mapped {len(self.concept_mappings)} concept-code relationships",
                data={"mappings": len(self.concept_mappings)}
            )
            
            # Stage 4: Generate Test Code
            await self._emit_event(
                PipelineStage.GENERATING_CODE,
                "Generating test scripts"
            )
            
            repo_info = self.repo_agent.get_repo_info(repo_url)
            self.generated_code = await self.coding_agent.generate_test_script(
                concepts=paper_concepts,
                code_mappings=self.concept_mappings,
                repo_info=repo_info,
                paper_summary=paper_summary
            )
            
            await self._emit_event(
                PipelineStage.GENERATING_CODE,
                f"Generated {len(self.generated_code)} code files",
                data={"files": [c.filename for c in self.generated_code]}
            )
            
            # Stage 5: Set Up Environment
            await self._emit_event(
                PipelineStage.SETTING_UP_ENV,
                "Setting up execution environment"
            )
            
            env = await self.coding_agent.create_sandbox(
                language=self.analyzed_repo.get_main_language(),
                repo_path=self.analyzed_repo.local_path
            )
            
            # Collect all dependencies
            all_deps = set()
            for code in self.generated_code:
                all_deps.update(code.dependencies)
            for dep in self.analyzed_repo.dependencies:
                all_deps.add(dep.name)
            
            await self.coding_agent.install_dependencies(
                env, list(all_deps)[:30]  # Limit dependencies
            )
            
            await self._emit_event(
                PipelineStage.SETTING_UP_ENV,
                f"Environment ready: {env.status}",
                data={"container": env.container_id is not None}
            )
            
            # Stage 6: Execute Code
            await self._emit_event(
                PipelineStage.EXECUTING_CODE,
                "Executing generated code"
            )
            
            for code in self.generated_code:
                result = await self.coding_agent.execute_code(env, code)
                
                # Auto-fix if enabled and failed
                if not result.success and auto_fix_errors:
                    code, result = await self.coding_agent.debug_and_fix(
                        code, result, max_attempts=max_retries
                    )
                
                self.execution_results.append(result)
                
                await self._emit_event(
                    PipelineStage.EXECUTING_CODE,
                    f"{'✓' if result.success else '✗'} {code.filename}",
                    data={
                        "file": code.filename,
                        "success": result.success,
                        "time": result.execution_time
                    }
                )
            
            # Stage 7: Generate Report
            await self._emit_event(
                PipelineStage.GENERATING_REPORT,
                "Generating comprehensive report"
            )
            
            report_path = await self._generate_report(paper_url, repo_url)
            
            await self._emit_event(
                PipelineStage.COMPLETED,
                "Pipeline completed successfully",
                data={"report": report_path}
            )
            
        except Exception as e:
            error = str(e)
            await self._emit_event(
                PipelineStage.FAILED,
                f"Pipeline failed: {error}",
                data={"traceback": traceback.format_exc()},
                is_error=True
            )
        
        finally:
            # Cleanup
            await self.coding_agent.cleanup()
            self.repo_agent.cleanup()
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        return PipelineResult(
            success=(error is None),
            paper_info=self.paper_agent.get_paper_info(paper_url) or {},
            repo_info=self.repo_agent.get_repo_info(repo_url) or {},
            concept_mappings=self.concept_mappings,
            generated_code=[
                {
                    "filename": c.filename,
                    "language": c.language,
                    "purpose": c.purpose,
                    "content": c.content
                }
                for c in self.generated_code
            ],
            execution_results=[
                {
                    "success": r.success,
                    "stdout": r.stdout,
                    "stderr": r.stderr,
                    "time": r.execution_time,
                    "output_files": r.output_files
                }
                for r in self.execution_results
            ],
            visualizations=self.coding_agent.get_all_visualizations(),
            knowledge_graph=self.knowledge_graph.to_dict(),
            report_path=report_path,
            events=self.events,
            total_time=total_time,
            error=error
        )
    
    async def _generate_report(
        self,
        paper_url: str,
        repo_url: str
    ) -> str:
        """Generate a comprehensive HTML report."""
        # Collect all data
        paper_info = self.paper_agent.get_paper_info(paper_url) or {}
        repo_info = self.repo_agent.get_repo_info(repo_url) or {}
        kg_stats = self.knowledge_graph.get_statistics()
        exec_summary = self.coding_agent.get_execution_summary()
        visualizations = self.coding_agent.get_all_visualizations()
        
        # Generate report using template
        report_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scientific Code Analysis Report</title>
    <style>
        :root {
            --bg-primary: #0f0f1a;
            --bg-secondary: #1a1a2e;
            --bg-tertiary: #252540;
            --text-primary: #e8e8f0;
            --text-secondary: #a0a0b8;
            --accent: #6366f1;
            --accent-light: #818cf8;
            --success: #10b981;
            --error: #ef4444;
            --warning: #f59e0b;
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        header {
            background: linear-gradient(135deg, var(--bg-secondary), var(--bg-tertiary));
            padding: 3rem 2rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        h1 {
            font-size: 2.5rem;
            background: linear-gradient(135deg, var(--accent-light), var(--accent));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            color: var(--text-secondary);
            font-size: 1.1rem;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .card {
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid rgba(255,255,255,0.05);
        }
        
        .card h2 {
            font-size: 1.25rem;
            color: var(--accent-light);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .card h2::before {
            content: '';
            width: 4px;
            height: 20px;
            background: var(--accent);
            border-radius: 2px;
        }
        
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }
        
        .stat {
            background: var(--bg-tertiary);
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 1.75rem;
            font-weight: bold;
            color: var(--accent-light);
        }
        
        .stat-label {
            font-size: 0.85rem;
            color: var(--text-secondary);
        }
        
        .code-block {
            background: var(--bg-primary);
            border-radius: 8px;
            padding: 1rem;
            overflow-x: auto;
            font-family: 'Fira Code', monospace;
            font-size: 0.9rem;
            border: 1px solid rgba(255,255,255,0.1);
            max-height: 400px;
            overflow-y: auto;
        }
        
        .success { color: var(--success); }
        .error { color: var(--error); }
        .warning { color: var(--warning); }
        
        .badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 500;
        }
        
        .badge-success { background: rgba(16, 185, 129, 0.2); color: var(--success); }
        .badge-error { background: rgba(239, 68, 68, 0.2); color: var(--error); }
        
        .timeline {
            position: relative;
            padding-left: 2rem;
        }
        
        .timeline::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 2px;
            background: var(--bg-tertiary);
        }
        
        .timeline-item {
            position: relative;
            padding-bottom: 1.5rem;
        }
        
        .timeline-item::before {
            content: '';
            position: absolute;
            left: -2rem;
            top: 0.5rem;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--accent);
            border: 2px solid var(--bg-primary);
        }
        
        .timeline-item.error::before {
            background: var(--error);
        }
        
        .mapping-item {
            background: var(--bg-tertiary);
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 0.75rem;
        }
        
        .mapping-arrow {
            color: var(--accent);
            margin: 0 0.5rem;
        }
        
        .viz-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
        }
        
        .viz-item {
            background: var(--bg-tertiary);
            border-radius: 8px;
            overflow: hidden;
        }
        
        .viz-item img {
            width: 100%;
            height: auto;
        }
        
        .section-title {
            font-size: 1.75rem;
            margin: 2rem 0 1rem;
            color: var(--text-primary);
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        th, td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--bg-tertiary);
        }
        
        th {
            color: var(--text-secondary);
            font-weight: 500;
        }
        
        a {
            color: var(--accent-light);
            text-decoration: none;
        }
        
        a:hover {
            text-decoration: underline;
        }
        
        .full-width {
            grid-column: 1 / -1;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Scientific Code Analysis Report</h1>
            <p class="subtitle">Automated analysis connecting research paper to implementation</p>
            <p style="margin-top: 1rem; color: var(--text-secondary);">
                Generated: {{ timestamp }}
            </p>
        </header>
        
        <div class="grid">
            <div class="card">
                <h2>Paper Information</h2>
                <p><strong>Title:</strong> {{ paper_info.get('title', 'N/A') }}</p>
                <p><strong>Authors:</strong> {{ paper_info.get('authors', ['N/A'])|join(', ') }}</p>
                <p style="margin-top: 1rem;"><strong>Abstract:</strong></p>
                <p style="color: var(--text-secondary); font-size: 0.9rem;">
                    {{ paper_info.get('abstract', 'N/A')[:500] }}...
                </p>
            </div>
            
            <div class="card">
                <h2>Repository Information</h2>
                <p><strong>Name:</strong> {{ repo_info.get('name', 'N/A') }}</p>
                <p><strong>Language:</strong> {{ repo_info.get('main_language', 'N/A') }}</p>
                <p><strong>Files:</strong> {{ repo_info.get('file_count', 0) }}</p>
                <p><strong>Dependencies:</strong> {{ repo_info.get('dependency_count', 0) }}</p>
                <p style="margin-top: 0.5rem;">
                    <a href="{{ repo_info.get('url', '#') }}" target="_blank">View Repository →</a>
                </p>
            </div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h2>Execution Summary</h2>
                <div class="stat-grid">
                    <div class="stat">
                        <div class="stat-value success">{{ exec_summary.get('successful', 0) }}</div>
                        <div class="stat-label">Passed</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value error">{{ exec_summary.get('failed', 0) }}</div>
                        <div class="stat-label">Failed</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{{ "%.1f"|format(exec_summary.get('total_time', 0)) }}s</div>
                        <div class="stat-label">Total Time</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{{ exec_summary.get('visualizations_generated', 0) }}</div>
                        <div class="stat-label">Visualizations</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>Knowledge Graph</h2>
                <div class="stat-grid">
                    <div class="stat">
                        <div class="stat-value">{{ kg_stats.get('total_nodes', 0) }}</div>
                        <div class="stat-label">Nodes</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{{ kg_stats.get('total_edges', 0) }}</div>
                        <div class="stat-label">Edges</div>
                    </div>
                </div>
            </div>
        </div>
        
        <h2 class="section-title">Concept-to-Code Mappings</h2>
        <div class="card">
            {% for mapping in mappings[:10] %}
            <div class="mapping-item">
                <strong>{{ mapping.get('concept_name', 'Unknown') }}</strong>
                <span class="mapping-arrow">→</span>
                <code>{{ mapping.get('code_element', 'N/A') }}</code>
                <span class="badge badge-{{ 'success' if mapping.get('confidence', 0) > 0.7 else 'warning' }}">
                    {{ "%.0f"|format(mapping.get('confidence', 0) * 100) }}%
                </span>
                <p style="color: var(--text-secondary); font-size: 0.85rem; margin-top: 0.5rem;">
                    {{ mapping.get('reasoning', '') }}
                </p>
            </div>
            {% else %}
            <p style="color: var(--text-secondary);">No mappings generated.</p>
            {% endfor %}
        </div>
        
        <h2 class="section-title">Generated Code</h2>
        <div class="grid">
            {% for code in generated_code %}
            <div class="card">
                <h2>{{ code.filename }}</h2>
                <p style="color: var(--text-secondary); margin-bottom: 1rem;">{{ code.purpose }}</p>
                <div class="code-block">
                    <pre>{{ code.content[:2000] }}{% if code.content|length > 2000 %}...{% endif %}</pre>
                </div>
            </div>
            {% endfor %}
        </div>
        
        {% if visualizations %}
        <h2 class="section-title">Visualizations</h2>
        <div class="card">
            <div class="viz-container">
                {% for viz in visualizations %}
                <div class="viz-item">
                    <img src="data:image/{{ viz.format }};base64,{{ viz.data }}" alt="{{ viz.filename }}">
                    <p style="padding: 0.5rem; text-align: center; color: var(--text-secondary);">
                        {{ viz.filename }}
                    </p>
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}
        
        <h2 class="section-title">Execution Timeline</h2>
        <div class="card">
            <div class="timeline">
                {% for event in events %}
                <div class="timeline-item {{ 'error' if event.is_error else '' }}">
                    <strong>{{ event.stage.value }}</strong>
                    <p>{{ event.message }}</p>
                    <small style="color: var(--text-secondary);">{{ event.timestamp }}</small>
                </div>
                {% endfor %}
            </div>
        </div>
        
        <h2 class="section-title">Execution Results</h2>
        <div class="card full-width">
            <table>
                <thead>
                    <tr>
                        <th>Status</th>
                        <th>Time</th>
                        <th>Output Files</th>
                        <th>Details</th>
                    </tr>
                </thead>
                <tbody>
                    {% for result in execution_results %}
                    <tr>
                        <td>
                            <span class="badge badge-{{ 'success' if result.success else 'error' }}">
                                {{ 'PASS' if result.success else 'FAIL' }}
                            </span>
                        </td>
                        <td>{{ "%.2f"|format(result.time) }}s</td>
                        <td>{{ result.output_files|join(', ') or 'None' }}</td>
                        <td>
                            {% if result.success %}
                            <small>{{ result.stdout[:100] }}{% if result.stdout|length > 100 %}...{% endif %}</small>
                            {% else %}
                            <small class="error">{{ result.stderr[:100] }}{% if result.stderr|length > 100 %}...{% endif %}</small>
                            {% endif %}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        
        <footer style="margin-top: 3rem; text-align: center; color: var(--text-secondary);">
            <p>Generated by Scientific Agent System</p>
            <p>Total processing time: {{ "%.1f"|format(total_time) }} seconds</p>
        </footer>
    </div>
</body>
</html>
"""
        
        template = Template(report_template)
        html_content = template.render(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            paper_info=paper_info,
            repo_info=repo_info,
            kg_stats=kg_stats,
            exec_summary=exec_summary,
            mappings=self.concept_mappings,
            generated_code=[
                {
                    "filename": c.filename,
                    "purpose": c.purpose,
                    "content": c.content
                }
                for c in self.generated_code
            ],
            visualizations=visualizations,
            events=self.events,
            execution_results=[
                {
                    "success": r.success,
                    "time": r.execution_time,
                    "output_files": r.output_files,
                    "stdout": r.stdout,
                    "stderr": r.stderr
                }
                for r in self.execution_results
            ],
            total_time=(datetime.now() - datetime.fromisoformat(self.events[0].timestamp)).total_seconds() if self.events else 0
        )
        
        # Save report
        report_path = self.output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Also save knowledge graph visualization
        kg_viz_path = self.output_dir / "knowledge_graph.html"
        with open(kg_viz_path, 'w', encoding='utf-8') as f:
            f.write(self.knowledge_graph.visualize_to_html())
        
        # Save JSON data
        json_path = self.output_dir / f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "paper": paper_info,
                "repo": repo_info,
                "mappings": self.concept_mappings,
                "knowledge_graph": self.knowledge_graph.to_dict()
            }, f, indent=2, default=str)
        
        return str(report_path)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            "stage": self.current_stage.value,
            "events_count": len(self.events),
            "last_event": self.events[-1].message if self.events else None,
            "paper_parsed": self.parsed_paper is not None,
            "repo_analyzed": self.analyzed_repo is not None,
            "code_generated": len(self.generated_code),
            "executions_completed": len(self.execution_results)
        }


# Convenience function to run the pipeline
async def run_analysis(
    paper_url: str,
    repo_url: str,
    gemini_api_key: str,
    github_token: Optional[str] = None,
    output_dir: str = "./outputs",
    use_docker: bool = True,
    progress_callback: Optional[Callable] = None
) -> PipelineResult:
    """
    Convenience function to run the complete analysis pipeline.
    
    Args:
        paper_url: URL to scientific paper
        repo_url: URL to GitHub repository
        gemini_api_key: Gemini API key
        github_token: Optional GitHub token
        output_dir: Output directory for reports
        use_docker: Whether to use Docker for sandboxing
        progress_callback: Optional callback for progress updates
    
    Returns:
        PipelineResult with all outputs
    """
    orchestrator = PipelineOrchestrator(
        gemini_api_key=gemini_api_key,
        github_token=github_token,
        output_dir=output_dir,
        use_docker=use_docker
    )
    
    if progress_callback:
        orchestrator.add_callback(progress_callback)
    
    return await orchestrator.run_pipeline(paper_url, repo_url)
