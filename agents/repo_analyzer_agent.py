"""
Repository Analyzer Agent - GitHub Repository Analysis and Understanding

This agent is responsible for:
- Cloning and analyzing GitHub repositories
- Understanding codebase structure and architecture
- Identifying dependencies and requirements
- Estimating compute resources needed
- Mapping code to paper concepts via knowledge graph
"""

import os
import re
import json
import asyncio
import tempfile
import shutil
from typing import Optional, Any, Dict, List, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
from collections import defaultdict

import httpx
from github import Github
import git

from core.gemini_client import AgentLLM, GeminiConfig, GeminiModel
from core.knowledge_graph import (
    KnowledgeGraph, NodeType, EdgeType,
    get_global_graph
)
from core.agent_prompts import (
    get_agent_prompt, AgentType,
    REPO_STRUCTURE_ANALYSIS_PROMPT,
    MAPPING_ANALYSIS_PROMPT,
    build_task_prompt
)


@dataclass
class CodeFile:
    """Represents a code file in the repository."""
    path: str
    language: str
    content: str
    size: int
    imports: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    docstrings: List[str] = field(default_factory=list)


@dataclass
class Dependency:
    """Represents a project dependency."""
    name: str
    version: Optional[str] = None
    source: str = ""  # pip, npm, etc.
    required: bool = True
    extras: List[str] = field(default_factory=list)


@dataclass
class AnalyzedRepository:
    """Complete analyzed repository structure."""
    name: str
    url: str
    description: str
    languages: Dict[str, int]  # language -> line count
    files: List[CodeFile]
    dependencies: List[Dependency]
    entry_points: List[str]
    readme_content: str
    structure: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    local_path: str = ""
    
    def get_main_language(self) -> str:
        """Get the primary programming language."""
        if not self.languages:
            return "unknown"
        return max(self.languages, key=self.languages.get)


class RepoAnalyzerAgent:
    """
    Agent for analyzing GitHub repositories.
    
    Capabilities:
    - Clone repositories locally
    - Analyze code structure and architecture
    - Extract dependencies from various sources
    - Identify entry points and main modules
    - Estimate compute requirements
    - Build knowledge graph from code
    - Map code elements to paper concepts
    """
    
    # File extensions to language mapping
    LANGUAGE_MAP = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
        ".rs": "rust",
        ".go": "go",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".r": "r",
        ".R": "r",
        ".jl": "julia",
        ".m": "matlab",
        ".sh": "shell",
        ".bash": "shell",
    }
    
    # Files to ignore
    IGNORE_PATTERNS = {
        "__pycache__", "node_modules", ".git", ".venv", "venv",
        "env", ".env", "dist", "build", ".idea", ".vscode",
        "*.pyc", "*.pyo", "*.egg-info", ".eggs"
    }
    
    def __init__(
        self,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        gemini_api_key: Optional[str] = None,
        github_token: Optional[str] = None
    ):
        self.knowledge_graph = knowledge_graph or get_global_graph()
        
        # Get specialized system prompt for this agent
        system_prompt = get_agent_prompt(AgentType.REPO_ANALYZER)
        
        self.llm = AgentLLM(
            agent_name="RepoAnalyzer",
            agent_role="GitHub repository analysis and code understanding",
            api_key=gemini_api_key,
            config=GeminiConfig(
                model=GeminiModel.FLASH,
                temperature=0.3,
                max_output_tokens=8192,
                system_instruction=system_prompt  # Use specialized prompt
            )
        )
        self.github = Github(github_token) if github_token else None
        self.analyzed_repos: Dict[str, AnalyzedRepository] = {}
        self.temp_dirs: List[str] = []
    
    async def analyze_repository(self, repo_url: str) -> AnalyzedRepository:
        """
        Analyze a GitHub repository.
        
        Args:
            repo_url: GitHub repository URL
        
        Returns:
            AnalyzedRepository object
        """
        # Clone the repository
        local_path = await self._clone_repository(repo_url)
        
        # Get repository metadata
        metadata = await self._get_repo_metadata(repo_url)
        
        # Analyze the codebase
        files = await self._analyze_files(local_path)
        
        # Extract dependencies
        dependencies = await self._extract_dependencies(local_path)
        
        # Find entry points
        entry_points = await self._find_entry_points(local_path, files)
        
        # Read README
        readme = await self._read_readme(local_path)
        
        # Calculate language statistics
        languages = self._calculate_language_stats(files)
        
        # Build structure tree
        structure = self._build_structure_tree(local_path)
        
        # Create analyzed repository object
        repo = AnalyzedRepository(
            name=metadata.get("name", Path(repo_url).stem),
            url=repo_url,
            description=metadata.get("description", ""),
            languages=languages,
            files=files,
            dependencies=dependencies,
            entry_points=entry_points,
            readme_content=readme,
            structure=structure,
            metadata=metadata,
            local_path=local_path
        )
        
        # Use LLM for deeper analysis
        await self._enhance_with_llm(repo)
        
        # Build knowledge graph
        await self._build_knowledge_graph(repo)
        
        # Store for later reference
        self.analyzed_repos[repo_url] = repo
        
        return repo
    
    async def _clone_repository(self, repo_url: str) -> str:
        """Clone a repository to a temporary directory."""
        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix="repo_")
        self.temp_dirs.append(temp_dir)
        
        # Clone the repository
        try:
            await asyncio.to_thread(
                git.Repo.clone_from,
                repo_url,
                temp_dir,
                depth=1  # Shallow clone for speed
            )
        except Exception as e:
            # Try with git command directly
            process = await asyncio.create_subprocess_exec(
                "git", "clone", "--depth", "1", repo_url, temp_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
        
        return temp_dir
    
    async def _get_repo_metadata(self, repo_url: str) -> Dict[str, Any]:
        """Get repository metadata from GitHub API."""
        metadata = {}
        
        # Extract owner and repo name from URL
        match = re.search(r'github\.com[/:]([^/]+)/([^/\.]+)', repo_url)
        if match:
            owner, repo_name = match.groups()
            
            try:
                if self.github:
                    repo = self.github.get_repo(f"{owner}/{repo_name}")
                    metadata = {
                        "name": repo.name,
                        "full_name": repo.full_name,
                        "description": repo.description,
                        "stars": repo.stargazers_count,
                        "forks": repo.forks_count,
                        "language": repo.language,
                        "topics": repo.get_topics(),
                        "created_at": str(repo.created_at),
                        "updated_at": str(repo.updated_at),
                        "license": repo.license.name if repo.license else None,
                        "default_branch": repo.default_branch
                    }
                else:
                    # Use public API without token
                    async with httpx.AsyncClient() as client:
                        response = await client.get(
                            f"https://api.github.com/repos/{owner}/{repo_name}"
                        )
                        if response.status_code == 200:
                            data = response.json()
                            metadata = {
                                "name": data.get("name"),
                                "full_name": data.get("full_name"),
                                "description": data.get("description"),
                                "stars": data.get("stargazers_count"),
                                "forks": data.get("forks_count"),
                                "language": data.get("language"),
                                "topics": data.get("topics", []),
                                "license": data.get("license", {}).get("name")
                            }
            except Exception as e:
                metadata["error"] = str(e)
        
        return metadata
    
    async def _analyze_files(self, repo_path: str) -> List[CodeFile]:
        """Analyze all code files in the repository."""
        files = []
        
        for root, dirs, filenames in os.walk(repo_path):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if not any(
                d == p or (p.startswith("*") and d.endswith(p[1:]))
                for p in self.IGNORE_PATTERNS
            )]
            
            for filename in filenames:
                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(file_path, repo_path)
                
                # Get file extension
                ext = os.path.splitext(filename)[1].lower()
                
                if ext in self.LANGUAGE_MAP:
                    language = self.LANGUAGE_MAP[ext]
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # Parse the file
                        code_file = await self._parse_code_file(
                            relative_path, language, content
                        )
                        files.append(code_file)
                    
                    except Exception as e:
                        # Skip files that can't be read
                        pass
        
        return files
    
    async def _parse_code_file(
        self,
        path: str,
        language: str,
        content: str
    ) -> CodeFile:
        """Parse a code file to extract structure."""
        imports = []
        classes = []
        functions = []
        docstrings = []
        
        if language == "python":
            # Extract Python imports
            import_patterns = [
                r'^import\s+([\w\.]+)',
                r'^from\s+([\w\.]+)\s+import'
            ]
            for pattern in import_patterns:
                imports.extend(re.findall(pattern, content, re.MULTILINE))
            
            # Extract class definitions
            classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
            
            # Extract function definitions
            functions = re.findall(r'^def\s+(\w+)', content, re.MULTILINE)
            
            # Extract docstrings
            docstrings = re.findall(r'"""(.+?)"""', content, re.DOTALL)[:5]
        
        elif language in ("javascript", "typescript"):
            # Extract JS/TS imports
            imports = re.findall(
                r"(?:import|require)\s*\(?['\"]([^'\"]+)['\"]",
                content
            )
            
            # Extract class definitions
            classes = re.findall(r'class\s+(\w+)', content)
            
            # Extract function definitions
            functions = re.findall(
                r'(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s*)?\()',
                content
            )
            functions = [f[0] or f[1] for f in functions if f[0] or f[1]]
        
        return CodeFile(
            path=path,
            language=language,
            content=content,
            size=len(content),
            imports=list(set(imports)),
            classes=classes,
            functions=functions,
            docstrings=docstrings
        )
    
    async def _extract_dependencies(self, repo_path: str) -> List[Dependency]:
        """Extract dependencies from various package files."""
        dependencies = []
        
        # Python - requirements.txt
        req_path = os.path.join(repo_path, "requirements.txt")
        if os.path.exists(req_path):
            with open(req_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Parse requirement line
                        match = re.match(r'^([a-zA-Z0-9_-]+)([<>=!]+)?(.+)?$', line)
                        if match:
                            name = match.group(1)
                            version = match.group(3) if match.group(2) else None
                            dependencies.append(Dependency(
                                name=name,
                                version=version,
                                source="pip"
                            ))
        
        # Python - setup.py
        setup_path = os.path.join(repo_path, "setup.py")
        if os.path.exists(setup_path):
            with open(setup_path, 'r') as f:
                content = f.read()
                # Extract install_requires
                requires_match = re.search(
                    r'install_requires\s*=\s*\[(.*?)\]',
                    content,
                    re.DOTALL
                )
                if requires_match:
                    deps = re.findall(r'["\']([^"\']+)["\']', requires_match.group(1))
                    for dep in deps:
                        name = re.match(r'^([a-zA-Z0-9_-]+)', dep)
                        if name:
                            dependencies.append(Dependency(
                                name=name.group(1),
                                source="pip"
                            ))
        
        # Python - pyproject.toml
        pyproject_path = os.path.join(repo_path, "pyproject.toml")
        if os.path.exists(pyproject_path):
            with open(pyproject_path, 'r') as f:
                content = f.read()
                # Simple parsing for dependencies
                deps_match = re.search(
                    r'dependencies\s*=\s*\[(.*?)\]',
                    content,
                    re.DOTALL
                )
                if deps_match:
                    deps = re.findall(r'["\']([^"\']+)["\']', deps_match.group(1))
                    for dep in deps:
                        name = re.match(r'^([a-zA-Z0-9_-]+)', dep)
                        if name:
                            dependencies.append(Dependency(
                                name=name.group(1),
                                source="pip"
                            ))
        
        # JavaScript/TypeScript - package.json
        package_path = os.path.join(repo_path, "package.json")
        if os.path.exists(package_path):
            with open(package_path, 'r') as f:
                try:
                    package = json.load(f)
                    for dep_type in ["dependencies", "devDependencies"]:
                        if dep_type in package:
                            for name, version in package[dep_type].items():
                                dependencies.append(Dependency(
                                    name=name,
                                    version=version,
                                    source="npm",
                                    required=(dep_type == "dependencies")
                                ))
                except json.JSONDecodeError:
                    pass
        
        # Remove duplicates
        seen = set()
        unique_deps = []
        for dep in dependencies:
            key = (dep.name, dep.source)
            if key not in seen:
                seen.add(key)
                unique_deps.append(dep)
        
        return unique_deps
    
    async def _find_entry_points(
        self,
        repo_path: str,
        files: List[CodeFile]
    ) -> List[str]:
        """Find likely entry points in the codebase."""
        entry_points = []
        
        # Common entry point patterns
        entry_patterns = [
            "main.py", "app.py", "run.py", "train.py", "test.py",
            "index.py", "cli.py", "__main__.py",
            "main.js", "index.js", "app.js", "server.js",
            "main.ts", "index.ts", "app.ts"
        ]
        
        for file in files:
            filename = os.path.basename(file.path)
            
            # Check if it's a common entry point
            if filename in entry_patterns:
                entry_points.append(file.path)
                continue
            
            # Check for if __name__ == "__main__" in Python files
            if file.language == "python" and '__name__' in file.content:
                if '__main__' in file.content:
                    entry_points.append(file.path)
        
        # Check setup.py for console_scripts
        setup_path = os.path.join(repo_path, "setup.py")
        if os.path.exists(setup_path):
            with open(setup_path, 'r') as f:
                content = f.read()
                scripts_match = re.search(r'console_scripts.*?\[(.*?)\]', content, re.DOTALL)
                if scripts_match:
                    scripts = re.findall(r'["\']([^"\'=]+)=', scripts_match.group(1))
                    entry_points.extend(scripts)
        
        return list(set(entry_points))
    
    async def _read_readme(self, repo_path: str) -> str:
        """Read the README file."""
        readme_patterns = [
            "README.md", "README.rst", "README.txt", "README",
            "readme.md", "readme.rst", "readme.txt", "readme"
        ]
        
        for pattern in readme_patterns:
            readme_path = os.path.join(repo_path, pattern)
            if os.path.exists(readme_path):
                try:
                    with open(readme_path, 'r', encoding='utf-8') as f:
                        return f.read()
                except:
                    pass
        
        return ""
    
    def _calculate_language_stats(self, files: List[CodeFile]) -> Dict[str, int]:
        """Calculate lines of code per language."""
        stats = defaultdict(int)
        
        for file in files:
            line_count = len(file.content.split('\n'))
            stats[file.language] += line_count
        
        return dict(stats)
    
    def _build_structure_tree(self, repo_path: str) -> Dict[str, Any]:
        """Build a tree representation of the repository structure."""
        def build_tree(path: str, depth: int = 0) -> Dict[str, Any]:
            if depth > 3:  # Limit depth
                return {"type": "truncated"}
            
            result = {}
            
            try:
                items = sorted(os.listdir(path))
            except PermissionError:
                return {"type": "permission_denied"}
            
            for item in items:
                if item in self.IGNORE_PATTERNS or item.startswith('.'):
                    continue
                
                item_path = os.path.join(path, item)
                
                if os.path.isdir(item_path):
                    result[item] = build_tree(item_path, depth + 1)
                else:
                    ext = os.path.splitext(item)[1]
                    result[item] = {
                        "type": "file",
                        "language": self.LANGUAGE_MAP.get(ext, "other")
                    }
            
            return result
        
        return build_tree(repo_path)
    
    async def _enhance_with_llm(self, repo: AnalyzedRepository):
        """Use LLM to enhance repository understanding."""
        # Prepare analysis prompt
        file_summary = "\n".join([
            f"- {f.path}: {len(f.classes)} classes, {len(f.functions)} functions"
            for f in repo.files[:30]
        ])
        
        dep_summary = "\n".join([
            f"- {d.name} ({d.source})" + (f" {d.version}" if d.version else "")
            for d in repo.dependencies[:20]
        ])
        
        analysis_prompt = f"""
Analyze this GitHub repository and provide insights.

Repository: {repo.name}
Description: {repo.description}
Main Language: {repo.get_main_language()}
Languages: {json.dumps(repo.languages)}

Files Overview:
{file_summary}

Dependencies:
{dep_summary}

Entry Points: {', '.join(repo.entry_points)}

README (first 2000 chars):
{repo.readme_content[:2000]}

Provide analysis in JSON format:
{{
    "purpose": "What this repository does",
    "architecture": "Overall architecture pattern",
    "main_components": [
        {{"name": "component", "type": "class/module/service", "purpose": "what it does", "file": "file path"}}
    ],
    "key_algorithms": ["list of algorithms implemented"],
    "data_flow": "How data flows through the system",
    "compute_requirements": {{
        "cpu": "CPU requirements",
        "memory": "RAM estimate",
        "gpu": "GPU requirements if any",
        "storage": "Storage needs",
        "estimated_runtime": "Rough runtime estimate"
    }},
    "setup_complexity": "easy/medium/hard",
    "documentation_quality": "poor/fair/good/excellent",
    "test_coverage": "none/minimal/partial/comprehensive",
    "potential_issues": ["list of potential issues"],
    "suggested_tests": ["test scenarios to verify functionality"]
}}
"""
        
        try:
            analysis = await self.llm.generate_structured(
                analysis_prompt,
                schema={"type": "object"}
                # System instruction already set in config - uses specialized RepoAnalyzer prompt
            )
            
            repo.metadata["llm_analysis"] = analysis
            
        except Exception as e:
            error_msg = str(e)
            print(f"LLM repository analysis failed: {error_msg}")
            
            # Graceful degradation - provide basic analysis
            repo.metadata["llm_analysis"] = {
                "error": error_msg,
                "fallback": True,
                "purpose": repo.description or "Repository analysis unavailable",
                "architecture": "unknown",
                "main_components": [
                    {"name": f.path, "type": "file", "purpose": "unknown"}
                    for f in repo.files[:10] if f.classes or f.functions
                ],
                "key_algorithms": [],
                "compute_requirements": {
                    "cpu": "unknown",
                    "memory": "unknown",
                    "gpu": "check dependencies for torch/tensorflow",
                    "storage": "unknown"
                },
                "setup_complexity": "unknown",
                "documentation_quality": "check README",
                "suggested_tests": ["Manual testing recommended"]
            }
    
    async def _build_knowledge_graph(self, repo: AnalyzedRepository):
        """Build knowledge graph nodes and edges from repository."""
        kg = self.knowledge_graph
        
        # Add repository node
        repo_id = await kg.add_node(
            node_type=NodeType.REPOSITORY,
            name=repo.name,
            content=repo.description,
            metadata={
                "url": repo.url,
                "languages": repo.languages,
                "stars": repo.metadata.get("stars", 0)
            },
            source="repo_analyzer"
        )
        
        # Add file nodes (limit to important files)
        important_files = [f for f in repo.files if f.classes or f.functions][:50]
        
        for file in important_files:
            file_id = await kg.add_node(
                node_type=NodeType.FILE,
                name=file.path,
                content=file.content[:1000],
                metadata={
                    "language": file.language,
                    "size": file.size,
                    "imports": file.imports[:10]
                },
                source="repo_analyzer"
            )
            
            await kg.add_edge(
                repo_id, file_id,
                EdgeType.CONTAINS,
                created_by="repo_analyzer"
            )
            
            # Add class nodes
            for class_name in file.classes:
                class_id = await kg.add_node(
                    node_type=NodeType.CLASS,
                    name=class_name,
                    metadata={"file": file.path},
                    source="repo_analyzer"
                )
                await kg.add_edge(
                    file_id, class_id,
                    EdgeType.CONTAINS,
                    created_by="repo_analyzer"
                )
            
            # Add function nodes
            for func_name in file.functions[:20]:  # Limit functions
                func_id = await kg.add_node(
                    node_type=NodeType.FUNCTION,
                    name=func_name,
                    metadata={"file": file.path},
                    source="repo_analyzer"
                )
                await kg.add_edge(
                    file_id, func_id,
                    EdgeType.CONTAINS,
                    created_by="repo_analyzer"
                )
        
        # Add dependency nodes
        for dep in repo.dependencies[:30]:  # Limit dependencies
            dep_id = await kg.add_node(
                node_type=NodeType.DEPENDENCY,
                name=dep.name,
                metadata={
                    "version": dep.version,
                    "source": dep.source,
                    "required": dep.required
                },
                source="repo_analyzer"
            )
            await kg.add_edge(
                repo_id, dep_id,
                EdgeType.DEPENDS_ON,
                created_by="repo_analyzer"
            )
    
    async def map_concepts_to_code(
        self,
        concepts: List[Dict[str, Any]],
        repo_url: str
    ) -> List[Dict[str, Any]]:
        """
        Map paper concepts to code elements.
        
        Args:
            concepts: List of concepts from paper analysis
            repo_url: Repository URL
        
        Returns:
            List of mappings between concepts and code
        """
        if repo_url not in self.analyzed_repos:
            await self.analyze_repository(repo_url)
        
        repo = self.analyzed_repos[repo_url]
        kg = self.knowledge_graph
        
        mappings = []
        
        # Prepare code context for LLM
        code_elements = []
        for file in repo.files[:30]:
            for cls in file.classes:
                code_elements.append(f"Class: {cls} (in {file.path})")
            for func in file.functions[:10]:
                code_elements.append(f"Function: {func} (in {file.path})")
        
        # Ask LLM to map concepts to code
        mapping_prompt = f"""
Map the following scientific paper concepts to code elements in the repository.

Concepts from paper:
{json.dumps(concepts, indent=2)}

Available code elements:
{chr(10).join(code_elements)}

For each concept, find the most relevant code element that implements it.

Respond in JSON format:
{{
    "mappings": [
        {{
            "concept_name": "concept from paper",
            "code_element": "matching code element",
            "code_type": "class/function/module",
            "file_path": "path to file",
            "confidence": 0.0-1.0,
            "reasoning": "why this mapping makes sense"
        }}
    ]
}}
"""
        
        try:
            # Use the specialized concept mapper prompt
            from core.agent_prompts import get_agent_prompt, AgentType
            mapper_prompt = get_agent_prompt(AgentType.MAPPER)
            
            result = await self.llm.generate_structured(
                mapping_prompt,
                schema={"type": "object"},
                system_instruction=mapper_prompt  # Use ConceptMapper specialized prompt
            )
            
            mappings = result.get("mappings", [])
            
            # Create edges in knowledge graph for mappings
            for mapping in mappings:
                concept_name = mapping.get("concept_name", "")
                code_element = mapping.get("code_element", "")
                
                # Find concept node
                concept_nodes = kg.search(concept_name, [NodeType.CONCEPT], limit=1)
                
                # Find code node
                code_nodes = kg.search(
                    code_element,
                    [NodeType.CLASS, NodeType.FUNCTION, NodeType.MODULE],
                    limit=1
                )
                
                if concept_nodes and code_nodes:
                    await kg.add_edge(
                        concept_nodes[0][0].id,
                        code_nodes[0][0].id,
                        EdgeType.IMPLEMENTS,
                        weight=mapping.get("confidence", 0.5),
                        metadata={"reasoning": mapping.get("reasoning", "")},
                        created_by="repo_analyzer"
                    )
        
        except Exception as e:
            mappings = [{"error": str(e)}]
        
        return mappings
    
    async def estimate_compute_requirements(
        self,
        repo_url: str
    ) -> Dict[str, Any]:
        """Estimate compute requirements for running the repository."""
        if repo_url not in self.analyzed_repos:
            await self.analyze_repository(repo_url)
        
        repo = self.analyzed_repos[repo_url]
        analysis = repo.metadata.get("llm_analysis", {})
        
        # Base estimates
        estimates = {
            "cpu_cores": 2,
            "memory_gb": 4,
            "gpu_required": False,
            "gpu_memory_gb": 0,
            "storage_gb": 1,
            "estimated_runtime_minutes": 10
        }
        
        # Check for GPU-related dependencies
        gpu_deps = {"torch", "tensorflow", "jax", "cupy", "cuda"}
        for dep in repo.dependencies:
            if any(g in dep.name.lower() for g in gpu_deps):
                estimates["gpu_required"] = True
                estimates["gpu_memory_gb"] = 8
                estimates["memory_gb"] = 16
                break
        
        # Check for data science dependencies
        data_deps = {"pandas", "numpy", "scipy", "sklearn"}
        if any(d.name.lower() in data_deps for d in repo.dependencies):
            estimates["memory_gb"] = max(estimates["memory_gb"], 8)
        
        # Use LLM analysis if available
        if "compute_requirements" in analysis:
            llm_req = analysis["compute_requirements"]
            if "gpu" in str(llm_req.get("gpu", "")).lower():
                estimates["gpu_required"] = True
        
        return estimates
    
    def cleanup(self):
        """Clean up temporary directories."""
        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
        self.temp_dirs = []
    
    def get_repo_info(self, repo_url: str) -> Optional[Dict[str, Any]]:
        """Get stored information about an analyzed repository."""
        if repo_url in self.analyzed_repos:
            repo = self.analyzed_repos[repo_url]
            return {
                "name": repo.name,
                "url": repo.url,
                "description": repo.description,
                "languages": repo.languages,
                "main_language": repo.get_main_language(),
                "file_count": len(repo.files),
                "dependency_count": len(repo.dependencies),
                "entry_points": repo.entry_points,
                "analysis": repo.metadata.get("llm_analysis", {}),
                "local_path": repo.local_path
            }
        return None
