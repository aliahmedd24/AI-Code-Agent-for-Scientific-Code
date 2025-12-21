"""
Repository Analyzer Agent - GitHub Repository Analysis and Understanding

This agent is responsible for:
- Cloning and analyzing GitHub repositories
- Understanding codebase structure and architecture
- Identifying dependencies and requirements
- Estimating compute resources needed
- Mapping code to paper concepts via knowledge graph

ENHANCED: Now includes multi-signal semantic mapping for improved accuracy
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
from difflib import SequenceMatcher
import logging

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

logger = logging.getLogger(__name__)


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
    source: str = ""
    required: bool = True
    extras: List[str] = field(default_factory=list)


@dataclass
class AnalyzedRepository:
    """Complete analyzed repository structure."""
    name: str
    url: str
    description: str
    languages: Dict[str, int]
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


# =============================================================================
# ENHANCED SEMANTIC MAPPING (Integrated from semantic_mapper.py)
# =============================================================================

@dataclass
class MappingEvidence:
    """Evidence supporting a concept-to-code mapping."""
    evidence_type: str
    score: float
    detail: str


@dataclass
class ConceptCodeMapping:
    """A mapping between a paper concept and code element."""
    concept_name: str
    concept_type: str
    concept_description: str
    code_element: str
    code_type: str
    file_path: str
    confidence: float
    evidence: List[MappingEvidence] = field(default_factory=list)
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'concept_name': self.concept_name,
            'concept_type': self.concept_type,
            'concept_description': self.concept_description,
            'code_element': self.code_element,
            'code_type': self.code_type,
            'file_path': self.file_path,
            'confidence': self.confidence,
            'evidence': [
                {'type': e.evidence_type, 'score': e.score, 'detail': e.detail}
                for e in self.evidence
            ],
            'reasoning': self.reasoning
        }


class LexicalMatcher:
    """Match concepts to code using lexical similarity."""
    
    STOP_TERMS = {
        'model', 'data', 'train', 'test', 'input', 'output', 'process',
        'compute', 'calculate', 'get', 'set', 'init', 'run', 'execute',
        'load', 'save', 'read', 'write', 'create', 'update', 'delete',
        'main', 'helper', 'utils', 'util', 'base', 'config', 'params'
    }
    
    def extract_terms(self, text: str) -> Set[str]:
        """Extract meaningful terms from text."""
        words = re.findall(r'[a-zA-Z][a-z]*|[A-Z]+(?=[A-Z][a-z]|\b)', text)
        return {w.lower() for w in words if len(w) > 2 and w.lower() not in self.STOP_TERMS}
    
    def compute_similarity(self, concept: str, concept_desc: str, code_name: str, code_docs: str) -> Tuple[float, str]:
        """Compute lexical similarity."""
        concept_terms = self.extract_terms(concept + " " + concept_desc)
        code_terms = self.extract_terms(code_name + " " + code_docs)
        
        if not concept_terms or not code_terms:
            return 0.0, "No meaningful terms"
        
        overlap = concept_terms & code_terms
        if not overlap:
            return 0.0, "No term overlap"
        
        union = concept_terms | code_terms
        similarity = len(overlap) / len(union)
        
        # Bonus for name match
        concept_name_terms = self.extract_terms(concept)
        code_name_terms = self.extract_terms(code_name)
        if concept_name_terms & code_name_terms:
            similarity = min(similarity + 0.2, 1.0)
        
        return similarity, f"Matching terms: {', '.join(sorted(overlap)[:5])}"
    
    def compute_name_similarity(self, concept: str, code_name: str) -> Tuple[float, str]:
        """Compute direct name similarity."""
        concept_normalized = concept.lower().replace(' ', '').replace('_', '').replace('-', '')
        code_normalized = code_name.lower().replace('_', '').replace('-', '')
        
        if concept_normalized == code_normalized:
            return 1.0, f"Exact name match: {concept} = {code_name}"
        
        if concept_normalized in code_normalized or code_normalized in concept_normalized:
            return 0.7, f"Substring match: {concept} ≈ {code_name}"
        
        ratio = SequenceMatcher(None, concept_normalized, code_normalized).ratio()
        if ratio > 0.6:
            return ratio, f"Name similarity: {ratio:.2f}"
        
        # Abbreviation check
        concept_abbrev = ''.join(w[0] for w in concept.split() if w)
        if concept_abbrev.lower() == code_normalized[:len(concept_abbrev)].lower():
            return 0.6, f"Abbreviation match: {concept_abbrev} → {code_name}"
        
        return 0.0, "No name similarity"


class SemanticMatcher:
    """Match concepts to code using semantic embeddings."""
    
    def __init__(self):
        self.model = None
        self._model_loaded = False
    
    def _ensure_model(self):
        if self._model_loaded:
            return self.model is not None
        
        self._model_loaded = True
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded sentence-transformers model")
            return True
        except ImportError:
            logger.warning("sentence-transformers not installed. Semantic matching disabled.")
            return False
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return False
    
    def compute_similarity(self, concept: str, concept_desc: str, code_name: str, code_docs: str) -> Tuple[float, str]:
        """Compute semantic similarity."""
        if not self._ensure_model():
            return 0.0, "Semantic model not available"
        
        try:
            import numpy as np
            
            concept_text = f"{concept}: {concept_desc}"
            code_text = f"{code_name}: {code_docs}"
            
            emb1 = self.model.encode(concept_text)
            emb2 = self.model.encode(code_text)
            
            similarity = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
            return max(similarity, 0.0), f"Semantic similarity: {similarity:.3f}"
            
        except Exception as e:
            logger.error(f"Semantic similarity failed: {e}")
            return 0.0, f"Error: {e}"


class StructuralMatcher:
    """Match concepts to code using AST patterns."""
    
    ALGORITHM_PATTERNS = {
        'attention': {'softmax', 'dot_product', 'transpose', 'matmul'},
        'normalization': {'mean', 'std', 'sqrt', 'eps'},
        'convolution': {'kernel', 'stride', 'padding', 'conv'},
        'transformer': {'attention', 'encoder', 'decoder', 'positional'},
        'embedding': {'lookup', 'index', 'vocab', 'embed'},
    }
    
    def extract_patterns(self, code: str) -> Set[str]:
        """Extract patterns from code."""
        patterns = set()
        code_lower = code.lower()
        
        # Function calls and operations
        patterns.update(re.findall(r'\b(\w+)\s*\(', code_lower))
        patterns.update(re.findall(r'\.(\w+)\s*\(', code_lower))
        
        return patterns
    
    def match_pattern(self, concept: str, code: str) -> Tuple[float, str]:
        """Match concept to code patterns."""
        concept_lower = concept.lower()
        code_patterns = self.extract_patterns(code)
        
        for pattern_name, required_ops in self.ALGORITHM_PATTERNS.items():
            if pattern_name in concept_lower:
                matches = sum(1 for op in required_ops if any(op in p for p in code_patterns))
                if matches >= 2:
                    return 0.7, f"Matches {pattern_name} pattern with {matches} indicators"
                elif matches >= 1:
                    return 0.4, f"Partial {pattern_name} pattern match"
        
        return 0.0, "No structural pattern match"


class DocumentaryMatcher:
    """Match concepts to code using docstrings."""
    
    def compute_similarity(self, concept: str, concept_desc: str, docstring: str) -> Tuple[float, str]:
        """Compute similarity based on documentation."""
        if not docstring:
            return 0.0, "No documentation"
        
        doc_text = docstring.lower()
        concept_terms = set(concept.lower().split()) | set(concept_desc.lower().split())
        
        stop_words = {'the', 'a', 'an', 'is', 'are', 'for', 'to', 'of', 'and', 'in', 'on', 'this', 'that'}
        concept_terms = {t for t in concept_terms if t not in stop_words and len(t) > 2}
        
        if not concept_terms:
            return 0.0, "No meaningful concept terms"
        
        matches = [t for t in concept_terms if t in doc_text]
        
        if not matches:
            return 0.0, "No documentation mentions"
        
        score = len(matches) / len(concept_terms)
        
        if concept.lower() in doc_text:
            score = min(score + 0.3, 1.0)
            return score, f"Concept explicitly mentioned: '{concept}'"
        
        return score, f"Documentation mentions: {', '.join(matches[:3])}"


class EnhancedSemanticMapper:
    """Multi-signal concept-to-code mapper."""
    
    SIGNAL_WEIGHTS = {
        'lexical': 0.20,
        'semantic': 0.30,
        'structural': 0.25,
        'documentary': 0.25
    }
    
    MIN_CONFIDENCE = 0.25
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.lexical = LexicalMatcher()
        self.semantic = SemanticMatcher()
        self.structural = StructuralMatcher()
        self.documentary = DocumentaryMatcher()
    
    def map_single(self, concept: Dict, code_file: CodeFile, element_name: str, element_type: str) -> Optional[ConceptCodeMapping]:
        """Map a single concept to a code element."""
        concept_name = concept.get('name', '')
        concept_desc = concept.get('description', '')
        
        # Get docstring
        docstring = '\n'.join(code_file.docstrings) if code_file.docstrings else ''
        
        evidence = []
        
        # Lexical matching
        lex_score, lex_detail = self.lexical.compute_similarity(
            concept_name, concept_desc, element_name, docstring
        )
        name_score, name_detail = self.lexical.compute_name_similarity(concept_name, element_name)
        lex_score = max(lex_score, name_score)
        lex_detail = name_detail if name_score > lex_score else lex_detail
        
        if lex_score > 0:
            evidence.append(MappingEvidence('lexical', lex_score, lex_detail))
        
        # Semantic matching
        sem_score, sem_detail = self.semantic.compute_similarity(
            concept_name, concept_desc, element_name, docstring
        )
        if sem_score > 0.3:
            evidence.append(MappingEvidence('semantic', sem_score, sem_detail))
        
        # Structural matching
        struct_score, struct_detail = self.structural.match_pattern(
            concept_name, code_file.content
        )
        if struct_score > 0:
            evidence.append(MappingEvidence('structural', struct_score, struct_detail))
        
        # Documentary matching
        doc_score, doc_detail = self.documentary.compute_similarity(
            concept_name, concept_desc, docstring
        )
        if doc_score > 0:
            evidence.append(MappingEvidence('documentary', doc_score, doc_detail))
        
        # Compute weighted score
        if not evidence:
            return None
        
        total_score = 0.0
        total_weight = 0.0
        for e in evidence:
            weight = self.SIGNAL_WEIGHTS.get(e.evidence_type, 0.1)
            total_score += e.score * weight
            total_weight += weight
        
        if total_weight > 0:
            final_score = total_score / total_weight
            if len(evidence) >= 3:
                final_score = min(final_score * 1.2, 1.0)
            elif len(evidence) >= 2:
                final_score = min(final_score * 1.1, 1.0)
        else:
            return None
        
        if final_score < self.MIN_CONFIDENCE:
            return None
        
        # Generate reasoning
        evidence_strs = []
        for e in sorted(evidence, key=lambda x: x.score, reverse=True):
            evidence_strs.append(f"{e.evidence_type}: {e.detail}")
        reasoning = f"Mapped based on: {'; '.join(evidence_strs[:3])}"
        
        return ConceptCodeMapping(
            concept_name=concept_name,
            concept_type=concept.get('type', 'concept'),
            concept_description=concept_desc,
            code_element=element_name,
            code_type=element_type,
            file_path=code_file.path,
            confidence=final_score,
            evidence=evidence,
            reasoning=reasoning
        )


# =============================================================================
# REPO ANALYZER AGENT
# =============================================================================

class RepoAnalyzerAgent:
    """
    Agent for analyzing GitHub repositories.
    
    ENHANCED: Now includes multi-signal semantic mapping for concept-to-code matching
    """
    
    LANGUAGE_EXTENSIONS = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.hpp': 'cpp',
        '.rs': 'rust',
        '.go': 'go',
        '.rb': 'ruby',
        '.r': 'r',
        '.R': 'r',
        '.jl': 'julia',
        '.scala': 'scala',
        '.kt': 'kotlin',
        '.swift': 'swift',
        '.m': 'matlab',
        '.ipynb': 'jupyter'
    }
    
    IGNORE_PATTERNS = {
        'node_modules', '__pycache__', '.git', '.svn', '.hg',
        'venv', 'env', '.env', 'build', 'dist', 'target',
        '.idea', '.vscode', '.vs', 'vendor', 'third_party',
        'docs', 'doc', 'examples', 'tests', 'test', 'spec',
        '.tox', '.pytest_cache', '.mypy_cache', 'coverage',
        'htmlcov', '.coverage', '*.egg-info'
    }
    
    def __init__(
        self,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        gemini_api_key: Optional[str] = None,
        github_token: Optional[str] = None
    ):
        self.knowledge_graph = knowledge_graph or get_global_graph()
        
        system_prompt = get_agent_prompt(AgentType.REPO_ANALYZER)
        
        self.llm = AgentLLM(
            agent_name="RepoAnalyzer",
            agent_role="Repository structure and code analysis",
            api_key=gemini_api_key,
            config=GeminiConfig(
                model=GeminiModel.FLASH,
                temperature=0.4,
                max_output_tokens=8192,
                system_instruction=system_prompt
            )
        )
        
        self.github_token = github_token
        self.github = Github(github_token) if github_token else None
        self.analyzed_repos: Dict[str, AnalyzedRepository] = {}
        
        # ENHANCED: Semantic mapper
        self.semantic_mapper = EnhancedSemanticMapper(llm_client=self.llm)
    
    async def analyze_repository(self, repo_url: str) -> AnalyzedRepository:
        """Analyze a GitHub repository."""
        # Clone repository
        repo_path = await self._clone_repository(repo_url)
        
        # Extract basic info
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        
        # Analyze files
        files = await self._analyze_files(repo_path)
        
        # Extract dependencies
        dependencies = await self._extract_dependencies(repo_path)
        
        # Find entry points
        entry_points = await self._find_entry_points(repo_path, files)
        
        # Read README
        readme = await self._read_readme(repo_path)
        
        # Calculate language stats
        languages = self._calculate_language_stats(files)
        
        # Build structure tree
        structure = self._build_structure_tree(repo_path)
        
        # Create repository object
        repo = AnalyzedRepository(
            name=repo_name,
            url=repo_url,
            description=readme[:500] if readme else "",
            languages=languages,
            files=files,
            dependencies=dependencies,
            entry_points=entry_points,
            readme_content=readme,
            structure=structure,
            local_path=repo_path
        )
        
        # LLM analysis
        await self._analyze_with_llm(repo)
        
        # Build knowledge graph
        await self._build_knowledge_graph(repo)
        
        self.analyzed_repos[repo_url] = repo
        
        return repo
    
    async def _clone_repository(self, repo_url: str) -> str:
        """Clone a GitHub repository."""
        temp_dir = tempfile.mkdtemp(prefix="repo_")
        
        clone_url = repo_url
        if self.github_token and 'github.com' in repo_url:
            clone_url = repo_url.replace('https://', f'https://{self.github_token}@')
        
        try:
            git.Repo.clone_from(
                clone_url,
                temp_dir,
                depth=1,
                single_branch=True
            )
        except Exception as e:
            logger.warning(f"Git clone failed: {e}, trying without auth")
            git.Repo.clone_from(repo_url, temp_dir, depth=1, single_branch=True)
        
        return temp_dir
    
    async def _analyze_files(self, repo_path: str, max_files: int = 200) -> List[CodeFile]:
        """Analyze code files in the repository."""
        files = []
        
        for root, dirs, filenames in os.walk(repo_path):
            # Filter directories
            dirs[:] = [d for d in dirs if d not in self.IGNORE_PATTERNS and not d.startswith('.')]
            
            for filename in filenames:
                if len(files) >= max_files:
                    break
                
                ext = os.path.splitext(filename)[1]
                if ext not in self.LANGUAGE_EXTENSIONS:
                    continue
                
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, repo_path)
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    if len(content) > 100000:
                        content = content[:100000]
                    
                    code_file = await self._analyze_single_file(
                        rel_path,
                        self.LANGUAGE_EXTENSIONS[ext],
                        content
                    )
                    files.append(code_file)
                    
                except Exception as e:
                    logger.warning(f"Failed to analyze {rel_path}: {e}")
        
        return files
    
    async def _analyze_single_file(self, path: str, language: str, content: str) -> CodeFile:
        """Analyze a single code file."""
        imports = []
        classes = []
        functions = []
        docstrings = []
        
        if language == "python":
            imports = re.findall(r'^(?:from\s+(\S+)\s+)?import\s+(\S+)', content, re.MULTILINE)
            imports = [f"{i[0]}.{i[1]}" if i[0] else i[1] for i in imports]
            classes = re.findall(r'class\s+(\w+)', content)
            functions = re.findall(r'def\s+(\w+)', content)
            docstrings = re.findall(r'"""(.*?)"""', content, re.DOTALL)[:5]
            
        elif language in ("javascript", "typescript"):
            imports = re.findall(r"(?:import|require)\s*\(?['\"]([^'\"]+)['\"]", content)
            classes = re.findall(r'class\s+(\w+)', content)
            functions = re.findall(r'(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s*)?\()', content)
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
        """Extract dependencies from package files."""
        dependencies = []
        
        # Python - requirements.txt
        req_path = os.path.join(repo_path, "requirements.txt")
        if os.path.exists(req_path):
            with open(req_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        match = re.match(r'^([a-zA-Z0-9_-]+)([<>=!]+)?(.+)?$', line)
                        if match:
                            dependencies.append(Dependency(
                                name=match.group(1),
                                version=match.group(3) if match.group(2) else None,
                                source="pip"
                            ))
        
        # Python - setup.py
        setup_path = os.path.join(repo_path, "setup.py")
        if os.path.exists(setup_path):
            with open(setup_path, 'r') as f:
                content = f.read()
                requires_match = re.search(r'install_requires\s*=\s*\[(.*?)\]', content, re.DOTALL)
                if requires_match:
                    deps = re.findall(r'["\']([^"\']+)["\']', requires_match.group(1))
                    for dep in deps:
                        name = re.split(r'[<>=!]', dep)[0]
                        if name not in [d.name for d in dependencies]:
                            dependencies.append(Dependency(name=name, source="pip"))
        
        # JavaScript - package.json
        pkg_path = os.path.join(repo_path, "package.json")
        if os.path.exists(pkg_path):
            with open(pkg_path, 'r') as f:
                try:
                    pkg = json.load(f)
                    for dep_type in ['dependencies', 'devDependencies']:
                        for name, version in pkg.get(dep_type, {}).items():
                            dependencies.append(Dependency(
                                name=name,
                                version=version,
                                source="npm",
                                required=(dep_type == 'dependencies')
                            ))
                except json.JSONDecodeError:
                    pass
        
        return dependencies
    
    async def _find_entry_points(self, repo_path: str, files: List[CodeFile]) -> List[str]:
        """Find entry points in the repository."""
        entry_points = []
        
        entry_patterns = [
            "main.py", "app.py", "run.py", "train.py", "test.py",
            "index.py", "cli.py", "__main__.py",
            "main.js", "index.js", "app.js", "server.js"
        ]
        
        for file in files:
            filename = os.path.basename(file.path)
            
            if filename in entry_patterns:
                entry_points.append(file.path)
                continue
            
            if file.language == "python" and '__name__' in file.content and '__main__' in file.content:
                entry_points.append(file.path)
        
        return list(set(entry_points))
    
    async def _read_readme(self, repo_path: str) -> str:
        """Read README file."""
        patterns = ["README.md", "README.rst", "README.txt", "README", "readme.md"]
        
        for pattern in patterns:
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
            stats[file.language] += len(file.content.split('\n'))
        return dict(stats)
    
    def _build_structure_tree(self, repo_path: str, depth: int = 0) -> Dict[str, Any]:
        """Build structure tree."""
        if depth > 3:
            return {"type": "truncated"}
        
        result = {}
        try:
            items = sorted(os.listdir(repo_path))
        except:
            return {"type": "error"}
        
        for item in items[:50]:
            if item in self.IGNORE_PATTERNS or item.startswith('.'):
                continue
            
            item_path = os.path.join(repo_path, item)
            if os.path.isdir(item_path):
                result[item] = self._build_structure_tree(item_path, depth + 1)
            else:
                result[item] = {"type": "file", "size": os.path.getsize(item_path)}
        
        return result
    
    async def _analyze_with_llm(self, repo: AnalyzedRepository):
        """Analyze repository with LLM."""
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
Main Language: {repo.get_main_language()}

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
    "compute_requirements": {{
        "cpu": "CPU requirements",
        "memory": "RAM estimate",
        "gpu": "GPU requirements if any"
    }},
    "suggested_tests": ["test scenarios to verify functionality"]
}}
"""
        
        try:
            analysis = await self.llm.generate_structured(analysis_prompt, schema={"type": "object"})
            repo.metadata["llm_analysis"] = analysis
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            repo.metadata["llm_analysis"] = {"error": str(e), "fallback": True}
    
    async def _build_knowledge_graph(self, repo: AnalyzedRepository):
        """Build knowledge graph from repository."""
        kg = self.knowledge_graph
        
        repo_id = await kg.add_node(
            node_type=NodeType.REPOSITORY,
            name=repo.name,
            content=repo.description,
            metadata={
                "url": repo.url,
                "language": repo.get_main_language(),
                "file_count": len(repo.files)
            },
            source="repo_analyzer"
        )
        
        for file in repo.files[:50]:
            file_id = await kg.add_node(
                node_type=NodeType.FILE,
                name=file.path,
                content=file.content[:2000],
                metadata={
                    "language": file.language,
                    "size": file.size,
                    "imports": file.imports[:10]
                },
                source="repo_analyzer"
            )
            await kg.add_edge(repo_id, file_id, EdgeType.CONTAINS, created_by="repo_analyzer")
            
            for cls in file.classes[:10]:
                cls_id = await kg.add_node(
                    node_type=NodeType.CLASS,
                    name=cls,
                    metadata={"file": file.path},
                    source="repo_analyzer"
                )
                await kg.add_edge(file_id, cls_id, EdgeType.CONTAINS, created_by="repo_analyzer")
            
            for func in file.functions[:20]:
                func_id = await kg.add_node(
                    node_type=NodeType.FUNCTION,
                    name=func,
                    metadata={"file": file.path},
                    source="repo_analyzer"
                )
                await kg.add_edge(file_id, func_id, EdgeType.CONTAINS, created_by="repo_analyzer")
        
        for dep in repo.dependencies[:30]:
            dep_id = await kg.add_node(
                node_type=NodeType.DEPENDENCY,
                name=dep.name,
                metadata={"version": dep.version, "source": dep.source},
                source="repo_analyzer"
            )
            await kg.add_edge(repo_id, dep_id, EdgeType.DEPENDS_ON, created_by="repo_analyzer")
    
    async def map_concepts_to_code(
        self,
        concepts: List[Dict[str, Any]],
        repo_url: str
    ) -> List[Dict[str, Any]]:
        """
        ENHANCED: Map paper concepts to code using multi-signal semantic matching.
        """
        if repo_url not in self.analyzed_repos:
            await self.analyze_repository(repo_url)
        
        repo = self.analyzed_repos[repo_url]
        all_mappings: List[ConceptCodeMapping] = []
        
        for concept in concepts:
            concept_mappings = []
            
            for file in repo.files[:50]:
                # Map to classes
                for cls in file.classes:
                    mapping = self.semantic_mapper.map_single(
                        concept, file, cls, 'class'
                    )
                    if mapping:
                        concept_mappings.append(mapping)
                
                # Map to functions
                for func in file.functions[:15]:
                    mapping = self.semantic_mapper.map_single(
                        concept, file, func, 'function'
                    )
                    if mapping:
                        concept_mappings.append(mapping)
            
            # Sort by confidence and take top matches
            concept_mappings.sort(key=lambda m: m.confidence, reverse=True)
            all_mappings.extend(concept_mappings[:3])
        
        # Sort all by confidence
        all_mappings.sort(key=lambda m: m.confidence, reverse=True)
        
        # Store mappings in knowledge graph
        kg = self.knowledge_graph
        for mapping in all_mappings:
            await kg.add_edge(
                mapping.concept_name,
                mapping.code_element,
                EdgeType.IMPLEMENTS,
                weight=mapping.confidence,
                metadata={
                    'evidence': [e.evidence_type for e in mapping.evidence],
                    'reasoning': mapping.reasoning
                },
                created_by="repo_analyzer"
            )
        
        return [m.to_dict() for m in all_mappings]
    
    def get_repo_info(self, repo_url: str) -> Optional[Dict[str, Any]]:
        """Get repository info."""
        if repo_url in self.analyzed_repos:
            repo = self.analyzed_repos[repo_url]
            return {
                "name": repo.name,
                "url": repo.url,
                "main_language": repo.get_main_language(),
                "languages": repo.languages,
                "file_count": len(repo.files),
                "dependency_count": len(repo.dependencies),
                "entry_points": repo.entry_points,
                "analysis": repo.metadata.get("llm_analysis", {}),
                "local_path": repo.local_path
            }
        return None

    def cleanup(self):
        """Clean up any temporary resources."""
        # Clean up temporary directories for cloned repos
        for repo in self.analyzed_repos.values():
            if repo.local_path and os.path.exists(repo.local_path):
                try:
                    import shutil
                    shutil.rmtree(repo.local_path, ignore_errors=True)
                    logger.info(f"Cleaned up repo: {repo.local_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {repo.local_path}: {e}")