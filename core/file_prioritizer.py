"""
Smart File Prioritization - Intelligent File Selection for Analysis

MEDIUM-PRIORITY ENHANCEMENT:
- Prioritize files based on import centrality
- Focus on entry points and core modules
- Skip test/example files unless relevant
- Adaptive file limit based on repository size
"""

import os
import re
import ast
from typing import Optional, Any, Dict, List, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FileScore:
    """Score and metadata for a file."""
    path: str
    total_score: float
    scores: Dict[str, float] = field(default_factory=dict)
    imports: List[str] = field(default_factory=list)
    imported_by: List[str] = field(default_factory=list)
    classes: int = 0
    functions: int = 0
    lines: int = 0
    is_entry_point: bool = False
    is_test: bool = False
    is_example: bool = False
    
    def __lt__(self, other: 'FileScore') -> bool:
        return self.total_score < other.total_score


@dataclass
class ImportGraph:
    """Graph of imports between files."""
    edges: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    reverse_edges: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    
    def add_import(self, importer: str, imported: str):
        """Add an import edge."""
        self.edges[importer].add(imported)
        self.reverse_edges[imported].add(importer)
    
    def get_imports(self, file: str) -> Set[str]:
        """Get files imported by a file."""
        return self.edges.get(file, set())
    
    def get_importers(self, file: str) -> Set[str]:
        """Get files that import a file."""
        return self.reverse_edges.get(file, set())
    
    def compute_centrality(self) -> Dict[str, float]:
        """Compute import centrality (how central a file is in the import graph)."""
        centrality = {}
        all_files = set(self.edges.keys()) | set(self.reverse_edges.keys())
        
        for file in all_files:
            # Combine in-degree and out-degree with weighting
            in_degree = len(self.reverse_edges.get(file, set()))
            out_degree = len(self.edges.get(file, set()))
            
            # Files that are imported more are more central
            centrality[file] = in_degree * 2 + out_degree
        
        # Normalize to 0-1
        if centrality:
            max_val = max(centrality.values())
            if max_val > 0:
                centrality = {k: v / max_val for k, v in centrality.items()}
        
        return centrality


# =============================================================================
# FILE PRIORITIZER
# =============================================================================

class SmartFilePrioritizer:
    """
    Intelligently prioritize files for analysis.
    
    MEDIUM-PRIORITY ENHANCEMENT:
    - Goes beyond simple "analyze first N files"
    - Considers import relationships
    - Identifies core vs peripheral code
    - Adapts to repository structure
    """
    
    # File patterns for classification
    ENTRY_POINT_PATTERNS = [
        r'^main\.py$',
        r'^app\.py$',
        r'^run\.py$',
        r'^cli\.py$',
        r'^server\.py$',
        r'^__main__\.py$',
        r'^index\.[jt]s$',
        r'^main\.[jt]s$',
    ]
    
    TEST_PATTERNS = [
        r'^test_',
        r'_test\.py$',
        r'^tests?/',
        r'spec\.py$',
        r'\.spec\.[jt]s$',
        r'\.test\.[jt]s$',
    ]
    
    EXAMPLE_PATTERNS = [
        r'^examples?/',
        r'^demos?/',
        r'^samples?/',
        r'^tutorials?/',
        r'example',
        r'demo\.py$',
    ]
    
    CONFIG_PATTERNS = [
        r'^config',
        r'settings\.py$',
        r'constants\.py$',
        r'^setup\.py$',
        r'conftest\.py$',
    ]
    
    CORE_PATTERNS = [
        r'^src/',
        r'^lib/',
        r'^core/',
        r'^models?/',
        r'^utils?/',
        r'^agents?/',
    ]
    
    IGNORE_PATTERNS = [
        r'__pycache__',
        r'\.pyc$',
        r'node_modules',
        r'\.git/',
        r'\.egg-info',
        r'dist/',
        r'build/',
        r'venv/',
        r'\.env',
    ]
    
    def __init__(
        self,
        max_files: int = 50,
        include_tests: bool = False,
        include_examples: bool = False,
        language_filter: Optional[List[str]] = None
    ):
        self.max_files = max_files
        self.include_tests = include_tests
        self.include_examples = include_examples
        self.language_filter = language_filter or ['python', 'javascript', 'typescript']
        
        self.import_graph = ImportGraph()
        self.file_scores: Dict[str, FileScore] = {}
    
    def prioritize(self, repo_path: str) -> List[str]:
        """
        Analyze repository and return prioritized list of files.
        
        Returns file paths sorted by importance.
        """
        # Reset state
        self.import_graph = ImportGraph()
        self.file_scores = {}
        
        # Collect all code files
        code_files = self._collect_code_files(repo_path)
        
        if not code_files:
            logger.warning("No code files found in repository")
            return []
        
        # Analyze each file
        for file_path in code_files:
            self._analyze_file(repo_path, file_path)
        
        # Build import graph
        self._build_import_graph(repo_path, code_files)
        
        # Compute centrality scores
        centrality = self.import_graph.compute_centrality()
        
        # Calculate final scores
        for file_path, score in self.file_scores.items():
            self._calculate_final_score(score, centrality)
        
        # Filter and sort
        prioritized = self._filter_and_sort()
        
        logger.info(f"Prioritized {len(prioritized)} files from {len(code_files)} total")
        
        return prioritized
    
    def _collect_code_files(self, repo_path: str) -> List[str]:
        """Collect all code files from repository."""
        code_files = []
        
        extensions = {
            'python': ['.py'],
            'javascript': ['.js', '.jsx'],
            'typescript': ['.ts', '.tsx'],
        }
        
        valid_extensions = set()
        for lang in self.language_filter:
            valid_extensions.update(extensions.get(lang, []))
        
        for root, dirs, files in os.walk(repo_path):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if not self._matches_patterns(d, self.IGNORE_PATTERNS)]
            
            for filename in files:
                ext = os.path.splitext(filename)[1]
                if ext not in valid_extensions:
                    continue
                
                file_path = os.path.relpath(os.path.join(root, filename), repo_path)
                
                if self._matches_patterns(file_path, self.IGNORE_PATTERNS):
                    continue
                
                code_files.append(file_path)
        
        return code_files
    
    def _analyze_file(self, repo_path: str, file_path: str):
        """Analyze a single file."""
        full_path = os.path.join(repo_path, file_path)
        
        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.debug(f"Could not read {file_path}: {e}")
            return
        
        # Create score object
        score = FileScore(
            path=file_path,
            total_score=0.0,
            lines=len(content.split('\n')),
            is_entry_point=self._matches_patterns(file_path, self.ENTRY_POINT_PATTERNS),
            is_test=self._matches_patterns(file_path, self.TEST_PATTERNS),
            is_example=self._matches_patterns(file_path, self.EXAMPLE_PATTERNS),
        )
        
        # Count classes and functions
        if file_path.endswith('.py'):
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        score.classes += 1
                    elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                        score.functions += 1
            except:
                pass
            
            # Check for __main__ block
            if '__name__' in content and '__main__' in content:
                score.is_entry_point = True
        
        # Extract imports for Python
        if file_path.endswith('.py'):
            score.imports = self._extract_python_imports(content)
        elif file_path.endswith(('.js', '.ts', '.jsx', '.tsx')):
            score.imports = self._extract_js_imports(content)
        
        self.file_scores[file_path] = score
    
    def _extract_python_imports(self, content: str) -> List[str]:
        """Extract imports from Python code."""
        imports = []
        
        # Standard imports
        for match in re.finditer(r'^import\s+(\S+)', content, re.MULTILINE):
            imports.append(match.group(1).split('.')[0])
        
        # From imports
        for match in re.finditer(r'^from\s+(\S+)\s+import', content, re.MULTILINE):
            module = match.group(1)
            if not module.startswith('.'):
                imports.append(module.split('.')[0])
            else:
                # Relative import
                imports.append(module)
        
        return list(set(imports))
    
    def _extract_js_imports(self, content: str) -> List[str]:
        """Extract imports from JavaScript/TypeScript code."""
        imports = []
        
        # ES6 imports
        for match in re.finditer(r"import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]", content):
            imports.append(match.group(1))
        
        # Require
        for match in re.finditer(r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", content):
            imports.append(match.group(1))
        
        return list(set(imports))
    
    def _build_import_graph(self, repo_path: str, code_files: List[str]):
        """Build the import graph between files."""
        # Map module names to file paths
        module_to_file: Dict[str, str] = {}
        
        for file_path in code_files:
            # Convert file path to module name
            if file_path.endswith('.py'):
                module = file_path[:-3].replace('/', '.').replace('\\', '.')
                if module.endswith('.__init__'):
                    module = module[:-9]
                module_to_file[module] = file_path
                
                # Also map just the filename
                base = os.path.splitext(os.path.basename(file_path))[0]
                if base not in module_to_file:
                    module_to_file[base] = file_path
        
        # Build edges
        for file_path, score in self.file_scores.items():
            for imp in score.imports:
                # Try to find the imported module
                if imp in module_to_file:
                    imported_file = module_to_file[imp]
                    if imported_file != file_path:
                        self.import_graph.add_import(file_path, imported_file)
                        
                        # Update reverse relationship
                        if imported_file in self.file_scores:
                            self.file_scores[imported_file].imported_by.append(file_path)
    
    def _calculate_final_score(self, score: FileScore, centrality: Dict[str, float]):
        """Calculate the final priority score for a file."""
        scores = {}
        
        # Base score from file characteristics (0-10)
        scores['base'] = min(10, (score.classes * 2 + score.functions * 0.5 + score.lines * 0.01))
        
        # Entry point bonus (0 or 15)
        scores['entry_point'] = 15 if score.is_entry_point else 0
        
        # Centrality score (0-10)
        scores['centrality'] = centrality.get(score.path, 0) * 10
        
        # Import depth score (files that import many others are more important) (0-5)
        scores['imports'] = min(5, len(score.imports) * 0.5)
        
        # Being imported score (files that are imported are more important) (0-10)
        scores['imported_by'] = min(10, len(score.imported_by) * 2)
        
        # Location bonus (core directories) (0 or 5)
        if self._matches_patterns(score.path, self.CORE_PATTERNS):
            scores['location'] = 5
        else:
            scores['location'] = 0
        
        # Test/example penalty
        if score.is_test:
            scores['test_penalty'] = -20 if not self.include_tests else 0
        else:
            scores['test_penalty'] = 0
        
        if score.is_example:
            scores['example_penalty'] = -15 if not self.include_examples else 0
        else:
            scores['example_penalty'] = 0
        
        # Config file bonus (sometimes useful but not primary)
        if self._matches_patterns(score.path, self.CONFIG_PATTERNS):
            scores['config'] = 3
        else:
            scores['config'] = 0
        
        # Calculate total
        score.scores = scores
        score.total_score = sum(scores.values())
    
    def _filter_and_sort(self) -> List[str]:
        """Filter and sort files by priority."""
        # Filter
        filtered = []
        for path, score in self.file_scores.items():
            # Skip tests if not included
            if score.is_test and not self.include_tests:
                continue
            
            # Skip examples if not included
            if score.is_example and not self.include_examples:
                continue
            
            # Skip very small files (likely empty or just imports)
            if score.lines < 5:
                continue
            
            filtered.append(score)
        
        # Sort by total score (descending)
        filtered.sort(key=lambda s: s.total_score, reverse=True)
        
        # Return top N paths
        return [s.path for s in filtered[:self.max_files]]
    
    def _matches_patterns(self, text: str, patterns: List[str]) -> bool:
        """Check if text matches any pattern."""
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def get_file_analysis(self, file_path: str) -> Optional[FileScore]:
        """Get analysis for a specific file."""
        return self.file_scores.get(file_path)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of file prioritization."""
        if not self.file_scores:
            return {}
        
        scores = list(self.file_scores.values())
        
        return {
            'total_files': len(scores),
            'entry_points': sum(1 for s in scores if s.is_entry_point),
            'test_files': sum(1 for s in scores if s.is_test),
            'example_files': sum(1 for s in scores if s.is_example),
            'avg_score': sum(s.total_score for s in scores) / len(scores),
            'max_score': max(s.total_score for s in scores),
            'import_edges': sum(len(e) for e in self.import_graph.edges.values()),
            'most_imported': self._get_most_imported(5),
            'most_importing': self._get_most_importing(5),
        }
    
    def _get_most_imported(self, n: int) -> List[Tuple[str, int]]:
        """Get the most imported files."""
        import_counts = [
            (path, len(score.imported_by))
            for path, score in self.file_scores.items()
        ]
        import_counts.sort(key=lambda x: x[1], reverse=True)
        return import_counts[:n]
    
    def _get_most_importing(self, n: int) -> List[Tuple[str, int]]:
        """Get files that import the most."""
        import_counts = [
            (path, len(score.imports))
            for path, score in self.file_scores.items()
        ]
        import_counts.sort(key=lambda x: x[1], reverse=True)
        return import_counts[:n]


# =============================================================================
# ADAPTIVE FILE LIMIT
# =============================================================================

def calculate_adaptive_limit(
    total_files: int,
    repo_size_mb: float = 0,
    time_budget_seconds: float = 300
) -> int:
    """
    Calculate an adaptive file limit based on repository characteristics.
    
    MEDIUM-PRIORITY: Automatically adjusts analysis scope based on repo size.
    """
    # Base limit
    base_limit = 30
    
    # Scale based on total files
    if total_files < 20:
        file_factor = 1.5  # Analyze more in small repos
    elif total_files < 100:
        file_factor = 1.0
    elif total_files < 500:
        file_factor = 0.7
    else:
        file_factor = 0.5
    
    # Scale based on time budget
    time_factor = max(0.5, min(2.0, time_budget_seconds / 300))
    
    # Calculate limit
    limit = int(base_limit * file_factor * time_factor)
    
    # Enforce bounds
    return max(10, min(100, limit))


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def prioritize_files(
    repo_path: str,
    max_files: int = 50,
    include_tests: bool = False,
    include_examples: bool = False
) -> List[str]:
    """
    Convenience function to prioritize files in a repository.
    
    Usage:
        files = prioritize_files("/path/to/repo", max_files=30)
        for file in files:
            analyze(file)
    """
    prioritizer = SmartFilePrioritizer(
        max_files=max_files,
        include_tests=include_tests,
        include_examples=include_examples
    )
    return prioritizer.prioritize(repo_path)


def get_core_files(repo_path: str, top_n: int = 10) -> List[str]:
    """
    Get the most important core files in a repository.
    
    Useful when you need to analyze just the essential files.
    """
    prioritizer = SmartFilePrioritizer(
        max_files=top_n,
        include_tests=False,
        include_examples=False
    )
    return prioritizer.prioritize(repo_path)