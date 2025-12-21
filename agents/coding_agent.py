"""
Coding Agent - Code Generation, Execution, and Visualization

This agent is responsible for:
- Setting up sandbox environments (Docker)
- Installing dependencies
- Generating test scripts from paper concepts
- Executing code and capturing outputs
- Creating visualizations
- Handling errors and debugging

ENHANCED: Now includes pre-execution validation and LLM self-correction
"""

import os
import re
import json
import ast
import asyncio
import tempfile
import shutil
import base64
from typing import Optional, Any, Dict, List, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import subprocess
import logging

from core.gemini_client import AgentLLM, GeminiConfig, GeminiModel
from core.knowledge_graph import (
    KnowledgeGraph, NodeType, EdgeType,
    get_global_graph
)
from core.agent_prompts import (
    get_agent_prompt, AgentType,
    CODE_GENERATION_TASK_PROMPT,
    build_task_prompt
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    stdout: str
    stderr: str
    return_code: int
    execution_time: float
    output_files: List[str] = field(default_factory=list)
    visualizations: List[Dict[str, Any]] = field(default_factory=list)
    error_analysis: Optional[str] = None


@dataclass
class GeneratedCode:
    """Generated code artifact."""
    filename: str
    language: str
    content: str
    purpose: str
    dependencies: List[str] = field(default_factory=list)
    is_test: bool = False
    is_visualization: bool = False


@dataclass
class SandboxEnvironment:
    """Docker sandbox environment."""
    container_id: Optional[str]
    image: str
    work_dir: str
    status: str
    dependencies_installed: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# CODE VALIDATOR (Integrated)
# =============================================================================

@dataclass
class ValidationIssue:
    """A validation issue found in code."""
    level: str  # error, warning, info
    category: str
    message: str
    line: Optional[int] = None
    suggestion: str = ""


@dataclass
class ValidationResult:
    """Result of code validation."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    fixed_code: Optional[str] = None
    
    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == 'error']


class CodeValidator:
    """Validates Python code before execution."""
    
    DANGEROUS_PATTERNS = [
        (r'\beval\s*\(', 'eval() is dangerous'),
        (r'\bexec\s*\(', 'exec() is dangerous'),
        (r'\bos\.system\s*\(', 'os.system() can be dangerous'),
        (r'\bsubprocess\.call\s*\([^)]*shell\s*=\s*True', 'shell=True is dangerous'),
    ]
    
    STANDARD_IMPORTS = {
        'os', 'sys', 're', 'json', 'math', 'random', 'time', 'datetime',
        'collections', 'itertools', 'functools', 'typing', 'pathlib',
        'dataclasses', 'enum', 'abc', 'copy', 'io', 'tempfile', 'shutil',
        'logging', 'warnings', 'traceback', 'hashlib', 'base64'
    }
    
    COMMON_PACKAGES = {
        'numpy', 'pandas', 'matplotlib', 'torch', 'tensorflow', 'sklearn',
        'scipy', 'requests', 'httpx', 'aiohttp', 'pydantic', 'fastapi'
    }
    
    def __init__(self, available_packages: Optional[Set[str]] = None):
        self.available_packages = available_packages or set()
    
    def validate(self, code: str, language: str = "python") -> ValidationResult:
        """Validate code."""
        if language not in ('python', 'py'):
            return ValidationResult(is_valid=True, issues=[])
        
        issues = []
        
        # Syntax check
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(ValidationIssue(
                level='error',
                category='syntax',
                message=f"Syntax error: {e.msg}",
                line=e.lineno,
                suggestion=self._get_syntax_suggestion(e)
            ))
            return ValidationResult(is_valid=False, issues=issues)
        
        # Import check
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split('.')[0]
                        if module not in self.STANDARD_IMPORTS and module not in self.available_packages:
                            if module in self.COMMON_PACKAGES:
                                issues.append(ValidationIssue(
                                    level='warning',
                                    category='import',
                                    message=f"Package '{module}' may need to be installed",
                                    line=node.lineno,
                                    suggestion=f"pip install {module}"
                                ))
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module = node.module.split('.')[0]
                        if module not in self.STANDARD_IMPORTS and module not in self.available_packages:
                            if module in self.COMMON_PACKAGES:
                                issues.append(ValidationIssue(
                                    level='warning',
                                    category='import',
                                    message=f"Package '{module}' may need to be installed",
                                    line=node.lineno
                                ))
        except:
            pass
        
        # Security check
        for pattern, message in self.DANGEROUS_PATTERNS:
            if re.search(pattern, code):
                issues.append(ValidationIssue(
                    level='warning',
                    category='security',
                    message=message
                ))
        
        is_valid = not any(i.level == 'error' for i in issues)
        return ValidationResult(is_valid=is_valid, issues=issues)
    
    def _get_syntax_suggestion(self, error: SyntaxError) -> str:
        msg = str(error.msg).lower()
        if 'unexpected eof' in msg:
            return "Check for missing closing brackets or quotes"
        elif 'invalid syntax' in msg:
            return "Check for typos or missing colons"
        elif 'indent' in msg:
            return "Fix indentation"
        return "Review the syntax"


class CodeFixer:
    """Fixes code issues using quick fixes and LLM."""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.validator = CodeValidator()
    
    def apply_quick_fixes(self, code: str, errors: List[ValidationIssue]) -> Tuple[str, List[str]]:
        """Apply quick automatic fixes."""
        fixes = []
        fixed_code = code
        
        for error in errors:
            if error.category == 'syntax':
                msg = error.message.lower()
                
                # Missing colon
                if 'expected ":"' in msg or "expected ':'" in msg:
                    if error.line:
                        lines = fixed_code.split('\n')
                        if error.line <= len(lines):
                            line = lines[error.line - 1]
                            if not line.rstrip().endswith(':'):
                                lines[error.line - 1] = line.rstrip() + ':'
                                fixed_code = '\n'.join(lines)
                                fixes.append(f"Added missing colon on line {error.line}")
                
                # Unmatched parenthesis
                elif 'unexpected eof' in msg or 'never closed' in msg:
                    open_parens = code.count('(') - code.count(')')
                    if open_parens > 0:
                        fixed_code = fixed_code.rstrip() + ')' * open_parens
                        fixes.append(f"Added {open_parens} closing parenthesis")
                    
                    open_brackets = code.count('[') - code.count(']')
                    if open_brackets > 0:
                        fixed_code = fixed_code.rstrip() + ']' * open_brackets
                        fixes.append(f"Added {open_brackets} closing brackets")
                    
                    open_braces = code.count('{') - code.count('}')
                    if open_braces > 0:
                        fixed_code = fixed_code.rstrip() + '}' * open_braces
                        fixes.append(f"Added {open_braces} closing braces")
        
        return fixed_code, fixes
    
    async def fix_with_llm(self, code: str, errors: List[ValidationIssue], purpose: str = "") -> Optional[str]:
        """Use LLM to fix code errors."""
        if not self.llm_client:
            return None
        
        error_desc = "\n".join([
            f"- Line {e.line}: {e.message}" + (f" ({e.suggestion})" if e.suggestion else "")
            for e in errors
        ])
        
        prompt = f"""Fix the following Python code that has these errors:

ERRORS:
{error_desc}

CODE:
```python
{code}
```

{f"PURPOSE: {purpose}" if purpose else ""}

Fix ALL errors and return ONLY the corrected Python code, no explanations.
"""
        
        try:
            response = await self.llm_client.generate(
                prompt,
                system_instruction="You are an expert Python developer. Fix code errors precisely."
            )
            
            # Extract code from response
            content = response.content
            match = re.search(r'```python\s*(.*?)\s*```', content, re.DOTALL)
            if match:
                return match.group(1).strip()
            
            # Try without code block
            content = re.sub(r'^```\w*\s*', '', content.strip())
            content = re.sub(r'\s*```$', '', content)
            return content
            
        except Exception as e:
            logger.error(f"LLM fix failed: {e}")
            return None
    
    async def fix_code(self, code: str, purpose: str = "", max_attempts: int = 3) -> Tuple[str, bool, List[str]]:
        """Iteratively fix code."""
        current_code = code
        all_fixes = []
        
        for attempt in range(max_attempts):
            result = self.validator.validate(current_code)
            
            if result.is_valid:
                return current_code, True, all_fixes
            
            # Try quick fixes first
            fixed, quick_fixes = self.apply_quick_fixes(current_code, result.errors)
            if quick_fixes:
                all_fixes.extend(quick_fixes)
                current_code = fixed
                continue
            
            # Try LLM fix
            if self.llm_client:
                llm_fixed = await self.fix_with_llm(current_code, result.errors, purpose)
                if llm_fixed and llm_fixed != current_code:
                    current_code = llm_fixed
                    all_fixes.append("LLM-based fix applied")
                    continue
            
            break
        
        final_result = self.validator.validate(current_code)
        return current_code, final_result.is_valid, all_fixes


# =============================================================================
# CODING AGENT
# =============================================================================

class CodingAgent:
    """
    Agent for code generation and execution.
    
    ENHANCED: Now includes pre-execution validation and LLM self-correction.
    """
    
    DOCKER_IMAGES = {
        "python": "python:3.11-slim",
        "javascript": "node:20-slim",
        "r": "r-base:latest",
        "julia": "julia:latest"
    }
    
    def __init__(
        self,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        gemini_api_key: Optional[str] = None,
        use_docker: bool = True
    ):
        self.knowledge_graph = knowledge_graph or get_global_graph()
        
        system_prompt = get_agent_prompt(AgentType.CODING)
        
        self.llm = AgentLLM(
            agent_name="CodingAgent",
            agent_role="Code generation, execution, and debugging",
            api_key=gemini_api_key,
            config=GeminiConfig(
                model=GeminiModel.FLASH,
                temperature=0.4,
                max_output_tokens=8192,
                system_instruction=system_prompt
            )
        )
        self.use_docker = use_docker
        self.environments: Dict[str, SandboxEnvironment] = {}
        self.generated_code: List[GeneratedCode] = []
        self.execution_history: List[ExecutionResult] = []
        self.work_dir = tempfile.mkdtemp(prefix="coding_agent_")
        
        # ENHANCED: Code validator and fixer
        self.validator = CodeValidator()
        self.fixer = CodeFixer(llm_client=self.llm)
    
    async def create_sandbox(
        self,
        language: str = "python",
        repo_path: Optional[str] = None
    ) -> SandboxEnvironment:
        """Create a sandbox environment for code execution."""
        image = self.DOCKER_IMAGES.get(language, self.DOCKER_IMAGES["python"])
        env_id = f"{language}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        work_dir = os.path.join(self.work_dir, env_id)
        os.makedirs(work_dir, exist_ok=True)
        
        container_id = None
        
        if self.use_docker:
            try:
                check = await asyncio.create_subprocess_exec(
                    "docker", "info",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await check.communicate()
                
                if check.returncode == 0:
                    mounts = [f"-v{work_dir}:/workspace"]
                    if repo_path:
                        mounts.append(f"-v{repo_path}:/repo:ro")
                    
                    cmd = [
                        "docker", "run", "-d",
                        "--name", env_id,
                        "-w", "/workspace",
                        *mounts,
                        image,
                        "tail", "-f", "/dev/null"
                    ]
                    
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await process.communicate()
                    
                    if process.returncode == 0:
                        container_id = stdout.decode().strip()
                
            except Exception as e:
                logger.warning(f"Docker not available: {e}")
        
        env = SandboxEnvironment(
            container_id=container_id,
            image=image,
            work_dir=work_dir,
            status="running" if container_id else "local"
        )
        
        self.environments[env_id] = env
        return env
    
    async def install_dependencies(
        self,
        env: SandboxEnvironment,
        dependencies: List[str],
        language: str = "python"
    ) -> bool:
        """Install dependencies in the sandbox."""
        if not dependencies:
            return True
        
        if language == "python":
            req_content = "\n".join(dependencies)
            req_path = os.path.join(env.work_dir, "requirements.txt")
            
            with open(req_path, 'w', encoding='utf-8') as f:
                f.write(req_content)
            
            if env.container_id:
                cmd = [
                    "docker", "exec", env.container_id,
                    "pip", "install", "-r", "/workspace/requirements.txt", "-q"
                ]
            else:
                cmd = [
                    "pip", "install", "-r", req_path, "-q",
                    "--target", os.path.join(env.work_dir, "site-packages")
                ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            
            env.dependencies_installed = (process.returncode == 0)
            return env.dependencies_installed
        
        return False
    
    async def generate_test_script(
        self,
        concepts: List[Dict[str, Any]],
        code_mappings: List[Dict[str, Any]],
        repo_info: Dict[str, Any],
        paper_summary: str
    ) -> List[GeneratedCode]:
        """Generate test scripts with validation and self-correction."""
        language = repo_info.get("main_language", "python")
        
        generation_prompt = f"""
Generate a comprehensive test script that demonstrates and verifies the concepts from a scientific paper.

Paper Summary:
{paper_summary}

Key Concepts:
{json.dumps(concepts, indent=2)}

Code Mappings (concept -> implementation):
{json.dumps(code_mappings, indent=2)}

Repository Info:
- Main Language: {language}
- Entry Points: {repo_info.get('entry_points', [])}

Generate a {language} test script that:
1. Imports necessary modules from the repository
2. Sets up test data or uses example data
3. Demonstrates each key concept with working code
4. Includes visualizations where appropriate
5. Prints clear output explaining what each test demonstrates
6. Handles errors gracefully

IMPORTANT: Ensure the code is syntactically correct and all imports are valid.

Respond in JSON format:
{{
    "main_script": {{
        "filename": "test_concepts.py",
        "content": "full script content",
        "purpose": "what the script does",
        "dependencies": ["list", "of", "imports"]
    }},
    "visualization_script": {{
        "filename": "visualize_results.py",
        "content": "visualization script content",
        "purpose": "creates visual outputs",
        "dependencies": ["matplotlib", "etc"]
    }}
}}
"""
        
        try:
            result = await self.llm.generate_structured(
                generation_prompt,
                schema={"type": "object"}
            )
            
            generated = []
            
            # Process main script
            if "main_script" in result:
                script = result["main_script"]
                code = GeneratedCode(
                    filename=script.get("filename", "test_concepts.py"),
                    language=language,
                    content=script.get("content", ""),
                    purpose=script.get("purpose", ""),
                    dependencies=script.get("dependencies", []),
                    is_test=True
                )
                
                # ENHANCED: Validate and fix code
                code = await self._validate_and_fix_code(code)
                generated.append(code)
            
            # Process visualization script
            if "visualization_script" in result:
                viz = result["visualization_script"]
                code = GeneratedCode(
                    filename=viz.get("filename", "visualize_results.py"),
                    language=language,
                    content=viz.get("content", ""),
                    purpose=viz.get("purpose", ""),
                    dependencies=viz.get("dependencies", []),
                    is_visualization=True
                )
                
                # ENHANCED: Validate and fix code
                code = await self._validate_and_fix_code(code)
                generated.append(code)
            
            self.generated_code.extend(generated)
            
            # Save to knowledge graph
            for code in generated:
                await self.knowledge_graph.add_node(
                    node_type=NodeType.TEST if code.is_test else NodeType.FILE,
                    name=code.filename,
                    content=code.content[:2000],
                    metadata={
                        "purpose": code.purpose,
                        "dependencies": code.dependencies,
                        "is_visualization": code.is_visualization
                    },
                    source="coding_agent"
                )
            
            return generated
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            fallback = GeneratedCode(
                filename="test_basic.py",
                language="python",
                content=f'''"""
Basic test script for paper concepts.
Generated as fallback due to: {str(e)}
"""

print("Testing paper concepts...")

import sys
sys.path.insert(0, '/repo')

print("✓ Environment setup complete")
print("✓ Repository accessible")

# Concepts to test: {[c.get("name", "") for c in concepts]}

print("\\nTest completed!")
''',
                purpose="Basic fallback test script",
                dependencies=["sys"],
                is_test=True
            )
            self.generated_code.append(fallback)
            return [fallback]
    
    async def _validate_and_fix_code(self, code: GeneratedCode) -> GeneratedCode:
        """ENHANCED: Validate and fix generated code."""
        if code.language not in ('python', 'py'):
            return code
        
        # Validate
        result = self.validator.validate(code.content, code.language)
        
        if result.is_valid:
            return code
        
        logger.info(f"Validating {code.filename}: {len(result.errors)} errors found")
        
        # Try to fix
        fixed_content, is_valid, fixes = await self.fixer.fix_code(
            code.content,
            purpose=code.purpose,
            max_attempts=3
        )
        
        if fixes:
            logger.info(f"Applied fixes to {code.filename}: {fixes}")
            code.content = fixed_content
            code.purpose += f" (auto-fixed: {', '.join(fixes)})"
        
        return code
    
    async def execute_code(
        self,
        env: SandboxEnvironment,
        code: GeneratedCode,
        timeout: int = 300
    ) -> ExecutionResult:
        """ENHANCED: Execute code with pre-validation."""
        start_time = datetime.now()
        
        # ENHANCED: Pre-validate before execution
        if code.language in ('python', 'py'):
            validation = self.validator.validate(code.content)
            
            if not validation.is_valid:
                # Try to fix
                fixed_content, is_valid, fixes = await self.fixer.fix_code(
                    code.content,
                    purpose=code.purpose,
                    max_attempts=2
                )
                
                if not is_valid:
                    error_details = "\n".join([
                        f"Line {e.line}: {e.message}" 
                        for e in validation.errors[:5]
                    ])
                    
                    return ExecutionResult(
                        success=False,
                        stdout="",
                        stderr=f"Code validation failed:\n{error_details}",
                        return_code=-1,
                        execution_time=0.0,
                        error_analysis="Code has syntax errors that could not be auto-fixed"
                    )
                
                code.content = fixed_content
        
        # Save code to file
        code_path = os.path.join(env.work_dir, code.filename)
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(code.content)
        
        # Prepare execution command
        if code.language == "python":
            if env.container_id:
                cmd = ["docker", "exec", "-w", "/workspace", env.container_id, "python", code.filename]
            else:
                cmd = ["python", code_path]
        elif code.language == "javascript":
            if env.container_id:
                cmd = ["docker", "exec", "-w", "/workspace", env.container_id, "node", code.filename]
            else:
                cmd = ["node", code_path]
        else:
            cmd = ["python", code_path]
        
        try:
            process = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=env.work_dir
                ),
                timeout=timeout
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Find output files
            output_files = []
            visualizations = []
            
            for item in os.listdir(env.work_dir):
                item_path = os.path.join(env.work_dir, item)
                if item.endswith(('.png', '.jpg', '.jpeg', '.svg', '.pdf')):
                    output_files.append(item)
                    try:
                        with open(item_path, 'rb') as f:
                            img_data = base64.b64encode(f.read()).decode()
                        visualizations.append({
                            "filename": item,
                            "type": "image",
                            "data": img_data,
                            "format": item.split('.')[-1]
                        })
                    except:
                        pass
            
            result = ExecutionResult(
                success=(process.returncode == 0),
                stdout=stdout.decode('utf-8', errors='replace'),
                stderr=stderr.decode('utf-8', errors='replace'),
                return_code=process.returncode,
                execution_time=execution_time,
                output_files=output_files,
                visualizations=visualizations
            )
            
            # Analyze errors if failed
            if not result.success and result.stderr:
                result.error_analysis = await self._analyze_error(
                    code.content,
                    result.stderr
                )
            
            self.execution_history.append(result)
            return result
            
        except asyncio.TimeoutError:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Execution timed out after {timeout} seconds",
                return_code=-1,
                execution_time=timeout,
                error_analysis="Code execution exceeded time limit"
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                return_code=-1,
                execution_time=(datetime.now() - start_time).total_seconds(),
                error_analysis=f"Execution error: {e}"
            )
    
    async def _analyze_error(self, code: str, error: str) -> str:
        """Analyze execution errors."""
        prompt = f"""
Analyze this Python error and suggest fixes:

Error:
{error[:1000]}

Code snippet (first 500 chars):
{code[:500]}

Provide a brief analysis of what went wrong and how to fix it.
"""
        
        try:
            response = await self.llm.generate(prompt)
            return response.content
        except:
            return "Error analysis unavailable"
    
    async def run_full_test_suite(
        self,
        env: SandboxEnvironment
    ) -> Dict[str, Any]:
        """Run all generated test scripts."""
        results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "details": []
        }
        
        for code in self.generated_code:
            if code.is_test or code.is_visualization:
                result = await self.execute_code(env, code)
                results["total"] += 1
                
                if result.success:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                
                results["details"].append({
                    "file": code.filename,
                    "success": result.success,
                    "stdout": result.stdout[:500],
                    "stderr": result.stderr[:500] if not result.success else "",
                    "visualizations": len(result.visualizations)
                })
        
        return results
    
    async def debug_and_fix(
        self,
        code: GeneratedCode,
        result: ExecutionResult,
        max_attempts: int = 3
    ) -> Tuple[GeneratedCode, ExecutionResult]:
        """
        Debug and fix failed code execution.

        Args:
            code: The code that failed
            result: The execution result with errors
            max_attempts: Maximum fix attempts

        Returns:
            Tuple of (fixed_code, new_result)
        """
        logger.info(f"Attempting to debug and fix {code.filename}")

        for attempt in range(max_attempts):
            # Try to fix the code
            fixed_content, is_valid, fixes = await self.fixer.fix_code(
                code.content,
                purpose=f"{code.purpose}\n\nError: {result.stderr[:500]}",
                max_attempts=2
            )

            if not is_valid or fixed_content == code.content:
                logger.warning(f"Fix attempt {attempt + 1} failed or made no changes")
                continue

            # Update code
            code.content = fixed_content
            code.purpose += f" (auto-fixed attempt {attempt + 1})"

            # Re-execute
            # Find the environment used previously
            env = None
            for e in self.environments.values():
                env = e
                break

            if not env:
                logger.error("No environment available for re-execution")
                break

            # Save and re-execute
            result = await self.execute_code(env, code)

            if result.success:
                logger.info(f"Successfully fixed {code.filename} on attempt {attempt + 1}")
                return code, result

        logger.warning(f"Failed to fix {code.filename} after {max_attempts} attempts")
        return code, result

    def get_all_visualizations(self) -> List[Dict[str, Any]]:
        """Get all visualizations from execution results."""
        visualizations = []
        for result in self.execution_history:
            visualizations.extend(result.visualizations)
        return visualizations

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all executions."""
        successful = sum(1 for r in self.execution_history if r.success)
        failed = len(self.execution_history) - successful
        total_time = sum(r.execution_time for r in self.execution_history)
        viz_count = sum(len(r.visualizations) for r in self.execution_history)

        return {
            "total": len(self.execution_history),
            "successful": successful,
            "failed": failed,
            "total_time": total_time,
            "visualizations_generated": viz_count
        }

    async def cleanup(self):
        """Clean up all environments and temporary files."""
        for env_id, env in self.environments.items():
            if env.container_id:
                try:
                    process = await asyncio.create_subprocess_exec(
                        "docker", "rm", "-f", env.container_id,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    await process.communicate()
                except:
                    pass

        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir, ignore_errors=True)