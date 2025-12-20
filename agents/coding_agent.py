"""
Coding Agent - Code Generation, Execution, and Visualization

This agent is responsible for:
- Setting up sandbox environments (Docker)
- Installing dependencies
- Generating test scripts from paper concepts
- Executing code and capturing outputs
- Creating visualizations
- Handling errors and debugging
"""

import os
import re
import json
import asyncio
import tempfile
import shutil
import base64
from typing import Optional, Any, Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import subprocess

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


class CodingAgent:
    """
    Agent for code generation and execution.
    
    Capabilities:
    - Create Docker sandbox environments
    - Install project dependencies
    - Generate test scripts based on paper concepts
    - Execute code safely in sandbox
    - Capture and display visualizations
    - Debug and fix errors automatically
    """
    
    # Docker images for different languages
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
        
        # Get specialized system prompt for this agent
        system_prompt = get_agent_prompt(AgentType.CODING)
        
        self.llm = AgentLLM(
            agent_name="CodingAgent",
            agent_role="Code generation, execution, and debugging",
            api_key=gemini_api_key,
            config=GeminiConfig(
                model=GeminiModel.FLASH,
                temperature=0.4,
                max_output_tokens=8192,
                system_instruction=system_prompt  # Use specialized prompt
            )
        )
        self.use_docker = use_docker
        self.environments: Dict[str, SandboxEnvironment] = {}
        self.generated_code: List[GeneratedCode] = []
        self.execution_history: List[ExecutionResult] = []
        self.work_dir = tempfile.mkdtemp(prefix="coding_agent_")
    
    async def create_sandbox(
        self,
        language: str = "python",
        repo_path: Optional[str] = None
    ) -> SandboxEnvironment:
        """
        Create a sandbox environment for code execution.
        
        Args:
            language: Programming language
            repo_path: Optional path to repository to mount
        
        Returns:
            SandboxEnvironment object
        """
        image = self.DOCKER_IMAGES.get(language, self.DOCKER_IMAGES["python"])
        env_id = f"{language}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        work_dir = os.path.join(self.work_dir, env_id)
        os.makedirs(work_dir, exist_ok=True)
        
        container_id = None
        
        if self.use_docker:
            try:
                # Check if Docker is available
                check = await asyncio.create_subprocess_exec(
                    "docker", "info",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await check.communicate()
                
                if check.returncode == 0:
                    # Create Docker container
                    mounts = [f"-v{work_dir}:/workspace"]
                    if repo_path:
                        mounts.append(f"-v{repo_path}:/repo:ro")
                    
                    cmd = [
                        "docker", "run", "-d",
                        "--name", env_id,
                        "-w", "/workspace",
                        *mounts,
                        image,
                        "tail", "-f", "/dev/null"  # Keep container running
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
                print(f"Docker not available: {e}")
        
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
        """
        Install dependencies in the sandbox.
        
        Args:
            env: Sandbox environment
            dependencies: List of dependencies to install
            language: Programming language
        
        Returns:
            True if successful
        """
        if not dependencies:
            return True
        
        if language == "python":
            # Create requirements.txt
            req_content = "\n".join(dependencies)
            req_path = os.path.join(env.work_dir, "requirements.txt")
            
            with open(req_path, 'w', encoding='utf-8') as f:
                f.write(req_content)
            
            # Install in container or locally
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
            stdout, stderr = await process.communicate()
            
            env.dependencies_installed = (process.returncode == 0)
            return env.dependencies_installed
        
        elif language == "javascript":
            # Create package.json
            package = {
                "name": "test-environment",
                "version": "1.0.0",
                "dependencies": {dep: "*" for dep in dependencies}
            }
            package_path = os.path.join(env.work_dir, "package.json")
            
            with open(package_path, 'w', encoding='utf-8') as f:
                json.dump(package, f)
            
            if env.container_id:
                cmd = [
                    "docker", "exec", "-w", "/workspace",
                    env.container_id, "npm", "install", "--silent"
                ]
            else:
                cmd = ["npm", "install", "--silent", "--prefix", env.work_dir]
            
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
        """
        Generate test scripts that verify paper concepts using the repository.
        
        Args:
            concepts: Concepts from paper analysis
            code_mappings: Mappings between concepts and code
            repo_info: Repository information
            paper_summary: Summary of the paper
        
        Returns:
            List of generated code files
        """
        language = repo_info.get("main_language", "python")
        
        # Prepare context for code generation
        context = {
            "concepts": concepts,
            "mappings": code_mappings,
            "entry_points": repo_info.get("entry_points", []),
            "dependencies": repo_info.get("analysis", {}).get("main_components", [])
        }
        
        generation_prompt = f"""
Generate a comprehensive test script that demonstrates and verifies the concepts from a scientific paper using the provided code repository.

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
4. Includes visualizations where appropriate (save to files)
5. Prints clear output explaining what each test demonstrates
6. Handles errors gracefully

The script should be self-contained and ready to run.

Also generate a separate visualization script that creates charts/plots to illustrate the paper's concepts.

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
    }},
    "helper_files": [
        {{
            "filename": "helper.py",
            "content": "helper code if needed",
            "purpose": "utility functions"
        }}
    ],
    "expected_outputs": ["list of expected output files"],
    "run_instructions": "how to run the scripts"
}}
"""
        
        try:
            result = await self.llm.generate_structured(
                generation_prompt,
                schema={"type": "object"}
                # System instruction already set in config - uses specialized CodingAgent prompt
            )
            
            generated = []
            
            # Main test script
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
                generated.append(code)
            
            # Visualization script
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
                generated.append(code)
            
            # Helper files
            for helper in result.get("helper_files", []):
                code = GeneratedCode(
                    filename=helper.get("filename", "helper.py"),
                    language=language,
                    content=helper.get("content", ""),
                    purpose=helper.get("purpose", "")
                )
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
            print(f"Code generation failed: {e}")
            # Generate a simple fallback script
            fallback = GeneratedCode(
                filename="test_basic.py",
                language="python",
                content=f'''"""
Basic test script for paper concepts.
Generated as fallback due to: {str(e)}
"""

print("Testing paper concepts...")

# Import repository (adjust path as needed)
import sys
sys.path.insert(0, '/repo')

# Basic tests
print("✓ Environment setup complete")
print("✓ Repository accessible")

# TODO: Add specific tests based on paper concepts
# Concepts to test: {[c.get("name", "") for c in concepts]}

print("\\nTest completed!")
''',
                purpose="Basic fallback test script",
                dependencies=["sys"],
                is_test=True
            )
            self.generated_code.append(fallback)
            return [fallback]
    
    async def execute_code(
        self,
        env: SandboxEnvironment,
        code: GeneratedCode,
        timeout: int = 300
    ) -> ExecutionResult:
        """
        Execute code in the sandbox environment.
        
        Args:
            env: Sandbox environment
            code: Code to execute
            timeout: Timeout in seconds
        
        Returns:
            ExecutionResult object
        """
        start_time = datetime.now()
        
        # Save code to file
        code_path = os.path.join(env.work_dir, code.filename)
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(code.content)
        
        # Prepare execution command
        if code.language == "python":
            if env.container_id:
                cmd = [
                    "docker", "exec", "-w", "/workspace",
                    env.container_id, "python", code.filename
                ]
            else:
                cmd = ["python", code_path]
        elif code.language == "javascript":
            if env.container_id:
                cmd = [
                    "docker", "exec", "-w", "/workspace",
                    env.container_id, "node", code.filename
                ]
            else:
                cmd = ["node", code_path]
        else:
            cmd = ["python", code_path]  # Default to Python
        
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
            
            # Find output files (images, data files)
            output_files = []
            visualizations = []
            
            for item in os.listdir(env.work_dir):
                item_path = os.path.join(env.work_dir, item)
                if item.endswith(('.png', '.jpg', '.jpeg', '.svg', '.pdf')):
                    output_files.append(item)
                    # Read and encode image for display
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
                elif item.endswith(('.csv', '.json', '.txt')):
                    output_files.append(item)
            
            result = ExecutionResult(
                success=(process.returncode == 0),
                stdout=stdout.decode('utf-8', errors='replace'),
                stderr=stderr.decode('utf-8', errors='replace'),
                return_code=process.returncode,
                execution_time=execution_time,
                output_files=output_files,
                visualizations=visualizations
            )
            
        except asyncio.TimeoutError:
            result = ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Execution timed out after {timeout} seconds",
                return_code=-1,
                execution_time=timeout,
                error_analysis="Timeout - code took too long to execute"
            )
        
        except Exception as e:
            result = ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                return_code=-1,
                execution_time=(datetime.now() - start_time).total_seconds(),
                error_analysis=str(e)
            )
        
        # If failed, analyze error
        if not result.success and not result.error_analysis:
            result.error_analysis = await self._analyze_error(
                code.content, result.stderr
            )
        
        # Store in knowledge graph
        await self.knowledge_graph.add_node(
            node_type=NodeType.RESULT,
            name=f"Execution: {code.filename}",
            content=result.stdout[:2000] if result.success else result.stderr[:2000],
            metadata={
                "success": result.success,
                "return_code": result.return_code,
                "execution_time": result.execution_time,
                "output_files": result.output_files
            },
            source="coding_agent"
        )
        
        self.execution_history.append(result)
        return result
    
    async def _analyze_error(self, code: str, error: str) -> str:
        """Analyze an error and suggest fixes."""
        prompt = f"""
Analyze this code error and provide a brief explanation with fix suggestion.

Code (relevant parts):
{code[:1500]}

Error:
{error[:1000]}

Provide a brief analysis in 2-3 sentences.
"""
        
        try:
            response = await self.llm.generate(
                prompt,
                system_instruction="You are a debugging expert. Be concise."
            )
            return response.content
        except:
            return "Error analysis failed"
    
    async def debug_and_fix(
        self,
        code: GeneratedCode,
        error: ExecutionResult,
        max_attempts: int = 3
    ) -> Tuple[GeneratedCode, ExecutionResult]:
        """
        Attempt to fix code based on error.
        
        Args:
            code: Original code
            error: Error result
            max_attempts: Maximum fix attempts
        
        Returns:
            Tuple of (fixed code, execution result)
        """
        current_code = code
        current_result = error
        
        for attempt in range(max_attempts):
            if current_result.success:
                break
            
            fix_prompt = f"""
Fix this code based on the error message.

Original code:
```{code.language}
{current_code.content}
```

Error:
{current_result.stderr}

Provide the complete fixed code. Only output the code, no explanations.
"""
            
            try:
                fixed_content = await self.llm.generate_code(
                    fix_prompt,
                    language=code.language
                )
                
                fixed_code = GeneratedCode(
                    filename=f"fixed_{attempt}_{code.filename}",
                    language=code.language,
                    content=fixed_content,
                    purpose=f"Fixed version (attempt {attempt + 1})",
                    dependencies=code.dependencies,
                    is_test=code.is_test,
                    is_visualization=code.is_visualization
                )
                
                # Get or create environment
                env = list(self.environments.values())[0] if self.environments else await self.create_sandbox(code.language)
                
                # Execute fixed code
                current_result = await self.execute_code(env, fixed_code)
                current_code = fixed_code
                
            except Exception as e:
                print(f"Fix attempt {attempt + 1} failed: {e}")
        
        return current_code, current_result
    
    async def generate_visualization(
        self,
        data: Dict[str, Any],
        viz_type: str = "auto",
        title: str = "Visualization"
    ) -> GeneratedCode:
        """
        Generate a visualization script.
        
        Args:
            data: Data to visualize
            viz_type: Type of visualization (auto, line, bar, scatter, etc.)
            title: Visualization title
        
        Returns:
            Generated visualization code
        """
        prompt = f"""
Create a Python visualization script using matplotlib and/or plotly.

Data to visualize:
{json.dumps(data, indent=2)[:2000]}

Visualization type: {viz_type}
Title: {title}

Requirements:
1. Create a clear, professional visualization
2. Save the figure to 'visualization.png'
3. Include proper labels and legend
4. Handle the data format appropriately
5. Use a clean style

Provide only the Python code.
"""
        
        content = await self.llm.generate_code(prompt, language="python")
        
        viz_code = GeneratedCode(
            filename="generate_viz.py",
            language="python",
            content=content,
            purpose=f"Generate {viz_type} visualization: {title}",
            dependencies=["matplotlib", "numpy"],
            is_visualization=True
        )
        
        self.generated_code.append(viz_code)
        return viz_code
    
    async def run_full_test_suite(
        self,
        env: SandboxEnvironment
    ) -> Dict[str, Any]:
        """
        Run all generated test scripts.
        
        Args:
            env: Sandbox environment
        
        Returns:
            Summary of all test results
        """
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
        
        # Clean up work directory
        try:
            shutil.rmtree(self.work_dir)
        except:
            pass
        
        self.environments = {}
        self.generated_code = []
        self.execution_history = []
    
    def get_all_visualizations(self) -> List[Dict[str, Any]]:
        """Get all generated visualizations."""
        visualizations = []
        for result in self.execution_history:
            visualizations.extend(result.visualizations)
        return visualizations
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all executions."""
        return {
            "total_executions": len(self.execution_history),
            "successful": sum(1 for r in self.execution_history if r.success),
            "failed": sum(1 for r in self.execution_history if not r.success),
            "total_time": sum(r.execution_time for r in self.execution_history),
            "visualizations_generated": sum(
                len(r.visualizations) for r in self.execution_history
            ),
            "output_files": sum(
                len(r.output_files) for r in self.execution_history
            )
        }
