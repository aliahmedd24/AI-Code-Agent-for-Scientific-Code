"""
Gemini API Client - Core LLM Interface for the Agentic System

This module provides a unified interface to Google's Gemini API with:
- Automatic retry logic with exponential backoff
- Multi-key rotation for rate limit handling
- Structured output parsing
- Streaming support
- Context management for long conversations
- Function calling capabilities
"""

import os
import json
import asyncio
from typing import Optional, Any, Dict, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential,
    retry_if_exception_type
)
from rich.console import Console

console = Console()


class GeminiModel(Enum):
    """Available Gemini models with their capabilities."""
    FLASH = "gemini-2.0-flash-exp"
    PRO = "gemini-1.5-pro"
    FLASH_8B = "gemini-1.5-flash-8b"
    FLASH_LATEST = "gemini-1.5-flash-latest"  # Fallback option


@dataclass
class GeminiConfig:
    """Configuration for Gemini API calls."""
    model: GeminiModel = GeminiModel.FLASH
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    max_output_tokens: int = 8192
    system_instruction: Optional[str] = None
    safety_settings: Optional[Dict] = None
    
    def __post_init__(self):
        if self.safety_settings is None:
            # Relaxed safety settings for scientific content
            self.safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }


@dataclass
class Message:
    """Represents a message in a conversation."""
    role: str  # 'user' or 'model'
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Structured response from an agent."""
    content: str
    reasoning: Optional[str] = None
    actions: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    tokens_used: int = 0


class GeminiClient:
    """
    Advanced Gemini API client with agent-specific features.
    
    Features:
    - Multi-turn conversation management
    - Structured output extraction
    - Tool/function calling
    - Automatic context window management
    - Multi-key rotation for rate limit handling
    """
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[GeminiConfig] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.config = config or GeminiConfig()
        self._model = None
        self._chat = None
        self._conversation_history: List[Message] = []
        self._key_manager = None
        self._use_key_rotation = False
        
        # Try to use key manager if available
        try:
            from core.api_key_manager import get_key_manager
            self._key_manager = get_key_manager()
            if self.api_key:
                self._key_manager.add_key(self.api_key)
            self._use_key_rotation = self._key_manager.get_available_count() > 0
        except ImportError:
            pass
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
    
    def _configure_with_key(self, api_key: str):
        """Configure the API with a specific key."""
        genai.configure(api_key=api_key)
    
    def _get_model(self, system_instruction: Optional[str] = None) -> genai.GenerativeModel:
        """Get or create the generative model instance."""
        instruction = system_instruction or self.config.system_instruction
        
        generation_config = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "max_output_tokens": self.config.max_output_tokens,
        }
        
        return genai.GenerativeModel(
            model_name=self.config.model.value,
            generation_config=generation_config,
            safety_settings=self.config.safety_settings,
            system_instruction=instruction
        )
    
    async def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        output_format: Optional[str] = None,
        context: Optional[List[Message]] = None,
        max_retries: int = 5
    ) -> AgentResponse:
        """
        Generate a response from the model with automatic key rotation.
        
        Args:
            prompt: The user prompt
            system_instruction: Optional system instruction override
            output_format: Expected output format (json, markdown, code, etc.)
            context: Optional conversation context
            max_retries: Maximum retry attempts across all keys
        
        Returns:
            AgentResponse with the generated content
        """
        last_error = None
        
        for attempt in range(max_retries):
            # Get an API key (with rotation if available)
            current_key = self.api_key
            if self._use_key_rotation and self._key_manager:
                key = await self._key_manager.get_key(wait=True, timeout=30.0)
                if key:
                    current_key = key
                    self._configure_with_key(key)
            
            if not current_key:
                raise ValueError("No API key available. Please set GEMINI_API_KEY.")
            
            try:
                model = self._get_model(system_instruction)
                
                # Build the full prompt with format instructions
                full_prompt = prompt
                if output_format:
                    full_prompt = f"{prompt}\n\n[OUTPUT FORMAT: {output_format}]"
                
                # Add context if provided
                if context:
                    history = []
                    for msg in context:
                        history.append({"role": msg.role, "parts": [msg.content]})
                    chat = model.start_chat(history=history)
                    response = await asyncio.to_thread(chat.send_message, full_prompt)
                else:
                    response = await asyncio.to_thread(model.generate_content, full_prompt)
                
                # Mark success if using key rotation
                if self._use_key_rotation and self._key_manager:
                    self._key_manager.mark_success(current_key)
                
                # Parse and structure the response
                content = response.text if response.text else ""
                
                # Try to extract structured data if JSON format was requested
                reasoning = None
                actions = []
                artifacts = {}
                
                if output_format == "json":
                    try:
                        json_str = self._extract_json(content)
                        if json_str:
                            parsed = json.loads(json_str)
                            if isinstance(parsed, dict):
                                reasoning = parsed.get("reasoning")
                                actions = parsed.get("actions", [])
                                artifacts = parsed.get("artifacts", {})
                    except json.JSONDecodeError:
                        pass
                
                return AgentResponse(
                    content=content,
                    reasoning=reasoning,
                    actions=actions,
                    artifacts=artifacts,
                    tokens_used=response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') and response.usage_metadata else 0
                )
                
            except Exception as e:
                error_str = str(e)
                last_error = e
                
                # Mark failure if using key rotation
                if self._use_key_rotation and self._key_manager:
                    self._key_manager.mark_failure(current_key, error_str)
                
                # Check if it's a rate limit error
                is_rate_limit = any(x in error_str for x in [
                    "ResourceExhausted", "429", "rate", "quota", "limit"
                ])
                
                if is_rate_limit:
                    console.print(f"[yellow]⚠️ Rate limit hit (attempt {attempt + 1}/{max_retries}), rotating key...[/yellow]")
                    # Wait before retry with exponential backoff
                    wait_time = min(2 ** attempt, 30)  # Max 30 seconds
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # For other errors, raise immediately
                    console.print(f"[red]✗ API error: {error_str[:100]}[/red]")
                    raise
        
        # All retries exhausted
        raise last_error or Exception("All API retry attempts exhausted")
    
    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON from text, handling code blocks."""
        # Try to find JSON in code blocks
        import re
        
        # Check for ```json blocks
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
        if json_match:
            return json_match.group(1)
        
        # Check for ``` blocks
        code_match = re.search(r'```\s*([\s\S]*?)\s*```', text)
        if code_match:
            return code_match.group(1)
        
        # Try the entire text
        try:
            # Find the first { and last }
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                return text[start:end+1]
        except:
            pass
        
        return None
    
    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a response that conforms to a specific JSON schema.
        
        Args:
            prompt: The user prompt
            schema: JSON schema the output should conform to
            system_instruction: Optional system instruction
        
        Returns:
            Parsed JSON response conforming to the schema
        """
        schema_prompt = f"""
{prompt}

You MUST respond with valid JSON that conforms to this schema:
```json
{json.dumps(schema, indent=2)}
```

Respond ONLY with the JSON, no additional text or explanation.
"""
        
        response = await self.generate(
            schema_prompt,
            system_instruction=system_instruction,
            output_format="json"
        )
        
        json_str = self._extract_json(response.content)
        if json_str:
            return json.loads(json_str)
        
        # Fallback: try parsing the entire content
        return json.loads(response.content)
    
    async def analyze_code(
        self,
        code: str,
        language: str = "python",
        analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Analyze code for structure, dependencies, and quality.
        
        Args:
            code: The source code to analyze
            language: Programming language
            analysis_type: Type of analysis (comprehensive, security, performance)
        
        Returns:
            Analysis results including structure, dependencies, issues
        """
        prompt = f"""
Analyze the following {language} code and provide a comprehensive analysis.

```{language}
{code}
```

Provide your analysis in the following JSON format:
{{
    "summary": "Brief description of what this code does",
    "main_components": [
        {{"name": "component_name", "type": "class/function/module", "purpose": "description"}}
    ],
    "dependencies": [
        {{"name": "package_name", "version": "if specified", "usage": "how it's used"}}
    ],
    "entry_points": ["list of main entry points"],
    "complexity": {{"score": 1-10, "reasoning": "explanation"}},
    "issues": [
        {{"severity": "high/medium/low", "description": "issue description", "location": "where in code"}}
    ],
    "suggestions": ["improvement suggestions"],
    "compute_requirements": {{
        "estimated_memory": "RAM estimate",
        "gpu_required": true/false,
        "estimated_runtime": "time estimate"
    }}
}}
"""
        
        return await self.generate_structured(
            prompt,
            schema={"type": "object"},
            system_instruction="You are an expert code analyst. Provide detailed, accurate analysis."
        )
    
    async def generate_code(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        language: str = "python",
        style: str = "clean"
    ) -> str:
        """
        Generate code based on a task description.
        
        Args:
            task: Description of what the code should do
            context: Additional context (existing code, dependencies, etc.)
            language: Target programming language
            style: Code style (clean, verbose, minimal)
        
        Returns:
            Generated code
        """
        context_str = ""
        if context:
            context_str = f"\nContext:\n{json.dumps(context, indent=2)}\n"
        
        prompt = f"""
Generate {language} code to accomplish the following task:

{task}
{context_str}

Requirements:
- Write clean, well-documented code
- Include error handling
- Add docstrings and comments
- Make it production-ready
- Style: {style}

Provide ONLY the code, wrapped in ```{language} code blocks.
"""
        
        response = await self.generate(
            prompt,
            system_instruction="You are an expert programmer. Generate clean, efficient, well-documented code."
        )
        
        # Extract code from response
        import re
        code_match = re.search(rf'```{language}\s*([\s\S]*?)\s*```', response.content)
        if code_match:
            return code_match.group(1)
        
        # Try generic code blocks
        code_match = re.search(r'```\s*([\s\S]*?)\s*```', response.content)
        if code_match:
            return code_match.group(1)
        
        return response.content
    
    def start_chat(self, system_instruction: Optional[str] = None):
        """Start a new chat session."""
        model = self._get_model(system_instruction)
        self._chat = model.start_chat(history=[])
        self._conversation_history = []
        return self
    
    async def chat(self, message: str) -> str:
        """Send a message in the current chat session."""
        if not self._chat:
            self.start_chat()
        
        self._conversation_history.append(Message(role="user", content=message))
        response = await asyncio.to_thread(self._chat.send_message, message)
        
        content = response.text if response.text else ""
        self._conversation_history.append(Message(role="model", content=content))
        
        return content
    
    def get_history(self) -> List[Message]:
        """Get the conversation history."""
        return self._conversation_history.copy()
    
    def clear_history(self):
        """Clear the conversation history."""
        self._conversation_history = []
        self._chat = None


class AgentLLM(GeminiClient):
    """
    Specialized LLM client for agent-specific operations.
    
    Extends GeminiClient with agent-focused features:
    - Role-based system prompts
    - Action parsing
    - Tool invocation handling
    """
    
    def __init__(
        self,
        agent_name: str,
        agent_role: str,
        api_key: Optional[str] = None,
        config: Optional[GeminiConfig] = None
    ):
        super().__init__(api_key, config)
        self.agent_name = agent_name
        self.agent_role = agent_role
        self._tools: Dict[str, Callable] = {}
    
    def register_tool(self, name: str, func: Callable, description: str):
        """Register a tool that the agent can use."""
        self._tools[name] = {
            "function": func,
            "description": description
        }
    
    async def think_and_act(
        self,
        task: str,
        available_actions: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Have the agent think through a task and decide on actions.
        
        Args:
            task: The task to accomplish
            available_actions: List of actions the agent can take
            context: Additional context for decision making
        
        Returns:
            AgentResponse with reasoning and planned actions
        """
        system_prompt = f"""
You are {self.agent_name}, an AI agent with the following role: {self.agent_role}

You must think step-by-step about the given task and decide what actions to take.

Available actions: {', '.join(available_actions)}

Respond in JSON format:
{{
    "reasoning": "Your step-by-step reasoning",
    "actions": [
        {{"action": "action_name", "parameters": {{}}, "expected_outcome": "what you expect"}}
    ],
    "confidence": 0.0-1.0
}}
"""
        
        context_str = ""
        if context:
            context_str = f"\n\nContext:\n{json.dumps(context, indent=2)}"
        
        prompt = f"Task: {task}{context_str}"
        
        response = await self.generate(
            prompt,
            system_instruction=system_prompt,
            output_format="json"
        )
        
        # Parse the response
        try:
            json_str = self._extract_json(response.content)
            if json_str:
                parsed = json.loads(json_str)
                return AgentResponse(
                    content=response.content,
                    reasoning=parsed.get("reasoning"),
                    actions=parsed.get("actions", []),
                    confidence=parsed.get("confidence", 1.0)
                )
        except:
            pass
        
        return response


# Convenience function to create a configured client
def create_gemini_client(
    api_key: Optional[str] = None,
    model: GeminiModel = GeminiModel.FLASH,
    temperature: float = 0.7
) -> GeminiClient:
    """Create a configured Gemini client."""
    config = GeminiConfig(model=model, temperature=temperature)
    return GeminiClient(api_key=api_key, config=config)
