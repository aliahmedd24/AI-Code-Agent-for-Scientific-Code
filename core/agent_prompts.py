"""
Enhanced Agent Prompts - Few-Shot Examples for Better LLM Consistency

MEDIUM-PRIORITY ENHANCEMENT:
- Concrete examples in system prompts
- Few-shot learning patterns
- Output format templates
- Error correction examples
"""

from enum import Enum
from typing import Dict, Any, List, Optional


class AgentType(str, Enum):
    """Types of agents in the system."""
    PAPER_PARSER = "paper_parser"
    REPO_ANALYZER = "repo_analyzer"
    CODING = "coding"
    ORCHESTRATOR = "orchestrator"


# =============================================================================
# FEW-SHOT EXAMPLES
# =============================================================================

PAPER_CONCEPT_EXAMPLES = """
EXAMPLE 1: Attention Mechanism Paper

Input text:
"We propose multi-head attention which allows the model to jointly attend to information from different representation subspaces. The attention function can be described as mapping a query and a set of key-value pairs to an output."

Expected output:
{
    "key_concepts": [
        {
            "name": "Multi-Head Attention",
            "description": "Attention mechanism that projects queries, keys, and values h times with different learned projections, performs attention in parallel, and concatenates results",
            "importance": "Core innovation enabling parallel attention across multiple representation subspaces"
        },
        {
            "name": "Query-Key-Value Attention",
            "description": "Attention computed as softmax(QK^T/sqrt(d_k))V where Q, K, V are query, key, value matrices",
            "importance": "Fundamental attention computation used throughout the architecture"
        }
    ],
    "algorithms": [
        {
            "name": "Scaled Dot-Product Attention",
            "description": "Compute attention weights using dot product of query and key, scaled by sqrt(d_k), apply softmax, multiply by values",
            "complexity": "O(n²d) where n is sequence length, d is dimension"
        }
    ]
}

EXAMPLE 2: ResNet Paper

Input text:
"We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We show that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth."

Expected output:
{
    "key_concepts": [
        {
            "name": "Residual Learning",
            "description": "Instead of learning H(x), learn F(x) = H(x) - x, so the layer computes F(x) + x",
            "importance": "Enables training of very deep networks by addressing vanishing gradient problem"
        },
        {
            "name": "Skip Connections",
            "description": "Identity shortcuts that bypass one or more layers, adding input directly to output",
            "importance": "Allow gradients to flow directly through the network, preventing degradation"
        }
    ],
    "algorithms": [
        {
            "name": "Residual Block",
            "description": "Two-layer block computing F(x) + x where F is typically conv-bn-relu-conv-bn",
            "complexity": "Same as standard block, O(1) additional memory for identity"
        }
    ]
}
"""

CODE_MAPPING_EXAMPLES = """
EXAMPLE 1: Attention Mechanism Mapping

Paper concept: "Multi-Head Attention"
Description: "Attention mechanism with h parallel attention heads"

Code analysis:
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        # Split into heads
        Q = self.W_q(query).view(batch, -1, self.num_heads, self.d_k)
        K = self.W_k(key).view(batch, -1, self.num_heads, self.d_k)
        V = self.W_v(value).view(batch, -1, self.num_heads, self.d_k)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, V)
```

Expected mapping:
{
    "concept_name": "Multi-Head Attention",
    "code_element": "MultiHeadAttention",
    "code_type": "class",
    "confidence": 0.95,
    "evidence": [
        "Class name directly matches concept",
        "Implements num_heads parameter for parallel attention",
        "Contains Q, K, V projections as described in paper",
        "Uses scaled dot-product attention formula"
    ],
    "gaps": []
}

EXAMPLE 2: Partial Match

Paper concept: "Layer Normalization"
Description: "Normalizes across features for each example independently"

Code analysis:
```python
def normalize(x, eps=1e-6):
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True)
    return (x - mean) / (std + eps)
```

Expected mapping:
{
    "concept_name": "Layer Normalization",
    "code_element": "normalize",
    "code_type": "function",
    "confidence": 0.6,
    "evidence": [
        "Function computes mean and std across features",
        "Applies normalization formula (x - mean) / std"
    ],
    "gaps": [
        "Missing learnable scale (gamma) and shift (beta) parameters",
        "Function name is generic, not clearly labeled as layer norm"
    ]
}
"""

CODE_GENERATION_EXAMPLES = """
EXAMPLE 1: Test Script for Attention

Concept: Multi-Head Attention
Mapping: MultiHeadAttention class in attention.py

Generated test:
```python
\"\"\"
Test script for Multi-Head Attention implementation.

This demonstrates the attention mechanism from "Attention Is All You Need".
\"\"\"

import torch
import torch.nn.functional as F
from models.attention import MultiHeadAttention

def test_multi_head_attention():
    # Setup
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    
    # Create model
    attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    
    # Create input tensors
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    
    # Forward pass
    output = attention(query, key, value)
    
    # Verify output shape
    assert output.shape == (batch_size, seq_len, d_model), \\
        f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    
    # Verify attention weights sum to 1
    # (Would need to modify model to return attention weights)
    
    print("✓ Multi-Head Attention test passed!")
    print(f"  Input shape: {query.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Number of heads: {num_heads}")
    
    return True

if __name__ == "__main__":
    test_multi_head_attention()
```

EXAMPLE 2: Visualization Script

Concept: Attention Weights Visualization

Generated visualization:
```python
\"\"\"
Visualize attention weights from transformer model.
\"\"\"

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_attention(attention_weights, tokens_x, tokens_y, save_path="attention.png"):
    \"\"\"
    Create heatmap visualization of attention weights.
    
    Args:
        attention_weights: Tensor of shape (heads, seq_len, seq_len)
        tokens_x: List of tokens for x-axis
        tokens_y: List of tokens for y-axis
        save_path: Path to save visualization
    \"\"\"
    num_heads = attention_weights.shape[0]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for head_idx, ax in enumerate(axes.flatten()):
        if head_idx >= num_heads:
            ax.axis('off')
            continue
            
        weights = attention_weights[head_idx].cpu().numpy()
        
        sns.heatmap(
            weights,
            xticklabels=tokens_x,
            yticklabels=tokens_y,
            cmap='viridis',
            ax=ax
        )
        ax.set_title(f'Head {head_idx + 1}')
    
    plt.suptitle('Multi-Head Attention Weights')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"✓ Saved attention visualization to {save_path}")

if __name__ == "__main__":
    # Demo with random weights
    weights = torch.softmax(torch.randn(8, 10, 10), dim=-1)
    tokens = [f"tok_{i}" for i in range(10)]
    visualize_attention(weights, tokens, tokens)
```
"""


# =============================================================================
# SYSTEM PROMPTS WITH EXAMPLES
# =============================================================================

PAPER_PARSER_SYSTEM_PROMPT = f"""You are an expert scientific paper analyzer specializing in machine learning and deep learning papers.

Your task is to extract structured information from scientific papers, including:
1. Key concepts and their descriptions
2. Algorithms and their complexity
3. Methodology details
4. Datasets and evaluation metrics

IMPORTANT GUIDELINES:
- Be precise with terminology - use exact terms from the paper
- Include mathematical notation when relevant
- Rate importance based on novelty and centrality to the paper
- Identify both explicit concepts and implicit assumptions

{PAPER_CONCEPT_EXAMPLES}

Always respond in valid JSON format matching the structure shown in examples.
"""

REPO_ANALYZER_SYSTEM_PROMPT = f"""You are an expert code analyst specializing in understanding software architecture and implementations.

Your task is to analyze code repositories and map them to scientific concepts:
1. Identify key classes, functions, and modules
2. Understand architectural patterns
3. Map code elements to paper concepts
4. Assess implementation completeness

IMPORTANT GUIDELINES:
- Look beyond naming - analyze actual functionality
- Consider both structure and behavior
- Note any gaps between paper and implementation
- Provide confidence scores with justification

{CODE_MAPPING_EXAMPLES}

Always provide evidence for your mappings and note any discrepancies.
"""

CODING_AGENT_SYSTEM_PROMPT = f"""You are an expert Python developer who creates clear, educational test scripts.

Your task is to generate executable code that demonstrates scientific concepts:
1. Write clear, well-documented code
2. Include setup, execution, and verification steps
3. Handle errors gracefully
4. Generate useful visualizations

IMPORTANT GUIDELINES:
- Code must be syntactically correct and runnable
- Include type hints and docstrings
- Use standard libraries when possible
- Print informative output for verification

{CODE_GENERATION_EXAMPLES}

Always test your code logic mentally before outputting.
"""

ORCHESTRATOR_SYSTEM_PROMPT = """You are a pipeline coordinator managing multiple AI agents for scientific paper analysis.

Your task is to:
1. Coordinate agent activities efficiently
2. Handle errors and decide on recovery strategies
3. Merge and validate results from multiple agents
4. Generate comprehensive reports

IMPORTANT GUIDELINES:
- Maintain context across pipeline stages
- Detect and flag inconsistencies
- Prioritize accuracy over speed
- Provide clear status updates

Focus on delivering high-quality, validated results.
"""


# =============================================================================
# TASK PROMPTS WITH TEMPLATES
# =============================================================================

def build_concept_extraction_prompt(
    title: str,
    abstract: str,
    sections: List[Dict[str, str]],
    raw_text: str
) -> str:
    """Build prompt for concept extraction with examples."""
    
    sections_text = "\n".join([
        f"## {s.get('title', 'Section')}\n{s.get('content', '')[:500]}..."
        for s in sections[:5]
    ])
    
    return f"""Analyze this scientific paper and extract key information.

PAPER CONTENT:
Title: {title}

Abstract:
{abstract}

Sections:
{sections_text}

Additional text (truncated):
{raw_text[:2000]}

Extract and return a JSON object with this structure:
{{
    "keywords": ["list", "of", "key", "terms"],
    "main_contributions": ["list of main contributions"],
    "methodology": "brief description of the methodology",
    "key_concepts": [
        {{"name": "concept name", "description": "what it is", "importance": "why it matters"}}
    ],
    "algorithms": [
        {{"name": "algorithm name", "description": "what it does", "complexity": "if mentioned"}}
    ],
    "datasets": ["list of datasets mentioned"],
    "metrics": ["list of evaluation metrics used"],
    "code_requirements": {{
        "languages": ["programming languages likely needed"],
        "libraries": ["libraries/frameworks mentioned or implied"],
        "compute": "estimated compute requirements"
    }}
}}

Remember to:
1. Use exact terminology from the paper
2. Include mathematical notation where relevant
3. Distinguish between core contributions and background
4. Note any novel techniques or architectures
"""


def build_mapping_prompt(
    concept: Dict[str, Any],
    code_elements: List[Dict[str, Any]],
    context: str = ""
) -> str:
    """Build prompt for concept-to-code mapping with examples."""
    
    elements_text = "\n".join([
        f"- {e['type']}: {e['name']} (in {e.get('file', 'unknown')})"
        + (f"\n  Docstring: {e.get('docstring', '')[:100]}..." if e.get('docstring') else "")
        for e in code_elements[:30]
    ])
    
    return f"""Map this paper concept to code elements in the repository.

PAPER CONCEPT:
Name: {concept.get('name', 'Unknown')}
Type: {concept.get('type', 'concept')}
Description: {concept.get('description', '')}

AVAILABLE CODE ELEMENTS:
{elements_text}

{f"ADDITIONAL CONTEXT:{chr(10)}{context}" if context else ""}

For each potential mapping, provide:
{{
    "mappings": [
        {{
            "concept_name": "{concept.get('name', '')}",
            "code_element": "element name",
            "code_type": "class/function/module",
            "file_path": "path/to/file.py",
            "confidence": 0.0-1.0,
            "evidence": ["list of evidence supporting this mapping"],
            "gaps": ["any gaps between paper description and implementation"]
        }}
    ]
}}

Guidelines:
1. Only include mappings with confidence > 0.3
2. Provide specific evidence for each mapping
3. Note any discrepancies between paper and code
4. Consider both naming AND functionality
"""


def build_code_generation_prompt(
    concepts: List[Dict[str, Any]],
    mappings: List[Dict[str, Any]],
    repo_info: Dict[str, Any],
    paper_summary: str
) -> str:
    """Build prompt for code generation with examples."""
    
    concepts_text = "\n".join([
        f"- {c.get('name', 'Unknown')}: {c.get('description', '')[:100]}..."
        for c in concepts[:10]
    ])
    
    mappings_text = "\n".join([
        f"- {m.get('concept_name', '')} → {m.get('code_element', '')} ({m.get('confidence', 0):.0%} confidence)"
        for m in mappings[:10]
    ])
    
    return f"""Generate test scripts to demonstrate paper concepts using the repository.

PAPER SUMMARY:
{paper_summary[:1000]}

KEY CONCEPTS:
{concepts_text}

CODE MAPPINGS:
{mappings_text}

REPOSITORY:
- Language: {repo_info.get('main_language', 'python')}
- Entry points: {', '.join(repo_info.get('entry_points', [])[:3])}

Generate a comprehensive test script that:
1. Imports necessary modules from the repository
2. Creates appropriate test data
3. Demonstrates each key concept
4. Verifies expected behavior
5. Prints clear output

IMPORTANT:
- Code must be syntactically correct Python
- Include proper error handling
- Add docstrings and comments
- Use type hints where appropriate

Return as JSON:
{{
    "main_script": {{
        "filename": "test_concepts.py",
        "content": "full python code here",
        "purpose": "what it demonstrates",
        "dependencies": ["list", "of", "imports"]
    }},
    "visualization_script": {{
        "filename": "visualize.py",
        "content": "visualization code",
        "purpose": "what it visualizes",
        "dependencies": ["matplotlib", "etc"]
    }}
}}
"""


# =============================================================================
# AGENT PROMPT GETTER
# =============================================================================

AGENT_PROMPTS = {
    AgentType.PAPER_PARSER: PAPER_PARSER_SYSTEM_PROMPT,
    AgentType.REPO_ANALYZER: REPO_ANALYZER_SYSTEM_PROMPT,
    AgentType.CODING: CODING_AGENT_SYSTEM_PROMPT,
    AgentType.ORCHESTRATOR: ORCHESTRATOR_SYSTEM_PROMPT,
}


def get_agent_prompt(agent_type: AgentType) -> str:
    """Get the system prompt for an agent type."""
    return AGENT_PROMPTS.get(agent_type, "")


def build_task_prompt(
    task_type: str,
    **kwargs
) -> str:
    """Build a task-specific prompt."""
    
    builders = {
        'concept_extraction': build_concept_extraction_prompt,
        'mapping': build_mapping_prompt,
        'code_generation': build_code_generation_prompt,
    }
    
    builder = builders.get(task_type)
    if builder:
        return builder(**kwargs)
    
    return ""


# =============================================================================
# PROMPT TEMPLATES FOR SPECIFIC TASKS
# =============================================================================

REPO_STRUCTURE_ANALYSIS_PROMPT = """Analyze this repository structure and provide insights:

Repository: {repo_name}
Main Language: {language}

Files:
{file_list}

Dependencies:
{dependencies}

Entry Points: {entry_points}

README:
{readme}

Provide analysis in JSON format:
{{
    "purpose": "what this repository does",
    "architecture": "overall architecture pattern",
    "main_components": [
        {{"name": "component", "type": "class/module", "purpose": "what it does", "file": "path"}}
    ],
    "key_algorithms": ["algorithms implemented"],
    "compute_requirements": {{
        "cpu": "requirements",
        "memory": "estimate",
        "gpu": "if needed"
    }},
    "suggested_tests": ["test scenarios"]
}}
"""

MAPPING_ANALYSIS_PROMPT = """Analyze the relationship between this paper concept and code element:

PAPER CONCEPT:
- Name: {concept_name}
- Description: {concept_description}
- From section: {paper_section}

CODE ELEMENT:
- Name: {code_name}
- Type: {code_type}
- File: {file_path}
- Code:
```
{code_snippet}
```

Determine:
1. Does this code implement this concept? (yes/partial/no)
2. Confidence level (0.0-1.0)
3. Evidence supporting your conclusion
4. Gaps between paper and implementation
5. Suggested test approach
"""

PAPER_CONCEPT_EXTRACTION_PROMPT = """Extract key concepts from this scientific paper section:

Section: {section_title}
Content:
{section_content}

Identify:
1. Key concepts introduced or discussed
2. Algorithms or methods described
3. Mathematical formulations
4. Important parameters or hyperparameters
5. Relationships to other concepts

Return as structured JSON.
"""

CODE_GENERATION_TASK_PROMPT = """Generate a test script for this concept:

CONCEPT: {concept_name}
DESCRIPTION: {concept_description}
PAPER SECTION: {paper_section}
CODE LOCATION: {code_location}

Requirements:
1. Import the relevant module
2. Create test data
3. Execute the implementation
4. Verify output
5. Generate visualization if applicable

The script should be educational and clearly demonstrate the concept.
"""