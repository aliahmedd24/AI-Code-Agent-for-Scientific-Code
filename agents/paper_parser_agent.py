"""
Paper Parser Agent - Scientific Paper Analysis and Understanding

This agent is responsible for:
- Downloading and parsing scientific papers (PDF, arXiv)
- Extracting structured information (sections, equations, figures)
- Understanding key concepts and methodologies
- Building knowledge graph nodes for paper content
- Connecting paper concepts to code implementations
"""

import os
import re
import json
import asyncio
import tempfile
from typing import Optional, Any, Dict, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import httpx
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
import arxiv

from core.gemini_client import AgentLLM, GeminiConfig, GeminiModel
from core.knowledge_graph import (
    KnowledgeGraph, NodeType, EdgeType, 
    get_global_graph
)
from core.agent_prompts import (
    get_agent_prompt, AgentType, 
    PAPER_CONCEPT_EXTRACTION_PROMPT,
    build_task_prompt
)


@dataclass
class PaperSection:
    """Represents a section of a paper."""
    title: str
    content: str
    level: int = 1
    subsections: List["PaperSection"] = field(default_factory=list)
    equations: List[str] = field(default_factory=list)
    figures: List[Dict[str, str]] = field(default_factory=list)
    tables: List[Dict[str, str]] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)


@dataclass
class ParsedPaper:
    """Complete parsed paper structure."""
    title: str
    authors: List[str]
    abstract: str
    sections: List[PaperSection]
    references: List[Dict[str, str]]
    keywords: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    source_url: str = ""
    
    def get_full_text(self) -> str:
        """Get the complete text of the paper."""
        parts = [self.title, self.abstract]
        
        def extract_section_text(section: PaperSection) -> str:
            text = f"{section.title}\n{section.content}"
            for sub in section.subsections:
                text += "\n" + extract_section_text(sub)
            return text
        
        for section in self.sections:
            parts.append(extract_section_text(section))
        
        return "\n\n".join(parts)


class PaperParserAgent:
    """
    Agent for parsing and understanding scientific papers.
    
    Capabilities:
    - Download papers from URLs, arXiv, etc.
    - Extract text, equations, figures from PDFs
    - Use LLM to understand and structure content
    - Build knowledge graph from paper content
    - Identify key concepts for code mapping
    """
    
    def __init__(
        self,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        gemini_api_key: Optional[str] = None
    ):
        self.knowledge_graph = knowledge_graph or get_global_graph()
        
        # Get specialized system prompt for this agent
        system_prompt = get_agent_prompt(AgentType.PAPER_PARSER)
        
        self.llm = AgentLLM(
            agent_name="PaperParser",
            agent_role="Scientific paper analysis and knowledge extraction",
            api_key=gemini_api_key,
            config=GeminiConfig(
                model=GeminiModel.FLASH,
                temperature=0.3,  # Lower temperature for accurate extraction
                max_output_tokens=8192,
                system_instruction=system_prompt  # Use specialized prompt
            )
        )
        self.parsed_papers: Dict[str, ParsedPaper] = {}
    
    async def parse_paper(self, source: str) -> ParsedPaper:
        """
        Parse a scientific paper from various sources.
        
        Args:
            source: URL, file path, or arXiv ID
        
        Returns:
            ParsedPaper object
        """
        # Determine source type and download if needed
        if source.startswith("http"):
            paper_content = await self._download_paper(source)
        elif "arxiv" in source.lower() or re.match(r'^\d+\.\d+', source):
            paper_content = await self._fetch_arxiv_paper(source)
        else:
            paper_content = await self._read_local_file(source)
        
        # Parse the PDF content
        parsed = await self._parse_pdf_content(paper_content["path"], paper_content.get("metadata", {}))
        parsed.source_url = source
        
        # Use LLM to enhance understanding
        enhanced = await self._enhance_with_llm(parsed)
        
        # Build knowledge graph
        await self._build_knowledge_graph(enhanced)
        
        # Store for later reference
        self.parsed_papers[source] = enhanced
        
        return enhanced
    
    async def _download_paper(self, url: str) -> Dict[str, Any]:
        """Download a paper from a URL."""
        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            # Save to temp file
            suffix = ".pdf" if "pdf" in url.lower() or response.headers.get("content-type", "").startswith("application/pdf") else ".html"
            
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(response.content)
                return {"path": f.name, "metadata": {"url": url}}
    
    async def _fetch_arxiv_paper(self, arxiv_id: str) -> Dict[str, Any]:
        """Fetch a paper from arXiv."""
        # Clean up the ID
        arxiv_id = arxiv_id.replace("arxiv:", "").replace("arXiv:", "")
        if "arxiv.org" in arxiv_id:
            match = re.search(r'(\d+\.\d+)', arxiv_id)
            if match:
                arxiv_id = match.group(1)
        
        # Search for the paper
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())
        
        # Download PDF
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            paper.download_pdf(filename=f.name)
            
            return {
                "path": f.name,
                "metadata": {
                    "arxiv_id": arxiv_id,
                    "title": paper.title,
                    "authors": [str(a) for a in paper.authors],
                    "abstract": paper.summary,
                    "categories": paper.categories,
                    "published": str(paper.published),
                    "updated": str(paper.updated),
                    "doi": paper.doi,
                    "pdf_url": paper.pdf_url
                }
            }
    
    async def _read_local_file(self, path: str) -> Dict[str, Any]:
        """Read a local PDF file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Paper not found: {path}")
        return {"path": path, "metadata": {"local_path": path}}
    
    async def _parse_pdf_content(
        self,
        pdf_path: str,
        metadata: Dict[str, Any]
    ) -> ParsedPaper:
        """Extract content from a PDF file."""
        doc = fitz.open(pdf_path)
        
        # Extract full text
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        
        # Try to extract title from first page
        first_page_text = doc[0].get_text()
        lines = [l.strip() for l in first_page_text.split('\n') if l.strip()]
        
        # Use metadata if available, otherwise extract from text
        title = metadata.get("title", lines[0] if lines else "Untitled")
        authors = metadata.get("authors", [])
        abstract = metadata.get("abstract", "")
        
        # If no abstract from metadata, try to find it
        if not abstract:
            abstract_match = re.search(
                r'(?:abstract|summary)[:\s]*(.+?)(?:introduction|1\.|keywords|$)',
                full_text,
                re.IGNORECASE | re.DOTALL
            )
            if abstract_match:
                abstract = abstract_match.group(1).strip()[:2000]
        
        # Extract sections using regex patterns
        sections = self._extract_sections(full_text)
        
        # Extract equations (LaTeX patterns)
        equations = self._extract_equations(full_text)
        
        # Extract references
        references = self._extract_references(full_text)
        
        doc.close()
        
        return ParsedPaper(
            title=title,
            authors=authors,
            abstract=abstract,
            sections=sections,
            references=references,
            keywords=[],
            metadata=metadata,
            raw_text=full_text
        )
    
    def _extract_sections(self, text: str) -> List[PaperSection]:
        """Extract sections from paper text."""
        sections = []
        
        # Common section patterns
        section_patterns = [
            r'^(\d+\.?\s+(?:Introduction|Background|Related Work|Methodology|Method|Methods|Approach|Experiments|Results|Discussion|Conclusion|Conclusions|References|Acknowledgments))',
            r'^((?:I{1,3}|IV|V|VI|VII|VIII|IX|X)\.?\s+\w+)',
            r'^(\d+\.\d+\.?\s+\w+)',
        ]
        
        lines = text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            is_section_header = False
            
            for pattern in section_patterns:
                if re.match(pattern, line.strip(), re.IGNORECASE):
                    # Save previous section
                    if current_section:
                        current_section.content = '\n'.join(current_content)
                        sections.append(current_section)
                    
                    # Start new section
                    current_section = PaperSection(
                        title=line.strip(),
                        content="",
                        level=1 if re.match(r'^\d+\.?\s', line) else 2
                    )
                    current_content = []
                    is_section_header = True
                    break
            
            if not is_section_header and current_section:
                current_content.append(line)
        
        # Add last section
        if current_section:
            current_section.content = '\n'.join(current_content)
            sections.append(current_section)
        
        # If no sections found, create a single section with all content
        if not sections:
            sections.append(PaperSection(
                title="Content",
                content=text[:10000],  # Limit size
                level=1
            ))
        
        return sections
    
    def _extract_equations(self, text: str) -> List[str]:
        """Extract equations from text."""
        equations = []
        
        # LaTeX-style equations
        patterns = [
            r'\$\$(.+?)\$\$',
            r'\$(.+?)\$',
            r'\\begin\{equation\}(.+?)\\end\{equation\}',
            r'\\begin\{align\}(.+?)\\end\{align\}',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            equations.extend(matches)
        
        return equations[:50]  # Limit number of equations
    
    def _extract_references(self, text: str) -> List[Dict[str, str]]:
        """Extract references from paper."""
        references = []
        
        # Find references section
        ref_match = re.search(
            r'(?:references|bibliography)[:\s]*(.+?)(?:\Z|acknowledgments)',
            text,
            re.IGNORECASE | re.DOTALL
        )
        
        if ref_match:
            ref_text = ref_match.group(1)
            
            # Try to split by numbered references
            ref_items = re.split(r'\[\d+\]|\n\d+\.', ref_text)
            
            for item in ref_items:
                item = item.strip()
                if len(item) > 20:  # Filter out noise
                    references.append({
                        "raw": item[:500],
                        "parsed": False
                    })
        
        return references[:100]  # Limit number of references
    
    async def _enhance_with_llm(self, paper: ParsedPaper) -> ParsedPaper:
        """Use LLM to enhance paper understanding."""
        # Prepare a summary of the paper for analysis
        content_summary = f"""
Title: {paper.title}

Abstract: {paper.abstract}

Sections:
{chr(10).join([f"- {s.title}" for s in paper.sections])}

Sample content (first 3000 chars):
{paper.raw_text[:3000]}
"""
        
        # Ask LLM to extract key information
        analysis_prompt = f"""
Analyze this scientific paper and extract key information.

{content_summary}

Provide your analysis in JSON format:
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
    }},
    "reproducibility_info": {{
        "has_code": true/false,
        "has_data": true/false,
        "clarity": "high/medium/low"
    }}
}}
"""
        
        try:
            analysis = await self.llm.generate_structured(
                analysis_prompt,
                schema={"type": "object"}
                # System instruction already set in config - uses specialized PaperParser prompt
            )
            
            # Update paper with extracted information
            paper.keywords = analysis.get("keywords", [])
            paper.metadata["llm_analysis"] = analysis
            
        except Exception as e:
            error_msg = str(e)
            print(f"LLM enhancement failed: {error_msg}")
            
            # Graceful degradation - continue with basic analysis
            paper.metadata["llm_analysis"] = {
                "error": error_msg,
                "fallback": True,
                "key_concepts": [],
                "algorithms": [],
                "methodology": "LLM analysis unavailable - basic extraction only"
            }
            
            # Extract basic keywords from abstract
            if paper.abstract:
                words = paper.abstract.lower().split()
                # Simple keyword extraction
                common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                               'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                               'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                               'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                               'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                               'through', 'during', 'before', 'after', 'above', 'below',
                               'between', 'under', 'again', 'further', 'then', 'once',
                               'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
                               'neither', 'not', 'only', 'own', 'same', 'than', 'too',
                               'very', 'just', 'also', 'now', 'here', 'there', 'when',
                               'where', 'why', 'how', 'all', 'each', 'every', 'both',
                               'few', 'more', 'most', 'other', 'some', 'such', 'no',
                               'any', 'this', 'that', 'these', 'those', 'we', 'our',
                               'they', 'their', 'it', 'its', 'which', 'who', 'whom'}
                paper.keywords = [w for w in set(words) 
                                 if len(w) > 4 and w not in common_words][:20]
        
        return paper
    
    async def _build_knowledge_graph(self, paper: ParsedPaper):
        """Build knowledge graph nodes and edges from paper."""
        kg = self.knowledge_graph
        
        # Add paper node
        paper_id = await kg.add_node(
            node_type=NodeType.PAPER,
            name=paper.title,
            content=paper.abstract,
            metadata={
                "authors": paper.authors,
                "keywords": paper.keywords,
                "source_url": paper.source_url
            },
            source="paper_parser"
        )
        
        # Add author nodes
        for author in paper.authors:
            author_id = await kg.add_node(
                node_type=NodeType.AUTHOR,
                name=author,
                source="paper_parser"
            )
            await kg.add_edge(
                paper_id, author_id,
                EdgeType.CREATED_BY,
                created_by="paper_parser"
            )
        
        # Add section nodes
        for section in paper.sections:
            section_id = await kg.add_node(
                node_type=NodeType.SECTION,
                name=section.title,
                content=section.content[:2000],  # Limit content size
                source="paper_parser"
            )
            await kg.add_edge(
                paper_id, section_id,
                EdgeType.CONTAINS,
                created_by="paper_parser"
            )
        
        # Add concept nodes from LLM analysis
        if "llm_analysis" in paper.metadata:
            analysis = paper.metadata["llm_analysis"]
            
            for concept in analysis.get("key_concepts", []):
                concept_id = await kg.add_node(
                    node_type=NodeType.CONCEPT,
                    name=concept.get("name", ""),
                    content=concept.get("description", ""),
                    metadata={"importance": concept.get("importance", "")},
                    source="paper_parser"
                )
                await kg.add_edge(
                    paper_id, concept_id,
                    EdgeType.CONTAINS,
                    created_by="paper_parser"
                )
            
            for algo in analysis.get("algorithms", []):
                algo_id = await kg.add_node(
                    node_type=NodeType.CONCEPT,
                    name=algo.get("name", ""),
                    content=algo.get("description", ""),
                    metadata={
                        "type": "algorithm",
                        "complexity": algo.get("complexity", "")
                    },
                    source="paper_parser"
                )
                await kg.add_edge(
                    paper_id, algo_id,
                    EdgeType.CONTAINS,
                    created_by="paper_parser"
                )
    
    async def get_concepts_for_code_mapping(self, paper_source: str) -> List[Dict[str, Any]]:
        """
        Get concepts from a parsed paper that should be mapped to code.
        
        Returns list of concepts with their descriptions and expected
        code implementations.
        """
        if paper_source not in self.parsed_papers:
            await self.parse_paper(paper_source)
        
        paper = self.parsed_papers[paper_source]
        analysis = paper.metadata.get("llm_analysis", {})
        
        concepts = []
        
        # Key concepts
        for concept in analysis.get("key_concepts", []):
            concepts.append({
                "name": concept.get("name"),
                "type": "concept",
                "description": concept.get("description"),
                "importance": concept.get("importance"),
                "expected_implementation": f"Implementation of {concept.get('name')}"
            })
        
        # Algorithms
        for algo in analysis.get("algorithms", []):
            concepts.append({
                "name": algo.get("name"),
                "type": "algorithm",
                "description": algo.get("description"),
                "complexity": algo.get("complexity"),
                "expected_implementation": f"Function or class implementing {algo.get('name')}"
            })
        
        return concepts
    
    async def generate_paper_summary(self, paper_source: str) -> str:
        """Generate a comprehensive summary of the paper."""
        if paper_source not in self.parsed_papers:
            await self.parse_paper(paper_source)
        
        paper = self.parsed_papers[paper_source]
        
        summary_prompt = f"""
Create a comprehensive technical summary of this scientific paper.

Title: {paper.title}
Authors: {', '.join(paper.authors)}
Abstract: {paper.abstract}

Key sections:
{chr(10).join([f"## {s.title}{chr(10)}{s.content[:500]}..." for s in paper.sections[:5]])}

Provide a summary that covers:
1. Main problem being addressed
2. Key contributions and novelty
3. Methodology overview
4. Main results and findings
5. Practical implications
6. Potential code implementation considerations

Write in clear, technical language suitable for a developer who will implement this.
"""
        
        response = await self.llm.generate(
            summary_prompt,
            system_instruction="You are a technical writer who creates clear, accurate summaries of scientific papers for software developers."
        )
        
        return response.content
    
    def get_paper_info(self, paper_source: str) -> Optional[Dict[str, Any]]:
        """Get stored information about a parsed paper."""
        if paper_source in self.parsed_papers:
            paper = self.parsed_papers[paper_source]
            return {
                "title": paper.title,
                "authors": paper.authors,
                "abstract": paper.abstract,
                "sections": [s.title for s in paper.sections],
                "keywords": paper.keywords,
                "analysis": paper.metadata.get("llm_analysis", {}),
                "source_url": paper.source_url
            }
        return None
