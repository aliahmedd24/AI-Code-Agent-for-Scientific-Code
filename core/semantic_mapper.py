"""
Enhanced Semantic Mapper - Multi-Signal Concept-to-Code Mapping

This module provides accurate concept-to-code mapping by:
- Using semantic embeddings for similarity matching
- Analyzing AST patterns for structural matching
- Parsing docstrings and comments for documentation matching
- Combining multiple signals with weighted scoring
- Providing confidence scores and evidence trails

Author: Scientific Agent System
"""

import os
import re
import ast
import json
import asyncio
from typing import Optional, Any, Dict, List, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)


@dataclass
class MappingEvidence:
    """Evidence supporting a concept-to-code mapping."""
    evidence_type: str  # lexical, semantic, structural, documentary, behavioral
    score: float
    detail: str
    source: str = ""


@dataclass 
class ConceptCodeMapping:
    """A mapping between a paper concept and code element."""
    concept_name: str
    concept_type: str  # algorithm, data_structure, metric, component
    concept_description: str
    code_element: str
    code_type: str  # function, class, module, variable
    file_path: str
    confidence: float
    evidence: List[MappingEvidence] = field(default_factory=list)
    reasoning: str = ""
    code_snippet: str = ""
    
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
            'reasoning': self.reasoning,
            'code_snippet': self.code_snippet[:500] if self.code_snippet else ""
        }


@dataclass
class CodeElement:
    """Represents a code element for mapping."""
    name: str
    element_type: str  # function, class, module
    file_path: str
    content: str
    docstring: str = ""
    comments: str = ""
    imports: List[str] = field(default_factory=list)
    parameters: List[str] = field(default_factory=list)
    return_type: str = ""
    decorators: List[str] = field(default_factory=list)
    parent_class: str = ""
    ast_features: Dict[str, Any] = field(default_factory=dict)


class LexicalMatcher:
    """Match concepts to code using lexical similarity."""
    
    # Common terms that shouldn't drive matching alone
    STOP_TERMS = {
        'model', 'data', 'train', 'test', 'input', 'output', 'process',
        'compute', 'calculate', 'get', 'set', 'init', 'run', 'execute',
        'load', 'save', 'read', 'write', 'create', 'update', 'delete',
        'main', 'helper', 'utils', 'util', 'base', 'config', 'params'
    }
    
    def __init__(self):
        self.term_cache: Dict[str, Set[str]] = {}
    
    def extract_terms(self, text: str) -> Set[str]:
        """Extract meaningful terms from text."""
        if text in self.term_cache:
            return self.term_cache[text]
        
        # Tokenize
        words = re.findall(r'[a-zA-Z][a-z]*|[A-Z]+(?=[A-Z][a-z]|\b)', text)
        
        # Lowercase and filter
        terms = {
            w.lower() for w in words 
            if len(w) > 2 and w.lower() not in self.STOP_TERMS
        }
        
        self.term_cache[text] = terms
        return terms
    
    def compute_similarity(
        self, 
        concept: str, 
        concept_desc: str,
        code_name: str, 
        code_docs: str
    ) -> Tuple[float, str]:
        """Compute lexical similarity between concept and code."""
        # Extract terms
        concept_terms = self.extract_terms(concept + " " + concept_desc)
        code_terms = self.extract_terms(code_name + " " + code_docs)
        
        if not concept_terms or not code_terms:
            return 0.0, "No meaningful terms to compare"
        
        # Compute overlap
        overlap = concept_terms & code_terms
        
        if not overlap:
            return 0.0, "No term overlap"
        
        # Jaccard-like similarity
        union = concept_terms | code_terms
        similarity = len(overlap) / len(union)
        
        # Bonus for name match
        concept_name_terms = self.extract_terms(concept)
        code_name_terms = self.extract_terms(code_name)
        if concept_name_terms & code_name_terms:
            similarity = min(similarity + 0.2, 1.0)
        
        detail = f"Matching terms: {', '.join(sorted(overlap)[:5])}"
        return similarity, detail
    
    def compute_name_similarity(self, concept: str, code_name: str) -> Tuple[float, str]:
        """Compute direct name similarity."""
        # Normalize names
        concept_normalized = concept.lower().replace(' ', '').replace('_', '').replace('-', '')
        code_normalized = code_name.lower().replace('_', '').replace('-', '')
        
        # Exact match
        if concept_normalized == code_normalized:
            return 1.0, f"Exact name match: {concept} = {code_name}"
        
        # Substring match
        if concept_normalized in code_normalized or code_normalized in concept_normalized:
            return 0.7, f"Substring match: {concept} ≈ {code_name}"
        
        # Sequence matcher similarity
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
        self._embeddings_cache: Dict[str, List[float]] = {}
        self._model_loaded = False
    
    def _ensure_model(self):
        """Lazy load the embedding model."""
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
            logger.warning("Install with: pip install sentence-transformers")
            return False
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return False
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text."""
        if not self._ensure_model():
            return None
        
        if text in self._embeddings_cache:
            return self._embeddings_cache[text]
        
        try:
            embedding = self.model.encode(text).tolist()
            self._embeddings_cache[text] = embedding
            return embedding
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return None
    
    def compute_similarity(
        self,
        concept: str,
        concept_desc: str,
        code_name: str,
        code_docs: str
    ) -> Tuple[float, str]:
        """Compute semantic similarity using embeddings."""
        if not self._ensure_model():
            return 0.0, "Semantic model not available"
        
        try:
            # Combine concept info
            concept_text = f"{concept}: {concept_desc}"
            code_text = f"{code_name}: {code_docs}"
            
            # Get embeddings
            emb1 = self.get_embedding(concept_text)
            emb2 = self.get_embedding(code_text)
            
            if emb1 is None or emb2 is None:
                return 0.0, "Failed to compute embeddings"
            
            # Cosine similarity
            import numpy as np
            emb1 = np.array(emb1)
            emb2 = np.array(emb2)
            similarity = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
            
            return max(similarity, 0.0), f"Semantic similarity: {similarity:.3f}"
            
        except Exception as e:
            logger.error(f"Semantic similarity failed: {e}")
            return 0.0, f"Error: {e}"


class StructuralMatcher:
    """Match concepts to code using AST structural analysis."""
    
    # Patterns that indicate certain algorithm types
    ALGORITHM_PATTERNS = {
        'attention': {
            'softmax': True,
            'dot_product': True,
            'transpose': True,
            'matmul': True
        },
        'normalization': {
            'mean': True,
            'std': True,
            'sqrt': True,
            'eps': True
        },
        'dropout': {
            'random': True,
            'mask': True,
            'bernoulli': True
        },
        'convolution': {
            'kernel': True,
            'stride': True,
            'padding': True,
            'conv': True
        },
        'recurrent': {
            'hidden': True,
            'cell': True,
            'forget': True,
            'gate': True
        },
        'transformer': {
            'attention': True,
            'encoder': True,
            'decoder': True,
            'positional': True
        },
        'embedding': {
            'lookup': True,
            'index': True,
            'vocab': True,
            'embed': True
        }
    }
    
    def extract_ast_features(self, code: str) -> Dict[str, Any]:
        """Extract structural features from Python code."""
        features = {
            'has_loops': False,
            'has_recursion': False,
            'has_matrix_ops': False,
            'num_functions': 0,
            'num_classes': 0,
            'call_patterns': [],
            'operations': set(),
            'control_flow': []
        }
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return features
        
        class FeatureVisitor(ast.NodeVisitor):
            def __init__(self):
                self.current_function = None
                self.function_calls = set()
            
            def visit_FunctionDef(self, node):
                features['num_functions'] += 1
                old_func = self.current_function
                self.current_function = node.name
                self.generic_visit(node)
                self.current_function = old_func
            
            def visit_ClassDef(self, node):
                features['num_classes'] += 1
                self.generic_visit(node)
            
            def visit_For(self, node):
                features['has_loops'] = True
                features['control_flow'].append('for')
                self.generic_visit(node)
            
            def visit_While(self, node):
                features['has_loops'] = True
                features['control_flow'].append('while')
                self.generic_visit(node)
            
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    call_name = node.func.id.lower()
                    self.function_calls.add(call_name)
                    features['call_patterns'].append(call_name)
                    
                    # Check for recursion
                    if self.current_function and call_name == self.current_function.lower():
                        features['has_recursion'] = True
                    
                    # Check for matrix operations
                    if call_name in {'matmul', 'dot', 'mm', 'bmm', 'einsum'}:
                        features['has_matrix_ops'] = True
                        features['operations'].add('matrix_multiply')
                    
                    if call_name in {'transpose', 'permute', 'reshape', 'view'}:
                        features['operations'].add('tensor_reshape')
                    
                    if call_name in {'softmax', 'relu', 'sigmoid', 'tanh', 'gelu'}:
                        features['operations'].add(f'activation_{call_name}')
                
                elif isinstance(node.func, ast.Attribute):
                    attr = node.func.attr.lower()
                    features['call_patterns'].append(attr)
                    
                    if attr in {'matmul', 'mm', 'bmm', 'dot'}:
                        features['has_matrix_ops'] = True
                        features['operations'].add('matrix_multiply')
                
                self.generic_visit(node)
            
            def visit_BinOp(self, node):
                if isinstance(node.op, ast.MatMult):
                    features['has_matrix_ops'] = True
                    features['operations'].add('matrix_multiply')
                self.generic_visit(node)
        
        visitor = FeatureVisitor()
        visitor.visit(tree)
        features['operations'] = list(features['operations'])
        
        return features
    
    def match_pattern(self, concept: str, features: Dict[str, Any]) -> Tuple[float, str]:
        """Match concept to AST pattern."""
        concept_lower = concept.lower()
        
        # Check against known patterns
        for pattern_name, required_ops in self.ALGORITHM_PATTERNS.items():
            if pattern_name in concept_lower:
                # Check how many pattern indicators are present
                call_patterns = set(features.get('call_patterns', []))
                operations = set(features.get('operations', []))
                all_patterns = call_patterns | operations
                
                matches = sum(
                    1 for op in required_ops 
                    if any(op in p for p in all_patterns)
                )
                
                if matches >= 2:
                    return 0.7, f"Matches {pattern_name} pattern with {matches} indicators"
                elif matches >= 1:
                    return 0.4, f"Partial {pattern_name} pattern match"
        
        # Check for general structural matches
        if 'loop' in concept_lower and features.get('has_loops'):
            return 0.3, "Contains expected loop structure"
        
        if 'recursive' in concept_lower and features.get('has_recursion'):
            return 0.5, "Contains recursive structure"
        
        if any(term in concept_lower for term in ['matrix', 'tensor', 'linear']) \
           and features.get('has_matrix_ops'):
            return 0.4, "Contains matrix operations"
        
        return 0.0, "No structural pattern match"


class DocumentaryMatcher:
    """Match concepts to code using docstrings and comments."""
    
    def extract_documentation(self, code: str) -> Tuple[str, str]:
        """Extract docstrings and comments from code."""
        docstrings = []
        comments = []
        
        # Extract docstrings
        docstring_pattern = r'"""(.*?)"""|\'\'\'(.*?)\'\'\''
        for match in re.finditer(docstring_pattern, code, re.DOTALL):
            doc = match.group(1) or match.group(2)
            if doc:
                docstrings.append(doc.strip())
        
        # Extract comments
        for line in code.split('\n'):
            if '#' in line:
                comment_start = line.index('#')
                comment = line[comment_start + 1:].strip()
                if comment:
                    comments.append(comment)
        
        return ' '.join(docstrings), ' '.join(comments)
    
    def compute_similarity(
        self,
        concept: str,
        concept_desc: str,
        docstring: str,
        comments: str
    ) -> Tuple[float, str]:
        """Compute similarity based on documentation."""
        if not docstring and not comments:
            return 0.0, "No documentation found"
        
        doc_text = (docstring + " " + comments).lower()
        concept_terms = set(concept.lower().split()) | set(concept_desc.lower().split())
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'for', 'to', 'of', 'and', 'in', 'on', 'this', 'that'}
        concept_terms = {t for t in concept_terms if t not in stop_words and len(t) > 2}
        
        if not concept_terms:
            return 0.0, "No meaningful concept terms"
        
        # Count matches
        matches = [t for t in concept_terms if t in doc_text]
        
        if not matches:
            return 0.0, "No documentation mentions"
        
        # Score based on match ratio
        score = len(matches) / len(concept_terms)
        
        # Bonus if concept name is explicitly mentioned
        if concept.lower() in doc_text:
            score = min(score + 0.3, 1.0)
            return score, f"Concept explicitly mentioned in docs: '{concept}'"
        
        return score, f"Documentation mentions: {', '.join(matches[:3])}"


class EnhancedSemanticMapper:
    """
    Multi-signal concept-to-code mapper.
    
    Combines multiple matching strategies:
    - Lexical: Name and term similarity
    - Semantic: Embedding-based similarity
    - Structural: AST pattern matching
    - Documentary: Docstring/comment analysis
    """
    
    # Weights for different signals
    SIGNAL_WEIGHTS = {
        'lexical': 0.20,
        'semantic': 0.30,
        'structural': 0.25,
        'documentary': 0.25
    }
    
    # Minimum confidence to include a mapping
    MIN_CONFIDENCE = 0.25
    
    def __init__(self, llm_client=None):
        """Initialize the semantic mapper."""
        self.llm_client = llm_client
        self.lexical_matcher = LexicalMatcher()
        self.semantic_matcher = SemanticMatcher()
        self.structural_matcher = StructuralMatcher()
        self.documentary_matcher = DocumentaryMatcher()
    
    async def map_concepts_to_code(
        self,
        concepts: List[Dict[str, Any]],
        code_elements: List[CodeElement]
    ) -> List[ConceptCodeMapping]:
        """
        Map paper concepts to code elements using multi-signal matching.
        
        Args:
            concepts: List of concepts from paper analysis
            code_elements: List of code elements from repo analysis
            
        Returns:
            List of concept-to-code mappings with confidence scores
        """
        all_mappings: List[ConceptCodeMapping] = []
        
        for concept in concepts:
            concept_name = concept.get('name', '')
            concept_type = concept.get('type', 'concept')
            concept_desc = concept.get('description', '')
            
            if not concept_name:
                continue
            
            # Score each code element
            element_scores: List[Tuple[CodeElement, float, List[MappingEvidence]]] = []
            
            for element in code_elements:
                score, evidence = self._score_mapping(
                    concept_name, concept_desc, element
                )
                
                if score >= self.MIN_CONFIDENCE:
                    element_scores.append((element, score, evidence))
            
            # Sort by score and take top matches
            element_scores.sort(key=lambda x: x[1], reverse=True)
            
            for element, score, evidence in element_scores[:3]:  # Top 3 matches per concept
                # Generate reasoning
                reasoning = self._generate_reasoning(concept_name, element, evidence)
                
                mapping = ConceptCodeMapping(
                    concept_name=concept_name,
                    concept_type=concept_type,
                    concept_description=concept_desc,
                    code_element=element.name,
                    code_type=element.element_type,
                    file_path=element.file_path,
                    confidence=score,
                    evidence=evidence,
                    reasoning=reasoning,
                    code_snippet=element.content[:500]
                )
                
                all_mappings.append(mapping)
        
        # Sort all mappings by confidence
        all_mappings.sort(key=lambda m: m.confidence, reverse=True)
        
        return all_mappings
    
    def _score_mapping(
        self,
        concept_name: str,
        concept_desc: str,
        element: CodeElement
    ) -> Tuple[float, List[MappingEvidence]]:
        """Score a potential concept-to-code mapping."""
        evidence: List[MappingEvidence] = []
        
        # Get all documentation
        docstring = element.docstring
        comments = element.comments
        
        # 1. Lexical matching
        lexical_score, lexical_detail = self.lexical_matcher.compute_similarity(
            concept_name, concept_desc, element.name, docstring
        )
        name_score, name_detail = self.lexical_matcher.compute_name_similarity(
            concept_name, element.name
        )
        lexical_score = max(lexical_score, name_score)
        lexical_detail = name_detail if name_score > lexical_score else lexical_detail
        
        if lexical_score > 0:
            evidence.append(MappingEvidence(
                evidence_type='lexical',
                score=lexical_score,
                detail=lexical_detail
            ))
        
        # 2. Semantic matching
        semantic_score, semantic_detail = self.semantic_matcher.compute_similarity(
            concept_name, concept_desc, element.name, docstring
        )
        if semantic_score > 0.3:
            evidence.append(MappingEvidence(
                evidence_type='semantic',
                score=semantic_score,
                detail=semantic_detail
            ))
        
        # 3. Structural matching
        features = element.ast_features or self.structural_matcher.extract_ast_features(element.content)
        structural_score, structural_detail = self.structural_matcher.match_pattern(
            concept_name, features
        )
        if structural_score > 0:
            evidence.append(MappingEvidence(
                evidence_type='structural',
                score=structural_score,
                detail=structural_detail
            ))
        
        # 4. Documentary matching
        doc_score, doc_detail = self.documentary_matcher.compute_similarity(
            concept_name, concept_desc, docstring, comments
        )
        if doc_score > 0:
            evidence.append(MappingEvidence(
                evidence_type='documentary',
                score=doc_score,
                detail=doc_detail
            ))
        
        # Compute weighted score
        total_score = 0.0
        total_weight = 0.0
        
        for e in evidence:
            weight = self.SIGNAL_WEIGHTS.get(e.evidence_type, 0.1)
            total_score += e.score * weight
            total_weight += weight
        
        if total_weight > 0:
            # Normalize
            final_score = total_score / total_weight
            
            # Bonus for multiple evidence types
            if len(evidence) >= 3:
                final_score = min(final_score * 1.2, 1.0)
            elif len(evidence) >= 2:
                final_score = min(final_score * 1.1, 1.0)
        else:
            final_score = 0.0
        
        return final_score, evidence
    
    def _generate_reasoning(
        self,
        concept_name: str,
        element: CodeElement,
        evidence: List[MappingEvidence]
    ) -> str:
        """Generate human-readable reasoning for the mapping."""
        parts = [f"The {element.element_type} '{element.name}' in {element.file_path} "]
        
        if not evidence:
            return parts[0] + "has weak connection to the concept."
        
        evidence_strs = []
        for e in sorted(evidence, key=lambda x: x.score, reverse=True):
            if e.evidence_type == 'lexical':
                evidence_strs.append(f"name similarity ({e.detail})")
            elif e.evidence_type == 'semantic':
                evidence_strs.append(f"semantic similarity to concept description")
            elif e.evidence_type == 'structural':
                evidence_strs.append(f"structural patterns ({e.detail})")
            elif e.evidence_type == 'documentary':
                evidence_strs.append(f"documentation ({e.detail})")
        
        if evidence_strs:
            parts.append("likely implements the concept based on: ")
            parts.append(", ".join(evidence_strs[:3]))
        
        return "".join(parts)
    
    async def enhance_with_llm(
        self,
        mappings: List[ConceptCodeMapping],
        paper_context: str
    ) -> List[ConceptCodeMapping]:
        """Use LLM to verify and enhance mappings."""
        if not self.llm_client or not mappings:
            return mappings
        
        # Group mappings by concept for efficient LLM calls
        concept_groups: Dict[str, List[ConceptCodeMapping]] = defaultdict(list)
        for mapping in mappings:
            concept_groups[mapping.concept_name].append(mapping)
        
        enhanced_mappings = []
        
        for concept_name, group in concept_groups.items():
            # Prepare context for LLM
            code_candidates = "\n".join([
                f"- {m.code_element} ({m.code_type}) in {m.file_path}: confidence {m.confidence:.2f}"
                for m in group
            ])
            
            prompt = f"""Verify these concept-to-code mappings:

Concept: {concept_name}
Description: {group[0].concept_description}

Paper context:
{paper_context[:1000]}

Code candidates:
{code_candidates}

For each candidate, assess:
1. Is this a valid mapping? (yes/partial/no)
2. Confidence adjustment (-0.2 to +0.2)
3. Brief reasoning

Return JSON:
{{
    "assessments": [
        {{"code_element": "name", "valid": "yes|partial|no", "adjustment": 0.0, "reasoning": "..."}}
    ]
}}
"""
            
            try:
                result = await self.llm_client.generate_structured(
                    prompt,
                    schema={"type": "object"}
                )
                
                assessments = {
                    a['code_element']: a 
                    for a in result.get('assessments', [])
                }
                
                for mapping in group:
                    assessment = assessments.get(mapping.code_element, {})
                    
                    # Apply LLM adjustments
                    if assessment:
                        adjustment = assessment.get('adjustment', 0)
                        mapping.confidence = max(0, min(1, mapping.confidence + adjustment))
                        
                        if assessment.get('reasoning'):
                            mapping.reasoning += f" LLM: {assessment['reasoning']}"
                        
                        # Reduce confidence for "no" mappings
                        if assessment.get('valid') == 'no':
                            mapping.confidence *= 0.3
                        elif assessment.get('valid') == 'partial':
                            mapping.confidence *= 0.7
                    
                    enhanced_mappings.append(mapping)
                    
            except Exception as e:
                logger.warning(f"LLM enhancement failed: {e}")
                enhanced_mappings.extend(group)
        
        # Re-sort by confidence
        enhanced_mappings.sort(key=lambda m: m.confidence, reverse=True)
        
        return enhanced_mappings


def create_semantic_mapper(llm_client=None) -> EnhancedSemanticMapper:
    """Factory function to create an enhanced semantic mapper."""
    return EnhancedSemanticMapper(llm_client=llm_client)
