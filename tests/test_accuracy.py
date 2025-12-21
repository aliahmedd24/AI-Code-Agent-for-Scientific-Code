"""
Test Suite - Accuracy Benchmarks and Validation Tests

This module provides:
- Unit tests for all core components
- Integration tests for the complete pipeline
- Benchmark fixtures with known paper-code pairs
- Accuracy metrics and reporting

Author: Scientific Agent System
"""

import os
import sys
import json
import asyncio
import tempfile
from typing import Optional, Any, Dict, List, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import logging
import unittest

logger = logging.getLogger(__name__)


# =============================================================================
# BENCHMARK FIXTURES
# =============================================================================

@dataclass
class BenchmarkFixture:
    """A known paper-code pair for benchmarking."""
    name: str
    paper_url: str
    repo_url: str
    expected_concepts: List[str]
    expected_mappings: List[Dict[str, str]]
    expected_algorithms: List[str] = field(default_factory=list)
    difficulty: str = "medium"  # easy, medium, hard
    notes: str = ""


# Well-known paper-code pairs for benchmarking
BENCHMARK_FIXTURES = [
    BenchmarkFixture(
        name="Attention Is All You Need",
        paper_url="https://arxiv.org/abs/1706.03762",
        repo_url="https://github.com/tensorflow/tensor2tensor",
        expected_concepts=[
            "Transformer",
            "Self-Attention",
            "Multi-Head Attention",
            "Positional Encoding",
            "Feed-Forward Network",
            "Layer Normalization",
            "Encoder",
            "Decoder"
        ],
        expected_mappings=[
            {"concept": "Multi-Head Attention", "code_pattern": "multihead|multi_head|attention"},
            {"concept": "Positional Encoding", "code_pattern": "position|positional|timing"},
            {"concept": "Layer Normalization", "code_pattern": "layer_norm|layernorm"},
            {"concept": "Feed-Forward", "code_pattern": "feed_forward|ffn|mlp"},
        ],
        expected_algorithms=["Scaled Dot-Product Attention", "Softmax"],
        difficulty="medium",
        notes="Foundational transformer paper"
    ),
    BenchmarkFixture(
        name="BERT",
        paper_url="https://arxiv.org/abs/1810.04805",
        repo_url="https://github.com/google-research/bert",
        expected_concepts=[
            "BERT",
            "Masked Language Model",
            "Next Sentence Prediction",
            "WordPiece Tokenization",
            "Pre-training",
            "Fine-tuning"
        ],
        expected_mappings=[
            {"concept": "BERT", "code_pattern": "bert|BertModel"},
            {"concept": "Masked Language Model", "code_pattern": "mask|mlm"},
            {"concept": "WordPiece", "code_pattern": "wordpiece|tokeniz"},
        ],
        expected_algorithms=["Masked LM", "NSP"],
        difficulty="medium"
    ),
    BenchmarkFixture(
        name="ResNet",
        paper_url="https://arxiv.org/abs/1512.03385",
        repo_url="https://github.com/pytorch/vision",
        expected_concepts=[
            "Residual Learning",
            "Skip Connection",
            "Batch Normalization",
            "Bottleneck",
            "Identity Mapping"
        ],
        expected_mappings=[
            {"concept": "Residual", "code_pattern": "resnet|residual"},
            {"concept": "Skip Connection", "code_pattern": "skip|shortcut|identity"},
            {"concept": "Bottleneck", "code_pattern": "bottleneck"},
        ],
        difficulty="easy"
    ),
    BenchmarkFixture(
        name="GPT-2",
        paper_url="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
        repo_url="https://github.com/openai/gpt-2",
        expected_concepts=[
            "Language Model",
            "Transformer Decoder",
            "Byte Pair Encoding",
            "Autoregressive"
        ],
        expected_mappings=[
            {"concept": "GPT", "code_pattern": "gpt|model"},
            {"concept": "BPE", "code_pattern": "bpe|encoder"},
        ],
        difficulty="medium"
    ),
    BenchmarkFixture(
        name="YOLO",
        paper_url="https://arxiv.org/abs/1506.02640",
        repo_url="https://github.com/ultralytics/yolov5",
        expected_concepts=[
            "Object Detection",
            "Bounding Box",
            "Non-Maximum Suppression",
            "Anchor Box",
            "Grid Cell"
        ],
        expected_mappings=[
            {"concept": "YOLO", "code_pattern": "yolo|detect"},
            {"concept": "NMS", "code_pattern": "nms|non_max"},
            {"concept": "Bounding Box", "code_pattern": "box|bbox"},
        ],
        difficulty="medium"
    ),
]


# =============================================================================
# ACCURACY METRICS
# =============================================================================

@dataclass
class AccuracyMetrics:
    """Metrics for evaluating system accuracy."""
    concept_extraction_precision: float = 0.0
    concept_extraction_recall: float = 0.0
    concept_extraction_f1: float = 0.0
    
    mapping_precision: float = 0.0
    mapping_recall: float = 0.0
    mapping_f1: float = 0.0
    
    code_execution_success_rate: float = 0.0
    
    total_fixtures: int = 0
    successful_runs: int = 0
    
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'concept_extraction': {
                'precision': self.concept_extraction_precision,
                'recall': self.concept_extraction_recall,
                'f1': self.concept_extraction_f1
            },
            'mapping': {
                'precision': self.mapping_precision,
                'recall': self.mapping_recall,
                'f1': self.mapping_f1
            },
            'code_execution_success_rate': self.code_execution_success_rate,
            'total_fixtures': self.total_fixtures,
            'successful_runs': self.successful_runs,
            'overall_success_rate': self.successful_runs / max(self.total_fixtures, 1)
        }


class AccuracyBenchmark:
    """
    Benchmark system for measuring accuracy.
    
    Runs the pipeline against known fixtures and measures:
    - Concept extraction accuracy
    - Mapping accuracy
    - Code execution success rate
    """
    
    def __init__(self, orchestrator=None):
        """
        Initialize the benchmark.
        
        Args:
            orchestrator: PipelineOrchestrator instance
        """
        self.orchestrator = orchestrator
        self.fixtures = BENCHMARK_FIXTURES
        self.results: List[Dict[str, Any]] = []
    
    async def run_benchmark(
        self,
        fixtures: Optional[List[BenchmarkFixture]] = None,
        timeout_per_fixture: int = 300
    ) -> AccuracyMetrics:
        """
        Run the benchmark against all fixtures.
        
        Args:
            fixtures: List of fixtures to test (default: all)
            timeout_per_fixture: Timeout in seconds per fixture
            
        Returns:
            AccuracyMetrics with results
        """
        fixtures = fixtures or self.fixtures
        metrics = AccuracyMetrics(total_fixtures=len(fixtures))
        
        all_concept_results = []
        all_mapping_results = []
        all_execution_results = []
        
        for fixture in fixtures:
            logger.info(f"Running benchmark: {fixture.name}")
            
            try:
                result = await asyncio.wait_for(
                    self._run_single_fixture(fixture),
                    timeout=timeout_per_fixture
                )
                
                self.results.append(result)
                
                if result['success']:
                    metrics.successful_runs += 1
                
                all_concept_results.append(result.get('concept_metrics', {}))
                all_mapping_results.append(result.get('mapping_metrics', {}))
                all_execution_results.append(result.get('execution_success', False))
                
            except asyncio.TimeoutError:
                logger.error(f"Fixture {fixture.name} timed out")
                self.results.append({
                    'fixture': fixture.name,
                    'success': False,
                    'error': 'Timeout'
                })
            except Exception as e:
                logger.error(f"Fixture {fixture.name} failed: {e}")
                self.results.append({
                    'fixture': fixture.name,
                    'success': False,
                    'error': str(e)
                })
        
        # Aggregate metrics
        metrics = self._aggregate_metrics(
            all_concept_results,
            all_mapping_results,
            all_execution_results,
            metrics
        )
        
        return metrics
    
    async def _run_single_fixture(
        self,
        fixture: BenchmarkFixture
    ) -> Dict[str, Any]:
        """Run benchmark for a single fixture."""
        result = {
            'fixture': fixture.name,
            'success': False,
            'concept_metrics': {},
            'mapping_metrics': {},
            'execution_success': False
        }
        
        if not self.orchestrator:
            result['error'] = 'No orchestrator available'
            return result
        
        try:
            # Run the pipeline
            pipeline_result = await self.orchestrator.run_pipeline(
                paper_url=fixture.paper_url,
                repo_url=fixture.repo_url,
                auto_fix_errors=True
            )
            
            result['success'] = pipeline_result.success
            
            # Evaluate concept extraction
            extracted_concepts = self._extract_concept_names(pipeline_result)
            result['concept_metrics'] = self._evaluate_concepts(
                extracted_concepts,
                fixture.expected_concepts
            )
            
            # Evaluate mappings
            result['mapping_metrics'] = self._evaluate_mappings(
                pipeline_result.concept_mappings,
                fixture.expected_mappings
            )
            
            # Check code execution
            result['execution_success'] = any(
                exec_result.get('success', False)
                for exec_result in pipeline_result.execution_results
            )
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def _extract_concept_names(self, pipeline_result) -> Set[str]:
        """Extract concept names from pipeline result."""
        concepts = set()
        
        # From paper info
        paper_info = pipeline_result.paper_info or {}
        analysis = paper_info.get('analysis', {})
        
        for concept in analysis.get('key_concepts', []):
            if isinstance(concept, dict):
                concepts.add(concept.get('name', '').lower())
            else:
                concepts.add(str(concept).lower())
        
        for algo in analysis.get('algorithms', []):
            if isinstance(algo, dict):
                concepts.add(algo.get('name', '').lower())
            else:
                concepts.add(str(algo).lower())
        
        # From keywords
        for keyword in paper_info.get('keywords', []):
            concepts.add(keyword.lower())
        
        return {c for c in concepts if c}
    
    def _evaluate_concepts(
        self,
        extracted: Set[str],
        expected: List[str]
    ) -> Dict[str, float]:
        """Evaluate concept extraction accuracy."""
        expected_lower = {e.lower() for e in expected}
        
        # Fuzzy matching
        true_positives = 0
        for expected_concept in expected_lower:
            for extracted_concept in extracted:
                if expected_concept in extracted_concept or extracted_concept in expected_concept:
                    true_positives += 1
                    break
        
        precision = true_positives / len(extracted) if extracted else 0
        recall = true_positives / len(expected_lower) if expected_lower else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'extracted_count': len(extracted),
            'expected_count': len(expected_lower),
            'matches': true_positives
        }
    
    def _evaluate_mappings(
        self,
        actual_mappings: List[Dict],
        expected_mappings: List[Dict]
    ) -> Dict[str, float]:
        """Evaluate mapping accuracy."""
        import re
        
        matches = 0
        
        for expected in expected_mappings:
            concept = expected.get('concept', '').lower()
            pattern = expected.get('code_pattern', '')
            
            for actual in actual_mappings:
                actual_concept = actual.get('concept_name', '').lower()
                actual_code = actual.get('code_element', '').lower()
                
                # Check if concept matches and code pattern matches
                if concept in actual_concept or actual_concept in concept:
                    if re.search(pattern, actual_code, re.IGNORECASE):
                        matches += 1
                        break
        
        precision = matches / len(actual_mappings) if actual_mappings else 0
        recall = matches / len(expected_mappings) if expected_mappings else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'matches': matches
        }
    
    def _aggregate_metrics(
        self,
        concept_results: List[Dict],
        mapping_results: List[Dict],
        execution_results: List[bool],
        metrics: AccuracyMetrics
    ) -> AccuracyMetrics:
        """Aggregate metrics across all fixtures."""
        # Average concept metrics
        if concept_results:
            valid_results = [r for r in concept_results if r]
            if valid_results:
                metrics.concept_extraction_precision = sum(
                    r.get('precision', 0) for r in valid_results
                ) / len(valid_results)
                metrics.concept_extraction_recall = sum(
                    r.get('recall', 0) for r in valid_results
                ) / len(valid_results)
                metrics.concept_extraction_f1 = sum(
                    r.get('f1', 0) for r in valid_results
                ) / len(valid_results)
        
        # Average mapping metrics
        if mapping_results:
            valid_results = [r for r in mapping_results if r]
            if valid_results:
                metrics.mapping_precision = sum(
                    r.get('precision', 0) for r in valid_results
                ) / len(valid_results)
                metrics.mapping_recall = sum(
                    r.get('recall', 0) for r in valid_results
                ) / len(valid_results)
                metrics.mapping_f1 = sum(
                    r.get('f1', 0) for r in valid_results
                ) / len(valid_results)
        
        # Execution success rate
        if execution_results:
            metrics.code_execution_success_rate = sum(execution_results) / len(execution_results)
        
        return metrics
    
    def generate_report(self) -> str:
        """Generate a human-readable benchmark report."""
        lines = [
            "=" * 60,
            "SCIENTIFIC AGENT SYSTEM - BENCHMARK REPORT",
            "=" * 60,
            f"Generated: {datetime.now().isoformat()}",
            "",
        ]
        
        # Per-fixture results
        lines.append("FIXTURE RESULTS:")
        lines.append("-" * 40)
        
        for result in self.results:
            status = "✓" if result.get('success') else "✗"
            lines.append(f"{status} {result['fixture']}")
            
            if result.get('error'):
                lines.append(f"  Error: {result['error']}")
            
            if result.get('concept_metrics'):
                cm = result['concept_metrics']
                lines.append(f"  Concepts: P={cm.get('precision', 0):.2f} R={cm.get('recall', 0):.2f} F1={cm.get('f1', 0):.2f}")
            
            if result.get('mapping_metrics'):
                mm = result['mapping_metrics']
                lines.append(f"  Mappings: P={mm.get('precision', 0):.2f} R={mm.get('recall', 0):.2f} F1={mm.get('f1', 0):.2f}")
        
        return "\n".join(lines)


# =============================================================================
# UNIT TESTS
# =============================================================================

class TestEnhancedPDFParser(unittest.TestCase):
    """Unit tests for the enhanced PDF parser."""
    
    def setUp(self):
        # Import here to avoid circular imports
        try:
            from core.enhanced_pdf_parser import EnhancedPDFParser, PyMuPDFExtractor
            self.parser_available = True
            self.parser = EnhancedPDFParser()
        except ImportError:
            self.parser_available = False
    
    def test_parser_initialization(self):
        """Test that parser initializes correctly."""
        if not self.parser_available:
            self.skipTest("Parser not available")
        
        self.assertIsNotNone(self.parser)
        self.assertTrue(len(self.parser.backends) > 0)
    
    def test_quality_score_calculation(self):
        """Test quality score calculation."""
        if not self.parser_available:
            self.skipTest("Parser not available")
        
        from core.enhanced_pdf_parser import PyMuPDFExtractor
        
        extractor = PyMuPDFExtractor()
        
        # Good quality text
        good_text = """
        Abstract
        
        This paper presents a novel approach to machine learning.
        We propose a new algorithm that significantly improves accuracy.
        Our method achieves state-of-the-art results on multiple benchmarks.
        
        Introduction
        
        Machine learning has become increasingly important in recent years.
        """
        good_score = extractor.calculate_quality_score(good_text)
        self.assertGreater(good_score, 0.3)
        
        # Poor quality text (garbage)
        bad_text = "asdf jkl; zxcv"
        bad_score = extractor.calculate_quality_score(bad_text)
        self.assertLess(bad_score, good_score)
        
        # Empty text
        empty_score = extractor.calculate_quality_score("")
        self.assertEqual(empty_score, 0.0)


class TestSemanticMapper(unittest.TestCase):
    """Unit tests for the semantic mapper."""
    
    def setUp(self):
        try:
            from core.semantic_mapper import (
                EnhancedSemanticMapper, 
                LexicalMatcher,
                CodeElement
            )
            self.mapper_available = True
            self.mapper = EnhancedSemanticMapper()
            self.lexical = LexicalMatcher()
            self.CodeElement = CodeElement
        except ImportError:
            self.mapper_available = False
    
    def test_lexical_similarity(self):
        """Test lexical similarity computation."""
        if not self.mapper_available:
            self.skipTest("Mapper not available")
        
        # Exact match
        score, _ = self.lexical.compute_name_similarity(
            "MultiHeadAttention",
            "multi_head_attention"
        )
        self.assertGreater(score, 0.6)
        
        # Substring match
        score, _ = self.lexical.compute_name_similarity(
            "Attention",
            "SelfAttention"
        )
        self.assertGreater(score, 0.5)
        
        # No match
        score, _ = self.lexical.compute_name_similarity(
            "Transformer",
            "ConvolutionalNetwork"
        )
        self.assertLess(score, 0.3)
    
    def test_term_extraction(self):
        """Test term extraction from text."""
        if not self.mapper_available:
            self.skipTest("Mapper not available")
        
        terms = self.lexical.extract_terms("Multi-Head Attention Mechanism")
        
        self.assertIn("multi", terms)
        self.assertIn("head", terms)
        self.assertIn("attention", terms)
        self.assertIn("mechanism", terms)
        
        # Stop terms should be filtered
        self.assertNotIn("model", self.lexical.extract_terms("model data train"))


class TestCodeValidator(unittest.TestCase):
    """Unit tests for the code validator."""
    
    def setUp(self):
        try:
            from core.code_validator import CodeValidator, PythonValidator
            self.validator_available = True
            self.validator = CodeValidator()
            self.py_validator = PythonValidator()
        except ImportError:
            self.validator_available = False
    
    def test_valid_python(self):
        """Test validation of valid Python code."""
        if not self.validator_available:
            self.skipTest("Validator not available")
        
        valid_code = """
def hello(name: str) -> str:
    return f"Hello, {name}!"

if __name__ == "__main__":
    print(hello("World"))
"""
        result = self.validator.validate(valid_code, "python")
        self.assertTrue(result.is_valid)
    
    def test_syntax_error_detection(self):
        """Test detection of syntax errors."""
        if not self.validator_available:
            self.skipTest("Validator not available")
        
        invalid_code = """
def hello(name)
    return f"Hello, {name}!"
"""
        result = self.validator.validate(invalid_code, "python")
        self.assertFalse(result.is_valid)
        self.assertTrue(any(e.category == 'syntax' for e in result.errors))
    
    def test_unbalanced_brackets(self):
        """Test detection of unbalanced brackets."""
        if not self.validator_available:
            self.skipTest("Validator not available")
        
        invalid_code = """
def process(data):
    result = [item for item in data
    return result
"""
        result = self.validator.validate(invalid_code, "python")
        self.assertFalse(result.is_valid)
    
    def test_security_warning(self):
        """Test detection of security issues."""
        if not self.validator_available:
            self.skipTest("Validator not available")
        
        risky_code = """
user_input = input()
eval(user_input)
"""
        result = self.validator.validate(risky_code, "python")
        self.assertTrue(any(
            e.category == 'security' 
            for e in result.warnings
        ))


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        self.has_dependencies = True
        try:
            import google.generativeai
            import networkx
        except ImportError:
            self.has_dependencies = False
    
    def test_components_import(self):
        """Test that all components can be imported."""
        try:
            from core.enhanced_pdf_parser import EnhancedPDFParser
            from core.semantic_mapper import EnhancedSemanticMapper
            from core.code_validator import CodeValidator
        except ImportError as e:
            self.fail(f"Failed to import components: {e}")
    
    def test_benchmark_fixtures_valid(self):
        """Test that benchmark fixtures are properly defined."""
        for fixture in BENCHMARK_FIXTURES:
            self.assertTrue(fixture.name, "Fixture must have a name")
            self.assertTrue(fixture.paper_url, "Fixture must have a paper URL")
            self.assertTrue(fixture.repo_url, "Fixture must have a repo URL")
            self.assertTrue(len(fixture.expected_concepts) > 0, "Fixture must have expected concepts")


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_unit_tests() -> Tuple[int, int]:
    """
    Run all unit tests.
    
    Returns:
        Tuple of (passed, failed) counts
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedPDFParser))
    suite.addTests(loader.loadTestsFromTestCase(TestSemanticMapper))
    suite.addTests(loader.loadTestsFromTestCase(TestCodeValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    passed = result.testsRun - len(result.failures) - len(result.errors)
    failed = len(result.failures) + len(result.errors)
    
    return passed, failed


async def run_accuracy_benchmark(orchestrator=None) -> AccuracyMetrics:
    """
    Run the accuracy benchmark.
    
    Args:
        orchestrator: PipelineOrchestrator instance
        
    Returns:
        AccuracyMetrics with results
    """
    benchmark = AccuracyBenchmark(orchestrator)
    metrics = await benchmark.run_benchmark()
    
    print("\n" + benchmark.generate_report())
    
    return metrics


if __name__ == "__main__":
    print("Running unit tests...")
    passed, failed = run_unit_tests()
    print(f"\nResults: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)
