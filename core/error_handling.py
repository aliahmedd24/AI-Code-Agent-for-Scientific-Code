"""
Structured Error Handling - Retry Logic and Error Recovery

MEDIUM-PRIORITY ENHANCEMENT:
- Proper error classification and handling
- Configurable retry strategies
- Error aggregation and reporting
- Graceful degradation without silent failures
"""

import asyncio
import logging
import traceback
from typing import Optional, Any, Dict, List, Callable, TypeVar, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
import random

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# ERROR CLASSIFICATION
# =============================================================================

class ErrorSeverity(str, Enum):
    """Severity levels for errors."""
    CRITICAL = "critical"    # System cannot continue
    ERROR = "error"          # Operation failed, may retry
    WARNING = "warning"      # Operation completed with issues
    INFO = "info"           # Informational, not a problem


class ErrorCategory(str, Enum):
    """Categories of errors."""
    NETWORK = "network"           # Network-related errors
    API = "api"                   # API errors (rate limits, auth, etc.)
    PARSING = "parsing"           # Data parsing errors
    VALIDATION = "validation"     # Validation failures
    RESOURCE = "resource"         # Resource not found
    TIMEOUT = "timeout"           # Timeout errors
    INTERNAL = "internal"         # Internal/unexpected errors
    DEPENDENCY = "dependency"     # Missing dependencies


@dataclass
class StructuredError:
    """A structured error with full context."""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    original_exception: Optional[Exception] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    stage: str = ""
    agent: str = ""
    recoverable: bool = True
    suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'category': self.category.value,
            'severity': self.severity.value,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp,
            'stage': self.stage,
            'agent': self.agent,
            'recoverable': self.recoverable,
            'suggestions': self.suggestions,
            'traceback': traceback.format_exception(
                type(self.original_exception),
                self.original_exception,
                self.original_exception.__traceback__
            ) if self.original_exception else None
        }
    
    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.category.value}: {self.message}"


# =============================================================================
# ERROR COLLECTOR
# =============================================================================

class ErrorCollector:
    """Collects and aggregates errors from pipeline execution."""
    
    def __init__(self):
        self._errors: List[StructuredError] = []
        self._warnings: List[StructuredError] = []
    
    def add_error(self, error: StructuredError):
        """Add an error to the collection."""
        if error.severity == ErrorSeverity.WARNING:
            self._warnings.append(error)
        else:
            self._errors.append(error)
        
        # Log immediately
        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical(str(error))
        elif error.severity == ErrorSeverity.ERROR:
            logger.error(str(error))
        elif error.severity == ErrorSeverity.WARNING:
            logger.warning(str(error))
        else:
            logger.info(str(error))
    
    def has_critical_errors(self) -> bool:
        """Check if there are any critical errors."""
        return any(e.severity == ErrorSeverity.CRITICAL for e in self._errors)
    
    def has_errors(self) -> bool:
        """Check if there are any errors (not warnings)."""
        return len(self._errors) > 0
    
    def get_errors(self, category: Optional[ErrorCategory] = None) -> List[StructuredError]:
        """Get all errors, optionally filtered by category."""
        if category:
            return [e for e in self._errors if e.category == category]
        return self._errors.copy()
    
    def get_warnings(self) -> List[StructuredError]:
        """Get all warnings."""
        return self._warnings.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all errors."""
        return {
            'total_errors': len(self._errors),
            'total_warnings': len(self._warnings),
            'critical_count': sum(1 for e in self._errors if e.severity == ErrorSeverity.CRITICAL),
            'by_category': {
                cat.value: sum(1 for e in self._errors if e.category == cat)
                for cat in ErrorCategory
            },
            'by_stage': self._group_by_stage(),
            'recoverable_count': sum(1 for e in self._errors if e.recoverable)
        }
    
    def _group_by_stage(self) -> Dict[str, int]:
        """Group errors by pipeline stage."""
        stages = {}
        for error in self._errors:
            if error.stage:
                stages[error.stage] = stages.get(error.stage, 0) + 1
        return stages
    
    def clear(self):
        """Clear all collected errors."""
        self._errors.clear()
        self._warnings.clear()
    
    def to_report(self) -> str:
        """Generate a text report of errors."""
        lines = ["=" * 60, "ERROR REPORT", "=" * 60, ""]
        
        summary = self.get_summary()
        lines.append(f"Total Errors: {summary['total_errors']}")
        lines.append(f"Total Warnings: {summary['total_warnings']}")
        lines.append(f"Critical Errors: {summary['critical_count']}")
        lines.append("")
        
        if self._errors:
            lines.append("-" * 40)
            lines.append("ERRORS:")
            lines.append("-" * 40)
            for error in self._errors:
                lines.append(f"\n[{error.timestamp}] {error}")
                if error.stage:
                    lines.append(f"  Stage: {error.stage}")
                if error.agent:
                    lines.append(f"  Agent: {error.agent}")
                if error.suggestions:
                    lines.append(f"  Suggestions: {', '.join(error.suggestions)}")
        
        if self._warnings:
            lines.append("\n" + "-" * 40)
            lines.append("WARNINGS:")
            lines.append("-" * 40)
            for warning in self._warnings:
                lines.append(f"\n{warning}")
        
        return "\n".join(lines)


# =============================================================================
# RETRY STRATEGIES
# =============================================================================

class RetryStrategy:
    """Base class for retry strategies."""
    
    def get_delay(self, attempt: int) -> float:
        """Get delay before next retry attempt."""
        raise NotImplementedError
    
    def should_retry(self, attempt: int, exception: Exception) -> bool:
        """Determine if we should retry after an exception."""
        raise NotImplementedError


class ExponentialBackoff(RetryStrategy):
    """Exponential backoff retry strategy."""
    
    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        max_attempts: int = 5,
        jitter: bool = True,
        retryable_exceptions: Optional[tuple] = None
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_attempts = max_attempts
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or (Exception,)
    
    def get_delay(self, attempt: int) -> float:
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        if self.jitter:
            delay = delay * (0.5 + random.random())
        return delay
    
    def should_retry(self, attempt: int, exception: Exception) -> bool:
        if attempt >= self.max_attempts:
            return False
        return isinstance(exception, self.retryable_exceptions)


class LinearBackoff(RetryStrategy):
    """Linear backoff retry strategy."""
    
    def __init__(
        self,
        delay: float = 2.0,
        max_attempts: int = 3,
        retryable_exceptions: Optional[tuple] = None
    ):
        self.delay = delay
        self.max_attempts = max_attempts
        self.retryable_exceptions = retryable_exceptions or (Exception,)
    
    def get_delay(self, attempt: int) -> float:
        return self.delay * (attempt + 1)
    
    def should_retry(self, attempt: int, exception: Exception) -> bool:
        if attempt >= self.max_attempts:
            return False
        return isinstance(exception, self.retryable_exceptions)


class NoRetry(RetryStrategy):
    """No retry strategy."""
    
    def get_delay(self, attempt: int) -> float:
        return 0
    
    def should_retry(self, attempt: int, exception: Exception) -> bool:
        return False


# =============================================================================
# RETRY DECORATOR
# =============================================================================

def with_retry(
    strategy: Optional[RetryStrategy] = None,
    error_collector: Optional[ErrorCollector] = None,
    stage: str = "",
    agent: str = "",
    fallback: Optional[Callable[[], T]] = None
):
    """
    Decorator that adds retry logic to async functions.
    
    MEDIUM-PRIORITY ENHANCEMENT: Replaces silent failures with proper retry logic.
    """
    if strategy is None:
        strategy = ExponentialBackoff()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            attempt = 0
            
            while True:
                try:
                    return await func(*args, **kwargs)
                    
                except Exception as e:
                    last_exception = e
                    
                    # Classify error
                    error = classify_exception(e, stage=stage, agent=agent)
                    
                    # Should we retry?
                    if strategy.should_retry(attempt, e):
                        delay = strategy.get_delay(attempt)
                        
                        logger.warning(
                            f"Retry {attempt + 1}/{strategy.max_attempts} for {func.__name__}: "
                            f"{e}. Waiting {delay:.1f}s..."
                        )
                        
                        # Add as warning
                        if error_collector:
                            error.severity = ErrorSeverity.WARNING
                            error.message = f"Retrying: {error.message}"
                            error_collector.add_error(error)
                        
                        await asyncio.sleep(delay)
                        attempt += 1
                        continue
                    
                    # No more retries - record error
                    if error_collector:
                        error_collector.add_error(error)
                    
                    # Try fallback
                    if fallback is not None:
                        logger.warning(f"Using fallback for {func.__name__}")
                        return fallback()
                    
                    # Re-raise
                    raise
            
            # Should never reach here
            raise last_exception
        
        return wrapper
    return decorator


def classify_exception(
    exception: Exception,
    stage: str = "",
    agent: str = ""
) -> StructuredError:
    """Classify an exception into a structured error."""
    
    exc_name = type(exception).__name__
    exc_msg = str(exception)
    
    # Network errors
    if any(name in exc_name for name in ['Connection', 'Timeout', 'Network', 'Socket', 'HTTP']):
        return StructuredError(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.ERROR,
            message=f"Network error: {exc_msg}",
            original_exception=exception,
            stage=stage,
            agent=agent,
            recoverable=True,
            suggestions=["Check network connection", "Retry later"]
        )
    
    # API errors
    if any(name in exc_name for name in ['API', 'RateLimit', 'Quota', 'Auth']):
        return StructuredError(
            category=ErrorCategory.API,
            severity=ErrorSeverity.ERROR,
            message=f"API error: {exc_msg}",
            original_exception=exception,
            stage=stage,
            agent=agent,
            recoverable='rate' in exc_msg.lower() or 'quota' in exc_msg.lower(),
            suggestions=["Check API key", "Wait for rate limit reset"]
        )
    
    # Parsing errors
    if any(name in exc_name for name in ['Parse', 'JSON', 'Decode', 'Syntax', 'XML']):
        return StructuredError(
            category=ErrorCategory.PARSING,
            severity=ErrorSeverity.ERROR,
            message=f"Parsing error: {exc_msg}",
            original_exception=exception,
            stage=stage,
            agent=agent,
            recoverable=False,
            suggestions=["Check input format", "Validate data structure"]
        )
    
    # Resource errors
    if any(name in exc_name for name in ['NotFound', 'FileNotFound', 'NoSuch', 'Missing']):
        return StructuredError(
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.ERROR,
            message=f"Resource not found: {exc_msg}",
            original_exception=exception,
            stage=stage,
            agent=agent,
            recoverable=False,
            suggestions=["Verify resource exists", "Check URL/path"]
        )
    
    # Validation errors
    if any(name in exc_name for name in ['Validation', 'Invalid', 'Value', 'Type']):
        return StructuredError(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.ERROR,
            message=f"Validation error: {exc_msg}",
            original_exception=exception,
            stage=stage,
            agent=agent,
            recoverable=False,
            suggestions=["Check input data", "Review validation rules"]
        )
    
    # Timeout errors
    if 'timeout' in exc_name.lower() or 'timeout' in exc_msg.lower():
        return StructuredError(
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.ERROR,
            message=f"Timeout: {exc_msg}",
            original_exception=exception,
            stage=stage,
            agent=agent,
            recoverable=True,
            suggestions=["Increase timeout", "Retry with smaller input"]
        )
    
    # Import/dependency errors
    if any(name in exc_name for name in ['Import', 'Module', 'Package']):
        return StructuredError(
            category=ErrorCategory.DEPENDENCY,
            severity=ErrorSeverity.ERROR,
            message=f"Dependency error: {exc_msg}",
            original_exception=exception,
            stage=stage,
            agent=agent,
            recoverable=False,
            suggestions=["Install missing package", "Check requirements.txt"]
        )
    
    # Default: internal error
    return StructuredError(
        category=ErrorCategory.INTERNAL,
        severity=ErrorSeverity.ERROR,
        message=f"Internal error: {exc_msg}",
        original_exception=exception,
        stage=stage,
        agent=agent,
        recoverable=False,
        suggestions=["Check logs for details", "Report bug if persistent"]
    )


# =============================================================================
# RESULT WRAPPER
# =============================================================================

@dataclass
class Result(Generic[T]):
    """
    Result wrapper that can contain either a value or an error.
    
    Usage:
        result = await safe_operation()
        if result.is_ok():
            value = result.unwrap()
        else:
            error = result.error()
    """
    _value: Optional[T] = None
    _error: Optional[StructuredError] = None
    
    @classmethod
    def ok(cls, value: T) -> 'Result[T]':
        """Create a successful result."""
        return cls(_value=value)
    
    @classmethod
    def err(cls, error: StructuredError) -> 'Result[T]':
        """Create an error result."""
        return cls(_error=error)
    
    def is_ok(self) -> bool:
        """Check if result is successful."""
        return self._error is None
    
    def is_err(self) -> bool:
        """Check if result is an error."""
        return self._error is not None
    
    def unwrap(self) -> T:
        """Get the value, raising if error."""
        if self._error:
            raise ValueError(f"Cannot unwrap error result: {self._error}")
        return self._value
    
    def unwrap_or(self, default: T) -> T:
        """Get the value or a default if error."""
        if self._error:
            return default
        return self._value
    
    def error(self) -> Optional[StructuredError]:
        """Get the error if present."""
        return self._error
    
    def map(self, func: Callable[[T], U]) -> 'Result[U]':
        """Map the value if successful."""
        if self._error:
            return Result.err(self._error)
        return Result.ok(func(self._value))


from typing import Generic, TypeVar
U = TypeVar('U')


# =============================================================================
# SAFE EXECUTION HELPERS
# =============================================================================

async def safe_execute(
    func: Callable[..., T],
    *args,
    stage: str = "",
    agent: str = "",
    error_collector: Optional[ErrorCollector] = None,
    **kwargs
) -> Result[T]:
    """
    Execute a function safely, capturing any errors.
    
    MEDIUM-PRIORITY: Provides explicit error handling instead of silent failures.
    """
    try:
        result = await func(*args, **kwargs)
        return Result.ok(result)
    except Exception as e:
        error = classify_exception(e, stage=stage, agent=agent)
        if error_collector:
            error_collector.add_error(error)
        return Result.err(error)


def safe_execute_sync(
    func: Callable[..., T],
    *args,
    stage: str = "",
    agent: str = "",
    error_collector: Optional[ErrorCollector] = None,
    **kwargs
) -> Result[T]:
    """Synchronous version of safe_execute."""
    try:
        result = func(*args, **kwargs)
        return Result.ok(result)
    except Exception as e:
        error = classify_exception(e, stage=stage, agent=agent)
        if error_collector:
            error_collector.add_error(error)
        return Result.err(error)


# =============================================================================
# PIPELINE ERROR HANDLER
# =============================================================================

class PipelineErrorHandler:
    """
    Centralized error handling for the pipeline.
    
    MEDIUM-PRIORITY ENHANCEMENT: Replaces scattered try/except with centralized handling.
    """
    
    def __init__(self):
        self.collector = ErrorCollector()
        self.retry_strategies: Dict[str, RetryStrategy] = {
            'default': ExponentialBackoff(max_attempts=3),
            'api': ExponentialBackoff(base_delay=2.0, max_attempts=5),
            'network': ExponentialBackoff(base_delay=1.0, max_attempts=3),
            'parse': NoRetry(),
        }
    
    def get_strategy(self, category: ErrorCategory) -> RetryStrategy:
        """Get retry strategy for error category."""
        mapping = {
            ErrorCategory.API: 'api',
            ErrorCategory.NETWORK: 'network',
            ErrorCategory.PARSING: 'parse',
            ErrorCategory.TIMEOUT: 'network',
        }
        return self.retry_strategies.get(mapping.get(category, 'default'), self.retry_strategies['default'])
    
    def handle_error(self, error: StructuredError) -> bool:
        """
        Handle an error and determine if pipeline should continue.
        
        Returns True if pipeline can continue, False if it should stop.
        """
        self.collector.add_error(error)
        
        if error.severity == ErrorSeverity.CRITICAL:
            return False
        
        if not error.recoverable:
            return error.severity != ErrorSeverity.ERROR
        
        return True
    
    def get_report(self) -> str:
        """Get error report."""
        return self.collector.to_report()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get error summary."""
        return self.collector.get_summary()
    
    def should_abort(self) -> bool:
        """Check if pipeline should abort."""
        return self.collector.has_critical_errors()
    
    def clear(self):
        """Clear all errors."""
        self.collector.clear()


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_global_error_handler: Optional[PipelineErrorHandler] = None


def get_error_handler() -> PipelineErrorHandler:
    """Get or create global error handler."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = PipelineErrorHandler()
    return _global_error_handler


def reset_error_handler():
    """Reset global error handler."""
    global _global_error_handler
    _global_error_handler = None