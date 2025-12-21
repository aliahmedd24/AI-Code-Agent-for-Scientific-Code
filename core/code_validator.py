"""
Code Validator - Pre-Execution Validation and Self-Correction

This module provides:
- Syntax validation for multiple languages
- Import/dependency checking
- Security scanning for dangerous patterns
- LLM-based self-correction for fixing errors
- Iterative improvement until code is valid

Author: Scientific Agent System
"""

import os
import re
import ast
import sys
import asyncio
import tempfile
import subprocess
from typing import Optional, Any, Dict, List, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Severity level of validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """A validation issue found in code."""
    level: ValidationLevel
    category: str  # syntax, import, security, style, logic
    message: str
    line: Optional[int] = None
    column: Optional[int] = None
    suggestion: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'level': self.level.value,
            'category': self.category,
            'message': self.message,
            'line': self.line,
            'column': self.column,
            'suggestion': self.suggestion
        }


@dataclass
class ValidationResult:
    """Result of code validation."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    fixed_code: Optional[str] = None
    fix_attempts: int = 0
    
    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == ValidationLevel.ERROR]
    
    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == ValidationLevel.WARNING]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_valid': self.is_valid,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'issues': [i.to_dict() for i in self.issues],
            'fix_attempts': self.fix_attempts
        }


class PythonValidator:
    """Validates Python code."""
    
    # Dangerous patterns that should be flagged
    DANGEROUS_PATTERNS = [
        (r'\beval\s*\(', 'eval() is dangerous - avoid using it'),
        (r'\bexec\s*\(', 'exec() is dangerous - avoid using it'),
        (r'\b__import__\s*\(', '__import__() can be dangerous'),
        (r'\bos\.system\s*\(', 'os.system() can be dangerous - use subprocess instead'),
        (r'\bsubprocess\.call\s*\([^)]*shell\s*=\s*True', 'shell=True is dangerous'),
        (r'\bpickle\.loads?\s*\(', 'pickle can execute arbitrary code'),
        (r'\brm\s+-rf\s+/', 'Dangerous file deletion pattern'),
    ]
    
    # Common imports that should be available
    STANDARD_IMPORTS = {
        'os', 'sys', 're', 'json', 'math', 'random', 'time', 'datetime',
        'collections', 'itertools', 'functools', 'typing', 'pathlib',
        'dataclasses', 'enum', 'abc', 'copy', 'io', 'tempfile', 'shutil',
        'logging', 'warnings', 'traceback', 'hashlib', 'base64'
    }
    
    # Common third-party packages
    COMMON_PACKAGES = {
        'numpy': 'np',
        'pandas': 'pd',
        'matplotlib': 'plt',
        'torch': 'torch',
        'tensorflow': 'tf',
        'sklearn': 'sklearn',
        'scipy': 'scipy',
        'requests': 'requests',
        'httpx': 'httpx',
        'aiohttp': 'aiohttp',
        'pydantic': 'pydantic',
        'fastapi': 'fastapi',
    }
    
    def __init__(self, available_packages: Optional[Set[str]] = None):
        """
        Initialize the Python validator.
        
        Args:
            available_packages: Set of packages known to be installed
        """
        self.available_packages = available_packages or set()
    
    def validate(self, code: str) -> ValidationResult:
        """
        Validate Python code.
        
        Args:
            code: Python source code
            
        Returns:
            ValidationResult with any issues found
        """
        issues: List[ValidationIssue] = []
        
        # Check syntax
        syntax_issues = self._check_syntax(code)
        issues.extend(syntax_issues)
        
        # If syntax is invalid, don't continue with other checks
        if any(i.level == ValidationLevel.ERROR for i in syntax_issues):
            return ValidationResult(is_valid=False, issues=issues)
        
        # Check imports
        import_issues = self._check_imports(code)
        issues.extend(import_issues)
        
        # Check for dangerous patterns
        security_issues = self._check_security(code)
        issues.extend(security_issues)
        
        # Check for common issues
        style_issues = self._check_style(code)
        issues.extend(style_issues)
        
        # Determine if valid (no errors)
        is_valid = not any(i.level == ValidationLevel.ERROR for i in issues)
        
        return ValidationResult(is_valid=is_valid, issues=issues)
    
    def _check_syntax(self, code: str) -> List[ValidationIssue]:
        """Check Python syntax."""
        issues = []
        
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                category='syntax',
                message=f"Syntax error: {e.msg}",
                line=e.lineno,
                column=e.offset,
                suggestion=self._get_syntax_suggestion(e)
            ))
        except Exception as e:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                category='syntax',
                message=f"Parse error: {str(e)}"
            ))
        
        return issues
    
    def _get_syntax_suggestion(self, error: SyntaxError) -> str:
        """Get suggestion for fixing syntax error."""
        msg = str(error.msg).lower()
        
        if 'unexpected eof' in msg:
            return "Check for missing closing brackets, parentheses, or quotes"
        elif 'invalid syntax' in msg:
            return "Check for typos, missing colons, or incorrect indentation"
        elif 'expected' in msg:
            return f"Add the expected token: {error.msg}"
        elif 'indent' in msg:
            return "Fix the indentation - use consistent spaces or tabs"
        
        return "Review the syntax at the indicated line"
    
    def _check_imports(self, code: str) -> List[ValidationIssue]:
        """Check import statements."""
        issues = []
        
        try:
            tree = ast.parse(code)
        except:
            return issues
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split('.')[0]
                    issue = self._check_module_available(module, node.lineno)
                    if issue:
                        issues.append(issue)
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split('.')[0]
                    issue = self._check_module_available(module, node.lineno)
                    if issue:
                        issues.append(issue)
        
        return issues
    
    def _check_module_available(self, module: str, line: int) -> Optional[ValidationIssue]:
        """Check if a module is likely available."""
        # Standard library is always available
        if module in self.STANDARD_IMPORTS:
            return None
        
        # Known available packages
        if module in self.available_packages:
            return None
        
        # Common packages (warn but don't error)
        if module in self.COMMON_PACKAGES:
            return ValidationIssue(
                level=ValidationLevel.WARNING,
                category='import',
                message=f"Package '{module}' may need to be installed",
                line=line,
                suggestion=f"Run: pip install {module}"
            )
        
        # Unknown package
        return ValidationIssue(
            level=ValidationLevel.WARNING,
            category='import',
            message=f"Unknown import '{module}' - ensure it's installed",
            line=line,
            suggestion=f"Verify '{module}' is available in the execution environment"
        )
    
    def _check_security(self, code: str) -> List[ValidationIssue]:
        """Check for dangerous patterns."""
        issues = []
        
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            for pattern, message in self.DANGEROUS_PATTERNS:
                if re.search(pattern, line):
                    issues.append(ValidationIssue(
                        level=ValidationLevel.WARNING,
                        category='security',
                        message=message,
                        line=i,
                        suggestion="Consider using a safer alternative"
                    ))
        
        return issues
    
    def _check_style(self, code: str) -> List[ValidationIssue]:
        """Check for common style issues."""
        issues = []
        
        lines = code.split('\n')
        
        # Check for very long lines
        for i, line in enumerate(lines, 1):
            if len(line) > 120:
                issues.append(ValidationIssue(
                    level=ValidationLevel.INFO,
                    category='style',
                    message=f"Line exceeds 120 characters ({len(line)} chars)",
                    line=i,
                    suggestion="Consider breaking into multiple lines"
                ))
        
        # Check for bare except
        if re.search(r'\bexcept\s*:', code):
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category='style',
                message="Bare 'except:' catches all exceptions including KeyboardInterrupt",
                suggestion="Use 'except Exception:' or catch specific exceptions"
            ))
        
        # Check for mutable default arguments
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    for default in node.args.defaults:
                        if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                            issues.append(ValidationIssue(
                                level=ValidationLevel.WARNING,
                                category='style',
                                message=f"Mutable default argument in function '{node.name}'",
                                line=node.lineno,
                                suggestion="Use None as default and initialize inside function"
                            ))
        except:
            pass
        
        return issues


class JavaScriptValidator:
    """Validates JavaScript code."""
    
    DANGEROUS_PATTERNS = [
        (r'\beval\s*\(', 'eval() is dangerous'),
        (r'\bFunction\s*\(', 'Function() constructor can be dangerous'),
        (r'\bdocument\.write\s*\(', 'document.write() is discouraged'),
    ]
    
    def validate(self, code: str) -> ValidationResult:
        """Validate JavaScript code."""
        issues: List[ValidationIssue] = []
        
        # Basic syntax check using Node.js if available
        syntax_issues = self._check_syntax(code)
        issues.extend(syntax_issues)
        
        # Security patterns
        for pattern, message in self.DANGEROUS_PATTERNS:
            if re.search(pattern, code):
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category='security',
                    message=message
                ))
        
        is_valid = not any(i.level == ValidationLevel.ERROR for i in issues)
        return ValidationResult(is_valid=is_valid, issues=issues)
    
    def _check_syntax(self, code: str) -> List[ValidationIssue]:
        """Check JavaScript syntax using Node.js."""
        issues = []
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(code)
                temp_path = f.name
            
            result = subprocess.run(
                ['node', '--check', temp_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                error_msg = result.stderr.strip()
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    category='syntax',
                    message=f"JavaScript syntax error: {error_msg}"
                ))
            
            os.unlink(temp_path)
            
        except FileNotFoundError:
            # Node.js not available
            pass
        except subprocess.TimeoutExpired:
            pass
        except Exception as e:
            logger.warning(f"JS syntax check failed: {e}")
        
        return issues


class CodeValidator:
    """
    Multi-language code validator with self-correction support.
    
    Features:
    - Validates Python, JavaScript, and other languages
    - Checks syntax, imports, security, and style
    - Provides detailed error messages with suggestions
    - Supports LLM-based auto-correction
    """
    
    def __init__(
        self,
        llm_client=None,
        available_packages: Optional[Set[str]] = None
    ):
        """
        Initialize the code validator.
        
        Args:
            llm_client: Optional LLM client for self-correction
            available_packages: Set of packages known to be installed
        """
        self.llm_client = llm_client
        self.python_validator = PythonValidator(available_packages)
        self.js_validator = JavaScriptValidator()
    
    def validate(
        self,
        code: str,
        language: str = "python"
    ) -> ValidationResult:
        """
        Validate code in the specified language.
        
        Args:
            code: Source code to validate
            language: Programming language
            
        Returns:
            ValidationResult with issues and validity status
        """
        language = language.lower()
        
        if language in ('python', 'py'):
            return self.python_validator.validate(code)
        elif language in ('javascript', 'js', 'typescript', 'ts'):
            return self.js_validator.validate(code)
        else:
            # For unknown languages, just check for obvious issues
            return self._basic_validate(code)
    
    def _basic_validate(self, code: str) -> ValidationResult:
        """Basic validation for unknown languages."""
        issues = []
        
        # Check for unbalanced brackets
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        
        for i, char in enumerate(code):
            if char in brackets:
                stack.append((char, i))
            elif char in brackets.values():
                if stack and brackets[stack[-1][0]] == char:
                    stack.pop()
                else:
                    issues.append(ValidationIssue(
                        level=ValidationLevel.ERROR,
                        category='syntax',
                        message=f"Unmatched bracket '{char}' at position {i}"
                    ))
        
        for bracket, pos in stack:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                category='syntax',
                message=f"Unclosed bracket '{bracket}' at position {pos}"
            ))
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues
        )
    
    async def validate_and_fix(
        self,
        code: str,
        language: str = "python",
        max_attempts: int = 3,
        context: str = ""
    ) -> ValidationResult:
        """
        Validate code and attempt to fix any issues using LLM.
        
        Args:
            code: Source code to validate
            language: Programming language
            max_attempts: Maximum fix attempts
            context: Additional context about what the code should do
            
        Returns:
            ValidationResult with fixed code if successful
        """
        current_code = code
        all_issues: List[ValidationIssue] = []
        
        for attempt in range(max_attempts):
            result = self.validate(current_code, language)
            
            if result.is_valid:
                return ValidationResult(
                    is_valid=True,
                    issues=all_issues,
                    fixed_code=current_code if current_code != code else None,
                    fix_attempts=attempt
                )
            
            all_issues.extend(result.errors)
            
            # Try to fix with LLM
            if self.llm_client and result.errors:
                fixed_code = await self._fix_with_llm(
                    current_code,
                    result.errors,
                    language,
                    context
                )
                
                if fixed_code and fixed_code != current_code:
                    logger.info(f"Attempt {attempt + 1}: LLM provided fix")
                    current_code = fixed_code
                else:
                    logger.warning(f"Attempt {attempt + 1}: LLM couldn't fix")
                    break
            else:
                break
        
        # Final validation
        final_result = self.validate(current_code, language)
        
        return ValidationResult(
            is_valid=final_result.is_valid,
            issues=final_result.issues,
            fixed_code=current_code if current_code != code else None,
            fix_attempts=max_attempts
        )
    
    async def _fix_with_llm(
        self,
        code: str,
        errors: List[ValidationIssue],
        language: str,
        context: str
    ) -> Optional[str]:
        """Use LLM to fix code errors."""
        if not self.llm_client:
            return None
        
        error_descriptions = "\n".join([
            f"- Line {e.line}: {e.message}" + (f" (Suggestion: {e.suggestion})" if e.suggestion else "")
            for e in errors
        ])
        
        prompt = f"""Fix the following {language} code that has these errors:

ERRORS:
{error_descriptions}

ORIGINAL CODE:
```{language}
{code}
```

{f"CONTEXT: {context}" if context else ""}

INSTRUCTIONS:
1. Fix ALL the errors listed above
2. Keep the same functionality and structure
3. Don't add unnecessary changes
4. Ensure the fixed code is complete and runnable

Return ONLY the fixed code, no explanations. The code should be directly executable.
"""
        
        try:
            response = await self.llm_client.generate(
                prompt,
                system_instruction=f"You are an expert {language} developer. Fix code errors precisely and return only the corrected code."
            )
            
            # Extract code from response
            fixed_code = self._extract_code(response.content, language)
            
            return fixed_code
            
        except Exception as e:
            logger.error(f"LLM fix failed: {e}")
            return None
    
    def _extract_code(self, response: str, language: str) -> str:
        """Extract code block from LLM response."""
        # Try to find code block
        patterns = [
            rf'```{language}\s*(.*?)\s*```',
            rf'```\s*(.*?)\s*```',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no code block, assume the entire response is code
        # Remove any markdown artifacts
        code = response.strip()
        code = re.sub(r'^```\w*\s*', '', code)
        code = re.sub(r'\s*```$', '', code)
        
        return code


class CodeFixer:
    """
    Iterative code fixer using multiple strategies.
    
    Strategies:
    1. Syntax fixes (missing colons, brackets, etc.)
    2. Import fixes (add missing imports)
    3. Type fixes (fix type mismatches)
    4. LLM-based general fixes
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.validator = CodeValidator(llm_client)
    
    async def fix_code(
        self,
        code: str,
        language: str = "python",
        purpose: str = "",
        available_imports: Optional[List[str]] = None,
        max_iterations: int = 5
    ) -> Tuple[str, bool, List[str]]:
        """
        Iteratively fix code until it's valid.
        
        Args:
            code: Original code
            language: Programming language
            purpose: What the code should do
            available_imports: List of available imports
            max_iterations: Maximum fix iterations
            
        Returns:
            Tuple of (fixed_code, is_valid, list_of_fixes_applied)
        """
        current_code = code
        fixes_applied = []
        
        for iteration in range(max_iterations):
            result = self.validator.validate(current_code, language)
            
            if result.is_valid:
                return current_code, True, fixes_applied
            
            # Try quick fixes first
            fixed, quick_fixes = self._apply_quick_fixes(
                current_code, 
                result.errors,
                language
            )
            
            if quick_fixes:
                fixes_applied.extend(quick_fixes)
                current_code = fixed
                continue
            
            # Try LLM fix
            if self.llm_client:
                result = await self.validator.validate_and_fix(
                    current_code,
                    language,
                    max_attempts=1,
                    context=purpose
                )
                
                if result.fixed_code:
                    current_code = result.fixed_code
                    fixes_applied.append("LLM-based fix applied")
                    
                    if result.is_valid:
                        return current_code, True, fixes_applied
            else:
                break
        
        # Final validation
        final_result = self.validator.validate(current_code, language)
        
        return current_code, final_result.is_valid, fixes_applied
    
    def _apply_quick_fixes(
        self,
        code: str,
        errors: List[ValidationIssue],
        language: str
    ) -> Tuple[str, List[str]]:
        """Apply quick automatic fixes."""
        fixes = []
        fixed_code = code
        
        if language not in ('python', 'py'):
            return fixed_code, fixes
        
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
                
                # Unmatched parenthesis/bracket
                elif 'unexpected eof' in msg or 'never closed' in msg:
                    # Count brackets
                    open_count = code.count('(') - code.count(')')
                    if open_count > 0:
                        fixed_code = fixed_code.rstrip() + ')' * open_count
                        fixes.append(f"Added {open_count} closing parenthesis")
                    
                    open_count = code.count('[') - code.count(']')
                    if open_count > 0:
                        fixed_code = fixed_code.rstrip() + ']' * open_count
                        fixes.append(f"Added {open_count} closing bracket")
                    
                    open_count = code.count('{') - code.count('}')
                    if open_count > 0:
                        fixed_code = fixed_code.rstrip() + '}' * open_count
                        fixes.append(f"Added {open_count} closing brace")
        
        return fixed_code, fixes


def create_validator(
    llm_client=None,
    available_packages: Optional[Set[str]] = None
) -> CodeValidator:
    """Factory function to create a code validator."""
    return CodeValidator(llm_client, available_packages)


def create_fixer(llm_client=None) -> CodeFixer:
    """Factory function to create a code fixer."""
    return CodeFixer(llm_client)
