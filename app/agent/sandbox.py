import subprocess
import tempfile
import sys
import os
from pathlib import Path
from typing import Dict, Any, Tuple

import ast

class SecurityScanner(ast.NodeVisitor):
    """Zero-Trust AST Scanner to block dangerous Python operations."""
    def __init__(self):
        self.violations = []
        self.banned_imports = {"subprocess", "shutil", "socket", "urllib", "requests"}
        self.banned_calls = {"eval", "exec", "remove", "rmdir", "unlink", "system", "popen"}

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name.split('.')[0] in self.banned_imports:
                self.violations.append(f"Importing {alias.name} is blocked by AST Zero-Trust Scanner.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module and node.module.split('.')[0] in self.banned_imports:
            self.violations.append(f"Importing from {node.module} is blocked by AST Zero-Trust Scanner.")
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id in self.banned_calls:
                self.violations.append(f"Call to '{node.func.id}' is blocked by AST Zero-Trust Scanner.")
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in self.banned_calls:
                self.violations.append(f"Call to '{node.func.attr}' is blocked by AST Zero-Trust Scanner.")
        self.generic_visit(node)

def _scan_code_security(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
        scanner = SecurityScanner()
        scanner.visit(tree)
        return scanner.violations
    except SyntaxError as e:
        return [f"Syntax Error: {e}"]

class PythonSandbox:
    """
    Executes Python code locally in a subprocess.
    Captures stdout and stderr to feed back into the Agent Loop for bug solving.
    """
    
    def __init__(self, timeout_seconds: int = 30):
        self.timeout_seconds = timeout_seconds
        
    def execute(self, code: str) -> Dict[str, Any]:
        """
        Write code to a temporary file, run it, and capture output.
        """
        if not code or not code.strip():
            return {"success": False, "output": "No code provided to execute."}

        # Strip markdown formatting if present
        code = code.strip()
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
            
        # --- ENTERPRISE SECURITY: AST SCAN ---
        violations = _scan_code_security(code)
        if violations:
            violation_str = "\n".join(violations)
            return {
                "success": False, 
                "output": f"[SECURITY BLOCKED] Code execution was prevented:\n{violation_str}"
            }
        # -------------------------------------
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            temp_path = f.name
            
        try:
            # Run the python script
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds
            )
            
            output = result.stdout
            if result.stderr:
                output += "\n[STDERR]\n" + result.stderr
                
            return {
                "success": result.returncode == 0,
                "output": output.strip() if output.strip() else "[Code executed successfully with no output]"
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": f"Execution timed out after {self.timeout_seconds} seconds."
            }
        except Exception as e:
            return {
                "success": False,
                "output": f"Sandbox execution error: {str(e)}"
            }
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass
