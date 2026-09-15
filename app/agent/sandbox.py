import subprocess
import tempfile
import sys
import os
from pathlib import Path
from typing import Dict, Any, Tuple

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
