import subprocess, sys

if __name__ == "__main__":
    raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", "test_agent.py"]))
