"""
CLI Entry Point for the Sovereign RAG Engine (Harsha).

Usage:
  python -m rag "<question>"
  python -m rag                 # Run RAG engine self-check
"""

from pathlib import Path
import sys

# Ensure repository root is in sys.path for direct execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.rag import LocalRAG


def main():
    print("==================================================")
    print("SOVEREIGN INDUSTRIAL RAG ENGINE (SIH 26117)")
    print("==================================================")

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What is the trip limit for PT-101?"

    print(f"Query: {query}\n")
    print("Initializing LocalRAG...")
    try:
        rag = LocalRAG()
        result = rag.search(query, top_k=3)

        if not result.get("found"):
            print("Status: No relevant evidence found above threshold.")
            return

        print(f"Status     : Evidence Found (Top Score: {result.get('confidence', 0.0):.4f})")
        print(f"Sources    : {len(result.get('sources', []))} matching chunk(s)")
        print(f"Visuals    : {len(result.get('visual_handoffs', []))} CAD/diagram handoff(s)")
        print("--------------------------------------------------")
        print("Retrieved Evidence Context:")
        print("--------------------------------------------------")
        context = result.get("context", "")
        print(context[:1200])
        if len(context) > 1200:
            print(f"\n... [truncated, total characters: {len(context)}]")

    except Exception as e:
        print(f"RAG Error: {e}")
        sys.exit(1)

    print("==================================================")


if __name__ == "__main__":
    main()
