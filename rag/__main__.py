"""
Interactive & CLI Entry Point for the Sovereign RAG Engine (Harsha).

Usage:
  python -m rag                     # Interactive Q&A Mode in terminal
  python -m rag "<question>"        # Single-query execution
  python rag                        # Interactive Q&A Mode
  python cli.py rag                 # Interactive Q&A Mode
"""

from pathlib import Path
import sys

# Ensure repository root is in sys.path for direct execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.rag import LocalRAG


def handle_query(rag: LocalRAG, query: str):
    print("\n--------------------------------------------------")
    print(f"QUERY: {query}")
    print("--------------------------------------------------")

    result = rag.search(query, top_k=3)

    if not result.get("found"):
        print("[-] Status: No relevant evidence found above the confidence threshold (0.35).")
        return

    confidence = result.get("confidence", 0.0)
    sources = result.get("sources", [])
    visuals = result.get("visual_handoffs", [])

    print(f"[+] Status      : Evidence Found (Top Relevance Score: {confidence:.4f})")
    print(f"[+] Sources     : {len(sources)} matching chunk(s)")
    if visuals:
        print(f"[+] CAD/Vision  : {len(visuals)} engineering drawing handoff(s) for Qwen2.5-VL")

    print("\n[TOP MATCHING SOURCES]")
    for i, s in enumerate(sources, 1):
        ocr_flag = " (OCR Used)" if s.get("ocr_used") else ""
        print(f"  {i}. {s.get('filename')} | Page: {s.get('page_number')} | Score: {s.get('score', 0.0):.4f}{ocr_flag}")

    # Generate grounded answer using local LLM
    print("\n[GROUNDED ANSWER]")
    try:
        answer, _ = rag.answer(query, top_k=3)
        print(answer)
    except Exception as e:
        print(f"(LLM answer generation note: {e})")

    print("\n[RETRIEVED LOCAL EVIDENCE SNIPPET]")
    context = result.get("context", "")
    print(context[:800].strip())
    if len(context) > 800:
        print("\n... [truncated, evidence continues]")
    print("--------------------------------------------------\n")


def main():
    print("==================================================================")
    print("      SOVEREIGN INDUSTRIAL RAG ENGINE - TERMINAL INTERFACE        ")
    print("          Mangalore Refinery & Petrochemicals Ltd (MRPL)          ")
    print("==================================================================")
    print("[*] Initializing LocalRAG (Loading BGE-M3 Embeddings & Qdrant)...")

    try:
        rag = LocalRAG()
        print("[+] RAG Engine Ready!\n")
    except Exception as e:
        print(f"[-] Initialization Error: {e}")
        sys.exit(1)

    # If question passed via CLI arguments, run single query
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:]).strip()
        handle_query(rag, query)
        return

    # Otherwise enter Interactive Q&A Mode
    print("Type your industrial questions below.")
    print("Examples:")
    print("  - What is the trip limit for PT-101?")
    print("  - What were the safety audit findings for XV-301?")
    print("  - What is the installed capacity of MRPL Phase III?")
    print("  - What are the major findings in the inspection report?")
    print("(Type 'exit', 'quit', or 'q' to exit)\n")

    while True:
        try:
            query = input("Ask RAG > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Exiting RAG interface. Goodbye!")
                break

            handle_query(rag, query)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting RAG interface. Goodbye!")
            break


if __name__ == "__main__":
    main()
