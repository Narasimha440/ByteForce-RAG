"""
Interactive & CLI Entry Point for the Sovereign RAG Engine (Harsha).

Usage:
  python -m rag                     # Interactive streaming Q&A Mode
  python -m rag "<question>"        # Single-query streaming execution
  python -m rag --chat              # Multi-turn conversational mode with memory
  python -m rag --eval              # Run accuracy benchmark (Hit Rate, MRR, P@K)
  python cli.py rag "<question>"    # Via unified CLI
  python cli.py rag --chat          # Via CLI, multi-turn chat with memory
  python cli.py rag --eval          # Via CLI, accuracy benchmark
"""

from pathlib import Path
import sys

# Ensure repository root is in sys.path for direct execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.rag import LocalRAG, is_conversational_query


def handle_query_streaming(rag: LocalRAG, query: str, history=None):
    """
    Run a single query with live streaming output.
    Prints tokens as they arrive from Ollama instead of waiting for full response.
    """
    print("\n--------------------------------------------------")
    print(f"USER: {query}")
    print("--------------------------------------------------")
    print("\n[MRPL ASSISTANT]\n", end="", flush=True)

    full_answer = ""
    sources = []

    # Use streaming generator
    try:
        for token in rag.answer_stream(query, top_k=3, history=history):
            print(token, end="", flush=True)
            full_answer += token
    except Exception as e:
        # Fallback to blocking answer if stream fails
        ans, sources, confidence = rag.answer(query, top_k=3, history=history)
        print(ans)
        full_answer = ans
        _print_sources(sources, confidence)
        return full_answer, sources

    print("\n")

    # After streaming is done, retrieve sources for citation display
    # (sources come from the blocking path; for streaming we re-use retrieve)
    try:
        hits = rag.retrieve(query, top_k=3)
        if hits:
            _, sources = rag.build_context(hits, question=query)
            confidence = rag._classify_confidence(sources)
            _print_sources(sources, confidence)
    except Exception:
        pass

    print("--------------------------------------------------\n")
    return full_answer, sources


def _print_sources(sources: list, confidence: str = ""):
    """Print verified source citations to terminal."""
    if sources:
        conf_badge = f"  [{confidence} CONFIDENCE]" if confidence else ""
        print(f"[VERIFIED SOURCES & CITATIONS]{conf_badge}")
        for i, s in enumerate(sources, 1):
            ocr_flag = " (OCR Extracted)" if s.get("ocr_used") else ""
            score = s.get("score", 0.0)
            print(
                f"  [{i}] {s.get('filename')} "
                f"(Page {s.get('page_number', 'N/A')}) — "
                f"Relevance: {score:.4f}{ocr_flag}"
            )
        # CAD/Vision handoff note
        visuals = [s for s in sources if s.get("visual_handoff")]
        if visuals:
            print(f"\n[+] CAD/Vision Handoff: {len(visuals)} drawing(s) ready for Qwen2.5-VL")
    print("--------------------------------------------------\n")


def run_chat_mode(rag: LocalRAG):
    """
    Multi-turn conversational mode with ConversationMemory.
    Automatically rewrites vague follow-up questions into self-contained queries.
    """
    from app.memory import ConversationMemory

    memory = ConversationMemory(max_turns=6)

    print("\n[CHAT MODE] Multi-turn conversation with memory enabled.")
    print("Follow-up questions like 'what about FT-204?' will be automatically resolved.")
    print("Type 'clear' to reset memory. Type 'exit' or 'q' to quit.\n")

    while True:
        try:
            query = input("Ask RAG > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Exiting chat. Goodbye!")
                break
            if query.lower() == "clear":
                memory.clear()
                print("[Memory cleared]\n")
                continue

            history = memory.get_context_window()
            full_answer, sources = handle_query_streaming(rag, query, history=history)

            # Record turn in memory
            memory.add_turn(query, full_answer, sources)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat. Goodbye!")
            break


def run_eval_mode():
    """Run the RAG accuracy benchmark and print results to terminal."""
    print("\n[EVAL MODE] Running RAG accuracy benchmark...")
    print("This evaluates retrieval quality using Hit Rate, MRR, and Precision@K.\n")
    try:
        from app.eval import RAGEvaluator
        evaluator = RAGEvaluator(top_k=5)
        evaluator.run_eval(verbose=True)
    except Exception as e:
        print(f"[-] Evaluation failed: {e}")
        print("    Tip: Make sure data is ingested first: python cli.py ingest")


def main():
    print("==================================================================")
    print("      SOVEREIGN INDUSTRIAL RAG ENGINE - TERMINAL INTERFACE        ")
    print("          Mangalore Refinery & Petrochemicals Ltd (MRPL)          ")
    print("==================================================================")

    args = sys.argv[1:]

    # ── Eval mode ─────────────────────────────────────────────────────────
    if "--eval" in args:
        run_eval_mode()
        return

    # ── Initialize RAG ────────────────────────────────────────────────────
    print("[*] Initializing LocalRAG (Loading BGE-M3 Embeddings & Qdrant)...")
    try:
        rag = LocalRAG()
        print("[+] RAG Engine Ready! (Streaming mode enabled)\n")
    except Exception as e:
        print(f"[-] Initialization Error: {e}")
        sys.exit(1)

    # ── Chat mode (multi-turn with memory) ────────────────────────────────
    if "--chat" in args or (len(args) == 0):
        if "--chat" in args:
            run_chat_mode(rag)
            return

        # No arguments → standard interactive mode (streaming, no memory)
        print("Type your industrial questions below.")
        print("Examples:")
        print("  - What is the trip limit for PT-101?")
        print("  - What were the safety audit findings for XV-301?")
        print("  - What is the installed capacity of MRPL Phase III?")
        print("(Type 'exit', 'quit', or 'q' to quit | '--chat' for multi-turn mode)\n")

        while True:
            try:
                query = input("Ask RAG > ").strip()
                if not query:
                    continue
                if query.lower() in ("exit", "quit", "q"):
                    print("Exiting RAG interface. Goodbye!")
                    break
                handle_query_streaming(rag, query)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting RAG interface. Goodbye!")
                break
        return

    # ── Single query via CLI args ─────────────────────────────────────────
    query = " ".join(a for a in args if not a.startswith("--")).strip()
    if query:
        handle_query_streaming(rag, query)


if __name__ == "__main__":
    main()
