import sys
import os
from pathlib import Path

# Ensure the app module can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.rag import LocalRAG

def test_retrieval():
    print("==================================================")
    print("SIH 26117 - LOCAL RAG VERIFICATION TEST")
    print("==================================================\n")
    
    print("[1] Initializing RAG Engine (Loading Qdrant and BM25)...")
    try:
        rag = LocalRAG()
        print("    -> RAG Engine initialized successfully!\n")
    except Exception as e:
        print(f"    -> ERROR initializing RAG: {e}")
        print("    -> Tip: Make sure you aren't running multiple scripts accessing Qdrant at the same time.\n")
        return

    # Test Queries
    queries = [
        "What are the key findings in the scanned inspection report?",
        "What is the installed capacity of MRPL Phase III?",
        "What is the trip limit for PT-101?"
    ]

    for q in queries:
        print(f"--------------------------------------------------")
        print(f"QUERY: {q}")
        print(f"--------------------------------------------------")
        
        try:
            # P1 FIX: Verify the strict Agent JSON contract using search()
            result = rag.search(q, top_k=3)
            
            if not result.get("found"):
                print("    -> No documents retrieved or insufficient evidence.")
                continue
                
            sources = result.get("sources", [])
            print(f"    -> Retrieved {len(sources)} matching chunks.\n")
            print(f"    -> Retrieval Stats: {result.get('retrieval_stats')}\n")
            
            # Show the top retrieved chunk
            top_hit = sources[0]
            doc = top_hit.get('filename', 'Unknown')
            score = top_hit.get('score', 0.0)
            ocr = top_hit.get('ocr_used', False)
            
            print(f"    TOP MATCH (Score: {score:.4f})")
            print(f"    Document : {doc}")
            print(f"    OCR Used : {ocr}")
            print(f"    Chunk ID : {top_hit.get('chunk_id')}")
            
            # Print a snippet of the context citation
            context = result.get("context", "")
            print(f"    Context Snippet:\n{context[:250]}...\n")
            
        except Exception as e:
            print(f"    -> Query failed: {e}")

    print("==================================================")
    print("TEST COMPLETE")
    print("==================================================")

if __name__ == "__main__":
    test_retrieval()
