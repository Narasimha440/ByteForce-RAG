from app.rag import LocalRAG


def main():
    rag = LocalRAG()

    question = (
        "According to the local knowledge base, "
        "what information is required during inspection "
        "of an MAH factory?"
    )

    print("\n" + "=" * 70)
    print("RAG RETRIEVAL TEST")
    print("=" * 70)

    result = rag.search(question)

    print("\nFOUND:", result.get("found"))
    print("RETRIEVAL SCORE:", result.get("confidence"))

    print("\nRETRIEVAL STATS:")
    print(result.get("retrieval_stats"))

    print("\nSOURCES:")

    for i, source in enumerate(result.get("sources", []), 1):
        print(f"\n[{i}]")
        print("Document :", source.get("filename"))
        print("Page     :", source.get("page_number"))
        print("Section  :", source.get("section"))
        print("Score    :", source.get("score"))

    print("\n" + "=" * 70)
    print("CONTEXT")
    print("=" * 70)

    print(result.get("context", ""))


if __name__ == "__main__":
    main()