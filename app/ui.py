import sys
from pathlib import Path

# Add the project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import streamlit as st

from app.rag import LocalRAG


st.set_page_config(
    page_title="SIH 26117 - Private Industrial RAG",
    page_icon="🔒",
    layout="wide"
)


st.title(
    "SIH 26117 — Private Industrial Knowledge RAG"
)

st.caption(
    "Local BGE-M3 + Qdrant + Ollama + Qwen3 8B (Sovereign / Air-Gapped)"
)

st.divider()


@st.cache_resource
def load_rag():
    return LocalRAG()


question = st.text_area(
    "Ask your local knowledge base",
    placeholder=(
        "Example: What checks are required in an inspection checklist?"
    ),
    height=120
)


if st.button(
    "Ask Knowledge Base",
    type="primary"
):

    if not question.strip():
        st.warning(
            "Please enter a question."
        )
    else:
        with st.spinner(
            "Searching local knowledge base..."
        ):
            try:
                rag = load_rag()
                answer, sources = rag.answer(
                    question
                )

                st.subheader("Answer")
                st.write(answer)

                st.subheader("Sources & Provenance")

                if sources:
                    for source in sources:
                        ocr_badge = " *(Extracted via Local OCR)*" if source.get("ocr_used") else ""
                        tags_str = f"  \nIndustrial Tags: `{', '.join(source['tags'])}`" if source.get("tags") else ""
                        st.markdown(
                            f"**{source['filename']}**{ocr_badge}  \n"
                            f"Location: {source['location']} | Category: {source.get('category', 'general')} | Type: {source.get('document_type', 'general')}  \n"
                            f"Similarity Score: `{source['score']}`{tags_str}"
                        )
                        st.markdown("---")
                else:
                    st.write(
                        "No relevant sources found."
                    )

            except Exception as error:
                st.error(
                    f"RAG error: {error}"
                )
                st.exception(error)