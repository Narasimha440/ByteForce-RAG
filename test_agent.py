import streamlit as st

from app.agent.agent import Agent


st.set_page_config(
    page_title="SIH 26117 Agent",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 SIH 26117 — Agent Workbench")
st.caption("Live Agent Execution • Selective RAG • Qwen3")

@st.cache_resource
def get_agent():
    return Agent()


agent = get_agent()

question = st.text_area(
    "Enter your request",
    height=120,
    placeholder="Example: What information is required during inspection of an MAH factory?",
)

run = st.button("Run Agent", type="primary", use_container_width=True)

if run:
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    # Containers let us update the UI while the generator is running.
    trace_box = st.container()
    status_box = st.empty()
    answer_box = st.empty()
    rag_box = st.container()

    steps = []
    sources = []
    context = ""
    intent = "unknown"
    classification_reason = ""
    answer_started = False

    with trace_box:
        st.subheader("🧠 Live Agent Execution")
        st.caption(
            "Observable execution events are shown live. "
            "Private model chain-of-thought is not exposed."
        )

        step_placeholder = st.empty()

    with status_box:
        st.info("Agent starting...")

    answer_placeholder = None

    for event in agent.run_stream(question):

        event_type = event["type"]

        if event_type == "step":
            steps.append(event["message"])

            with step_placeholder.container():
                for i, step in enumerate(steps, start=1):
                    st.write(f"**{i}.** 🟢 {step}")

            status_box.info(event["message"])

        elif event_type == "classification":
            intent = event["intent"]
            classification_reason = event["reason"]

            with step_placeholder.container():
                for i, step in enumerate(steps, start=1):
                    st.write(f"**{i}.** 🟢 {step}")

                st.write(f"**Intent:** `{intent.upper()}`")
                st.write(f"**Reason:** {classification_reason}")

        elif event_type == "retrieval":
            sources = event.get("sources", [])
            context = event.get("context", "")

            with rag_box:
                st.subheader("📚 RAG Retrieval")

                st.write(
                    f"**Retrieved {len(sources)} source(s)**"
                )

                for i, source in enumerate(sources, start=1):
                    filename = source.get(
                        "filename",
                        source.get("source", "Unknown document"),
                    )

                    location = source.get(
                        "location",
                        source.get("page", "Unknown location"),
                    )

                    score = source.get("score")

                    title = f"Source {i} — {filename}"

                    if score is not None:
                        title += f" — score: {score:.4f}"

                    with st.expander(title):
                        st.write(f"**Document:** {filename}")
                        st.write(f"**Location:** {location}")

                        if score is not None:
                            st.write(
                                f"**Similarity score:** `{score:.4f}`"
                            )

                        metadata = {
                            k: v
                            for k, v in source.items()
                            if k not in {"content", "text"}
                        }

                        if metadata:
                            st.json(metadata)

                with st.expander(
                    "🔎 Evidence context sent to Qwen3",
                    expanded=True,
                ):
                    st.code(context, language="text")

        elif event_type == "answer_start":
            answer_started = True

            with answer_box:
                st.subheader("💬 Final Answer")
                answer_placeholder = st.empty()

        elif event_type == "token":
            if answer_placeholder is not None:
                # Re-render accumulated answer so the user sees it growing.
                if "streamed_answer" not in st.session_state:
                    st.session_state.streamed_answer = ""

                st.session_state.streamed_answer += event["content"]

                answer_placeholder.markdown(
                    st.session_state.streamed_answer
                )

        elif event_type == "done":
            result = event["result"]

            # Clear temporary streamed answer after completion.
            st.session_state.pop("streamed_answer", None)

            status_box.success("Agent execution completed.")

            with st.expander("Execution Summary", expanded=False):
                st.write(f"**Intent:** `{result.get('intent', 'unknown').upper()}`")
                st.write(
                    f"**Classification reason:** "
                    f"{result.get('classification_reason', '')}"
                )

                for i, step in enumerate(
                    result.get("steps", []),
                    start=1,
                ):
                    st.write(f"{i}. {step}")

            if result.get("intent") == "general":
                with rag_box:
                    st.divider()
                    st.success(
                        "RAG was not used for this request."
                    )

st.divider()

with st.expander("System Architecture"):
    st.code(
        """
USER
  ↓
AGENT
  ↓
TASK CLASSIFICATION
  ├── GENERAL ───────────────→ Qwen3
  │
  └── RAG
       ↓
     BGE-M3
       ↓
     Qdrant
       ↓
     Retrieved Evidence
       ↓
     Qwen3
       ↓
  STREAMED ANSWER
        """,
        language="text",
    )
