import streamlit as st

from app.agent.agent import Agent


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SIH 26117 Agent",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("🤖 SIH 26117 — Agent Workbench")

st.caption(
    "Live Agent Execution • Selective RAG • Dynamic Replanning • Qwen3"
)


# ============================================================
# AGENT
# ============================================================

@st.cache_resource
def get_agent():
    return Agent()


agent = get_agent()


# ============================================================
# USER INPUT
# ============================================================

question = st.text_area(
    "Enter your request",
    height=120,
    placeholder=(
        "Example: According to the local company knowledge base, "
        "what information is required during inspection of an MAH factory?"
    ),
)


run = st.button(
    "Run Agent",
    type="primary",
    use_container_width=True,
)


# ============================================================
# RUN AGENT
# ============================================================

if run:

    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    # Reset previous streamed answer.
    st.session_state.pop("streamed_answer", None)

    # --------------------------------------------------------
    # UI CONTAINERS
    # --------------------------------------------------------

    trace_box = st.container()

    status_box = st.empty()

    answer_box = st.container()

    rag_box = st.container()

    verifier_box = st.container()

    plan_box = st.container()

    # --------------------------------------------------------
    # STATE
    # --------------------------------------------------------

    steps = []

    sources = []

    context = ""

    intent = "unknown"

    classification_reason = ""

    replan_count = 0

    answer_placeholder = None

    # --------------------------------------------------------
    # LIVE EXECUTION HEADER
    # --------------------------------------------------------

    with trace_box:

        st.subheader("🧠 Live Agent Execution")

        st.caption(
            "Observable execution events are shown live. "
            "Private model chain-of-thought is not exposed."
        )

        step_placeholder = st.empty()

    status_box.info("Agent starting...")


    # ========================================================
    # EVENT LOOP
    # ========================================================

    for event in agent.run_stream(question):

        event_type = event.get("type")


        # ====================================================
        # GENERAL STEP
        # ====================================================

        if event_type == "step":

            message = event.get(
                "message",
                "Processing..."
            )

            steps.append(message)

            with step_placeholder.container():

                for i, step in enumerate(
                    steps,
                    start=1
                ):

                    st.write(
                        f"**{i}.** 🟢 {step}"
                    )

            status_box.info(message)


        # ====================================================
        # CLASSIFICATION
        # ====================================================

        elif event_type == "classification":

            intent = event.get(
                "intent",
                "unknown"
            )

            classification_reason = event.get(
                "reason",
                ""
            )

            with step_placeholder.container():

                for i, step in enumerate(
                    steps,
                    start=1
                ):

                    st.write(
                        f"**{i}.** 🟢 {step}"
                    )

                st.write(
                    f"**Intent:** `{intent.upper()}`"
                )

                if classification_reason:

                    st.write(
                        f"**Reason:** {classification_reason}"
                    )


        # ====================================================
        # EXECUTION PLAN
        # ====================================================

        elif event_type == "plan":

            replan_count = event.get(
                "replan_count",
                replan_count
            )

            with plan_box:

                st.subheader("📋 Execution Plan")

                if replan_count > 0:

                    st.warning(
                        f"🔄 Replacement Plan — Replan {replan_count}"
                    )

                else:

                    st.info(
                        "🟢 Initial Execution Plan"
                    )

                st.write(
                    f"**Goal:** "
                    f"{event.get('goal', '')}"
                )

                # IMPORTANT:
                # lowercase "complexity"
                st.write(
                    f"**Complexity:** "
                    f"`{event.get('complexity', 'single_path')}`"
                )

                plan_steps = event.get(
                    "steps",
                    []
                )

                for plan_step in plan_steps:

                    step_id = plan_step.get(
                        "id",
                        "?"
                    )

                    description = plan_step.get(
                        "description",
                        ""
                    )

                    tool = plan_step.get(
                        "tool",
                        "unknown"
                    )

                    st.write(
                        f"**{step_id}.** "
                        f"{description} "
                        f"`[{tool}]`"
                    )

                final_output = event.get(
                    "final_output"
                )

                if final_output:

                    st.caption(
                        f"Final output: `{final_output}`"
                    )


        # ====================================================
        # DYNAMIC REPLANNING
        # ====================================================

        elif event_type == "replan":

            attempt = event.get(
                "attempt",
                0
            )

            maximum = event.get(
                "max_attempts",
                0
            )

            reason = event.get(
                "reason",
                "The previous execution did not produce sufficient results."
            )

            replan_count = attempt

            with step_placeholder.container():

                for i, step in enumerate(
                    steps,
                    start=1
                ):

                    st.write(
                        f"**{i}.** 🟢 {step}"
                    )

                st.warning(
                    f"🔄 **Dynamic Replanning "
                    f"{attempt}/{maximum}**"
                )

                st.write(
                    f"**Reason:** {reason}"
                )

            status_box.warning(
                f"🔄 Replanning execution "
                f"({attempt}/{maximum})..."
            )


        # ====================================================
        # PLAN STEP START
        # ====================================================

        elif event_type == "plan_step_start":

            step_id = event.get(
                "step_id",
                "?"
            )

            total = event.get(
                "total",
                "?"
            )

            description = event.get(
                "description",
                event.get(
                    "action",
                    "Executing..."
                )
            )

            status_box.info(
                f"⚙️ Executing step "
                f"{step_id}/{total}: "
                f"{description}"
            )


        # ====================================================
        # PLAN STEP COMPLETE
        # ====================================================

        elif event_type == "plan_step_complete":

            step_id = event.get(
                "step_id",
                "?"
            )

            status_box.success(
                f"✅ Completed plan step {step_id}"
            )


        # ====================================================
        # RAG RETRIEVAL
        # ====================================================

        elif event_type == "retrieval":

            sources = event.get(
                "sources",
                []
            )

            context = event.get(
                "context",
                ""
            )

            with rag_box:

                st.subheader(
                    "📚 RAG Retrieval"
                )

                st.write(
                    f"**Retrieved "
                    f"{len(sources)} source(s)**"
                )

                for i, source in enumerate(
                    sources,
                    start=1
                ):

                    filename = source.get(
                        "filename",
                        source.get(
                            "source",
                            "Unknown document"
                        )
                    )

                    location = source.get(
                        "location",
                        source.get(
                            "page",
                            "Unknown location"
                        )
                    )

                    score = source.get(
                        "score"
                    )

                    title = (
                        f"Source {i} — "
                        f"{filename}"
                    )

                    if score is not None:

                        try:

                            title += (
                                f" — score: "
                                f"{float(score):.4f}"
                            )

                        except (
                            TypeError,
                            ValueError
                        ):

                            pass

                    with st.expander(title):

                        st.write(
                            f"**Document:** "
                            f"{filename}"
                        )

                        st.write(
                            f"**Location:** "
                            f"{location}"
                        )

                        if score is not None:

                            try:

                                st.write(
                                    f"**Similarity score:** "
                                    f"`{float(score):.4f}`"
                                )

                            except (
                                TypeError,
                                ValueError
                            ):

                                st.write(
                                    f"**Similarity score:** "
                                    f"`{score}`"
                                )

                        metadata = {
                            k: v
                            for k, v in source.items()
                            if k not in {
                                "content",
                                "text"
                            }
                        }

                        if metadata:

                            st.json(metadata)

                with st.expander(
                    "🔎 Evidence context sent to Qwen3",
                    expanded=True,
                ):

                    if context:

                        st.code(
                            context,
                            language="text"
                        )

                    else:

                        st.caption(
                            "No evidence context returned."
                        )


        # ====================================================
        # VERIFICATION
        # ====================================================

        elif event_type == "verification":

            verified = event.get(
                "verified",
                False
            )

            sufficient = event.get(
                "sufficient",
                verified
            )

            reason = event.get(
                "reason",
                ""
            )

            evidence_score = event.get(
                "score"
            )

            with verifier_box:

                st.subheader(
                    "🔍 Evidence Verification"
                )

                if sufficient:

                    st.success(
                        "✅ Retrieved evidence is sufficient."
                    )

                else:

                    st.warning(
                        "⚠️ Retrieved evidence is insufficient."
                    )

                if reason:

                    st.write(
                        f"**Verifier:** {reason}"
                    )

                if evidence_score is not None:

                    try:

                        st.write(
                            f"**Evidence score:** "
                            f"`{float(evidence_score):.4f}`"
                        )

                    except (
                        TypeError,
                        ValueError
                    ):

                        st.write(
                            f"**Evidence score:** "
                            f"`{evidence_score}`"
                        )


        # ====================================================
        # ANSWER START
        # ====================================================

        elif event_type == "answer_start":

            with answer_box:

                st.subheader(
                    "💬 Final Answer"
                )

                answer_placeholder = st.empty()

            # Always reset streamed answer here.
            st.session_state.streamed_answer = ""


        # ====================================================
        # STREAMED TOKEN
        # ====================================================

        elif event_type == "token":

            if answer_placeholder is not None:

                if (
                    "streamed_answer"
                    not in st.session_state
                ):

                    st.session_state.streamed_answer = ""

                st.session_state.streamed_answer += (
                    event.get(
                        "content",
                        ""
                    )
                )

                answer_placeholder.markdown(
                    st.session_state.streamed_answer
                )


        # ====================================================
        # AGENT ERROR
        # ====================================================

        elif event_type == "error":

            message = event.get(
                "message",
                "Unknown agent error."
            )

            status_box.error(
                f"❌ {message}"
            )


        # ====================================================
        # DONE
        # ====================================================

        elif event_type == "done":

            result = event.get(
                "result",
                {}
            )

            # ------------------------------------------------
            # Clear temporary streamed answer state
            # ------------------------------------------------

            st.session_state.pop(
                "streamed_answer",
                None
            )

            # ------------------------------------------------
            # Result status
            # ------------------------------------------------

            status = result.get(
                "status",
                "completed"
            )

            if status == "completed":

                status_box.success(
                    "✅ Agent execution completed."
                )

            elif status == "insufficient_evidence":

                status_box.warning(
                    "⚠️ Agent stopped safely — "
                    "insufficient local evidence."
                )

            elif status == "failed":

                status_box.error(
                    "❌ Agent execution failed."
                )

            else:

                status_box.info(
                    f"Agent finished with status: "
                    f"`{status}`"
                )


            # ------------------------------------------------
            # Execution Summary
            # ------------------------------------------------

            with st.expander(
                "Execution Summary",
                expanded=False
            ):

                st.write(
                    f"**Dynamic replans:** "
                    f"{result.get('replans', 0)}"
                )

                st.write(
                    f"**Status:** "
                    f"`{result.get('status', 'unknown')}`"
                )

                st.write(
                    f"**Intent:** "
                    f"`{result.get('intent', 'unknown').upper()}`"
                )

                st.write(
                    f"**Classification reason:** "
                    f"{result.get('classification_reason', '')}"
                )


                # --------------------------------------------
                # Execution trace
                # --------------------------------------------

                execution_steps = result.get(
                    "steps",
                    []
                )

                if execution_steps:

                    st.divider()

                    st.write(
                        "**Execution Trace**"
                    )

                    for i, step in enumerate(
                        execution_steps,
                        start=1
                    ):

                        st.write(
                            f"{i}. {step}"
                        )


                # --------------------------------------------
                # Verification summary
                # --------------------------------------------

                if "verification" in result:

                    st.divider()

                    st.write(
                        "**Verification Result**"
                    )

                    st.json(
                        result["verification"]
                    )


            # ------------------------------------------------
            # General request
            # ------------------------------------------------

            if result.get("intent") == "general":

                with rag_box:

                    st.divider()

                    st.success(
                        "ℹ️ RAG was not used for this request."
                    )


            # ------------------------------------------------
            # Insufficient evidence
            # ------------------------------------------------

            if status == "insufficient_evidence":

                with verifier_box:

                    st.divider()

                    st.warning(
                        "🛑 Safe stop: the agent could not "
                        "obtain sufficient evidence from "
                        "the local knowledge base after "
                        "the allowed replanning attempts."
                    )


# ============================================================
# SYSTEM ARCHITECTURE
# ============================================================

st.divider()

with st.expander(
    "System Architecture"
):

    st.code(
        """
USER
  ↓
AGENT
  ↓
TASK CLASSIFICATION
  ↓
PLANNER
  ↓
EXECUTE PLAN
  ↓
TOOL
  ↓
OBSERVE RESULT
  ↓
VERIFIER
  ↓
┌──────────────────────────┐
│ Evidence sufficient?     │
└────────────┬─────────────┘
             │
       ┌─────┴─────┐
       │           │
      YES          NO
       │           │
       ↓           ↓
   Next Step    Replan?
                   │
             ┌─────┴─────┐
             │           │
            YES          NO
             │           │
             ↓           ↓
        NEW PLAN      SAFE STOP
             │
             ↓
          EXECUTE
             │
             ↓
          VERIFY
             │
             ↓
           QWEN3
             │
             ↓
      STREAMED ANSWER
        """,
        language="text"
    )