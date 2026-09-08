import json
import requests

from typing import (
    Generator,
    Dict,
    Any,
    List,
)

from app.rag import LocalRAG, INSUFFICIENT_KNOWLEDGE_MESSAGE
from app.config import (
    OLLAMA_URL,
    OLLAMA_MODEL,
)

from app.agent.planner import Planner
from app.agent.verifier import EvidenceVerifier


class Agent:

    # Maximum number of dynamic replanning attempts.
    MAX_REPLANS = 2

    def __init__(self):

        self.rag = LocalRAG()

        # Reusable HTTP connection pool.
        self.session = requests.Session()

        self.planner = Planner(
            self.call_llm
        )

        self.verifier = EvidenceVerifier(
            self.call_llm
        )

        self.system_prompt = (
            "You are the reasoning model for a private, "
            "on-premise industrial AI workbench."
        )

    # ==========================================================
    # LLM CALL
    # ==========================================================

    def call_llm(
        self,
        prompt: str,
        temperature: float = 0.1,
    ) -> str:

        try:

            response = self.session.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": self.system_prompt,
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    "stream": False,
                    "options": {
                        "temperature": temperature
                    },
                },
                timeout=120,
            )

            response.raise_for_status()

            data = response.json()

            return data["message"]["content"]

        except requests.exceptions.RequestException as e:

            # Raise instead of returning fake JSON.
            # This prevents planner/verifier from treating
            # an LLM failure as a valid result.
            raise RuntimeError(
                f"LLM communication failed: {e}"
            ) from e

        except (
            KeyError,
            ValueError,
            TypeError,
        ) as e:

            raise RuntimeError(
                f"Invalid LLM response: {e}"
            ) from e

    # ==========================================================
    # STREAMING LLM
    # ==========================================================

    def call_llm_stream(
        self,
        prompt: str,
        temperature: float = 0.1,
    ) -> Generator[str, None, None]:

        try:

            response = self.session.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": self.system_prompt,
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    "stream": True,
                    "options": {
                        "temperature": temperature
                    },
                },
                stream=True,
                timeout=180,
            )

            response.raise_for_status()

            for line in response.iter_lines():

                if not line:
                    continue

                try:

                    data = json.loads(
                        line
                    )

                    token = (
                        data
                        .get("message", {})
                        .get("content", "")
                    )

                    if token:
                        yield token

                    if data.get(
                        "done",
                        False,
                    ):
                        break

                except json.JSONDecodeError:
                    continue

        except requests.exceptions.RequestException as e:

            yield (
                "\n[Error communicating with model: "
                f"{e}]"
            )

    # ==========================================================
    # CLASSIFICATION
    # ==========================================================

    def classify_task(
        self,
        question: str,
    ) -> Dict[str, str]:

        prompt = f"""
Classify the user's request into exactly ONE intent:

'rag'
or
'general'

Use 'rag' for:
- company/private knowledge
- industrial documentation
- SOPs
- operational procedures
- inspection requirements
- engineering documentation
- safety procedures
- technical documents
- schematics
- internal company information

Use 'general' for:
- generic coding
- generic mathematics
- greetings
- common knowledge
- general explanations

Return ONLY valid JSON:

{{
  "intent": "rag",
  "reason": "short reason"
}}

USER REQUEST:
{question}
"""

        try:

            raw = self.call_llm(
                prompt,
                temperature=0.0,
            )

            result = self._parse_json(
                raw
            )

            if not result:
                raise ValueError(
                    "Invalid classifier response"
                )

            intent = str(
                result.get(
                    "intent",
                    "rag",
                )
            ).lower()

            if intent not in {
                "rag",
                "general",
            }:
                intent = "rag"

            return {
                "intent": intent,
                "reason": str(
                    result.get(
                        "reason",
                        "",
                    )
                ),
            }

        except Exception:

            return {
                "intent": "rag",
                "reason": (
                    "Classification failed; "
                    "defaulting to RAG for safety."
                ),
            }

    # ==========================================================
    # PROMPTS
    # ==========================================================

    def build_rag_prompt(
        self,
        question: str,
        context: str,
    ) -> str:

        return f"""
You are operating inside a private industrial AI system.

Answer the user's question using ONLY the verified
retrieved local evidence below.

RULES:

- Do not use outside knowledge.
- Do not invent facts.
- Do not invent procedures.
- Do not invent measurements.
- Do not invent limits.
- Do not invent equipment specifications.
- If the evidence does not support something, say so.
- Clearly distinguish supported information from uncertainty.
- Mention relevant source files and locations when available.

If the evidence is insufficient, state:

"The local knowledge base does not contain enough
information to answer this."

VERIFIED LOCAL EVIDENCE:
{context}

USER QUESTION:
{question}
"""

    def build_general_prompt(
        self,
        question: str,
    ) -> str:

        return f"""
Answer the user's question directly and professionally.

USER QUESTION:
{question}
"""

    # ==========================================================
    # MAIN AGENT
    # ==========================================================

    def run_stream(
        self,
        question: str,
    ) -> Generator[
        Dict[str, Any],
        None,
        None,
    ]:

        # ------------------------------------------------------
        # EMPTY REQUEST
        # ------------------------------------------------------

        if not question or not question.strip():

            yield {
                "type": "step",
                "message": (
                    "Please provide a question or task."
                ),
            }

            yield {
                "type": "done",
                "result": {
                    "answer": (
                        "Please provide a question or task."
                    ),
                    "sources": [],
                    "steps": [],
                    "intent": "unknown",
                    "plan": [],
                    "status": "invalid_request",
                    "replans": 0,
                },
            }

            return

        # ------------------------------------------------------
        # EXECUTION STATE
        # ------------------------------------------------------

        steps_log: List[str] = []

        context = ""

        sources = []

        intent = "unknown"

        reason = ""

        plan = {}

        replans = 0

        # ------------------------------------------------------
        # RECEIVE REQUEST
        # ------------------------------------------------------

        steps_log.append(
            "Received user request"
        )

        yield {
            "type": "step",
            "message": "Received user request",
        }

        # ------------------------------------------------------
        # CLASSIFY
        # ------------------------------------------------------

        yield {
            "type": "step",
            "message": "Classifying intent...",
        }

        classification = self.classify_task(
            question
        )

        intent = classification[
            "intent"
        ]

        reason = classification[
            "reason"
        ]

        steps_log.append(
            f"Intent classified as {intent.upper()}"
        )

        yield {
            "type": "classification",
            "intent": intent,
            "reason": reason,
        }

        # ------------------------------------------------------
        # INITIAL PLAN
        # ------------------------------------------------------

        yield {
            "type": "step",
            "message": "Creating execution plan...",
        }

        plan = self.planner.create_plan(
            question,
            intent,
        )

        steps = plan.get(
            "steps",
            [],
        )

        steps_log.append(
            f"Created {len(steps)}-step execution plan"
        )

        yield {
            "type": "plan",
            "goal": plan.get(
                "goal",
                "",
            ),
            "complexity": plan.get(
                "complexity",
                "single_path",
            ),
            "steps": steps,
            "final_output": plan.get(
                "final_output",
                "answer",
            ),
            "replan_count": 0,
        }

        # ======================================================
        # EXECUTION / REPLANNING LOOP
        # ======================================================

        while True:

            should_replan = False

            replan_reason = ""

            verification = {}

            # --------------------------------------------------
            # EXECUTE CURRENT PLAN
            # --------------------------------------------------

            for plan_step in plan.get(
                "steps",
                [],
            ):

                tool = plan_step.get(
                    "tool"
                )

                step_id = plan_step.get(
                    "id"
                )

                params = plan_step.get(
                    "params",
                    {},
                )

                yield {
                    "type": "plan_step_start",
                    "step_id": step_id,
                    "total": len(
                        plan.get(
                            "steps",
                            [],
                        )
                    ),
                    "action": plan_step.get(
                        "action",
                        "",
                    ),
                    "tool": tool,
                    "description": plan_step.get(
                        "description",
                        "",
                    ),
                }

                # ==============================================
                # RAG SEARCH
                # ==============================================

                if tool == "rag_search":

                    search_query = params.get(
                        "query",
                        question,
                    )

                    yield {
                        "type": "step",
                        "message": (
                            "Searching knowledge base: "
                            f"'{search_query}'..."
                        ),
                    }

                    try:

                        rag_result = self.rag.search(
                            search_query
                        )

                    except Exception as e:

                        rag_result = {
                            "found": False,
                            "context": "",
                            "sources": [],
                            "error": str(e),
                        }

                    found = bool(
                        rag_result.get(
                            "found",
                            False,
                        )
                    )

                    context = rag_result.get(
                        "context",
                        "",
                    )

                    sources = rag_result.get(
                        "sources",
                        [],
                    )

                    # ------------------------------------------
                    # NO RETRIEVAL
                    # ------------------------------------------

                    if not found:

                        steps_log.append(
                            "No evidence found"
                        )

                        yield {
                            "type": "step",
                            "message": (
                                "No relevant local "
                                "evidence found."
                            ),
                        }

                        verification = {
                            "sufficient": False,
                            "reason": (
                                "The local knowledge base "
                                "returned no usable evidence."
                            ),
                            "supported_points": [],
                            "missing_points": [
                                question
                            ],
                        }

                        yield {
                            "type": "retrieval",
                            "query": search_query,
                            "sources": [],
                            "context": "",
                            "found": False,
                            "confidence": 0.0,
                            "retrieval_score": 0.0,
                            "retrieval_stats": rag_result.get("retrieval_stats", {}),
                        }

                        yield {
                            "type": "verification",
                            **verification,
                        }

                    else:

                        steps_log.append(
                            f"Retrieved {len(sources)} sources"
                        )

                        yield {
                            "type": "step",
                            "message": (
                                f"Retrieved "
                                f"{len(sources)} source(s)"
                            ),
                        }

                        yield {
                            "type": "retrieval",
                            "query": search_query,
                            "sources": sources,
                            "context": context,
                            "found": found,
                            "confidence": rag_result.get("confidence", 0.0),
                            "retrieval_score": rag_result.get("confidence", 0.0),
                            "retrieval_stats": rag_result.get("retrieval_stats", {}),
                        }

                        # --------------------------------------
                        # EVIDENCE VERIFICATION
                        # --------------------------------------

                        yield {
                            "type": "step",
                            "message": (
                                "Verifying retrieved "
                                "evidence..."
                            ),
                        }

                        verification = self.verifier.verify(
                            question=question,
                            rag_result=rag_result,
                        )

                        yield {
                            "type": "verification",
                            **verification,
                        }

                        if verification.get(
                            "sufficient",
                            False,
                        ):

                            steps_log.append(
                                "Evidence verification passed"
                            )

                            yield {
                                "type": "step",
                                "message": (
                                    "Evidence verification "
                                    "passed."
                                ),
                            }

                        else:

                            steps_log.append(
                                "Evidence verification failed"
                            )

                            yield {
                                "type": "step",
                                "message": (
                                    "Evidence is insufficient "
                                    "or not sufficiently relevant."
                                ),
                            }

                    # ------------------------------------------
                    # DECIDE WHETHER TO REPLAN
                    # ------------------------------------------

                    if not verification.get(
                        "sufficient",
                        False,
                    ):

                        should_replan = True

                        replan_reason = str(
                            verification.get(
                                "reason",
                                "Retrieved evidence "
                                "was insufficient.",
                            )
                        )

                        # Do not continue to generate_response.
                        break

                    yield {
                        "type": "plan_step_complete",
                        "step_id": step_id,
                    }

                # ==============================================
                # GENERATE RESPONSE
                # ==============================================

                elif tool == "generate_response":

                    # Safety check:
                    # RAG responses must NEVER be generated
                    # unless evidence has been verified.
                    if (
                        intent == "rag"
                        and not verification.get(
                            "sufficient",
                            False,
                        )
                    ):

                        should_replan = True

                        replan_reason = (
                            "Response generation was blocked "
                            "because the retrieved evidence "
                            "was not verified as sufficient."
                        )

                        break

                    if intent == "rag":

                        prompt = (
                            self.build_rag_prompt(
                                question,
                                context,
                            )
                        )

                        yield {
                            "type": "step",
                            "message": (
                                "Sending verified evidence "
                                "to Qwen3..."
                            ),
                        }

                    else:

                        prompt = (
                            self.build_general_prompt(
                                question
                            )
                        )

                    yield {
                        "type": "step",
                        "message": (
                            "Generating final response..."
                        ),
                    }

                    yield {
                        "type": "answer_start"
                    }

                    answer_tokens = []

                    for token in self.call_llm_stream(
                        prompt
                    ):

                        answer_tokens.append(
                            token
                        )

                        yield {
                            "type": "token",
                            "content": token,
                        }

                    answer = "".join(
                        answer_tokens
                    )

                    steps_log.append(
                        "Completed response generation"
                    )

                    yield {
                        "type": "plan_step_complete",
                        "step_id": step_id,
                    }

                    yield {
                        "type": "done",
                        "result": {
                            "answer": answer,
                            "sources": sources,
                            "steps": steps_log,
                            "intent": intent,
                            "classification_reason": reason,
                            "context": context,
                            "plan": plan.get(
                                "steps",
                                [],
                            ),
                            "status": "completed",
                            "replans": replans,
                            "verification": verification,
                        },
                    }

                    return

            # ==================================================
            # REPLANNING
            # ==================================================

            if should_replan:

                if replans >= self.MAX_REPLANS:

                    steps_log.append(
                        "Maximum replanning attempts reached"
                    )

                    yield {
                        "type": "step",
                        "message": (
                            "Maximum replanning attempts "
                            "reached. Stopping safely."
                        ),
                    }

                    safe_answer = INSUFFICIENT_KNOWLEDGE_MESSAGE

                    yield {
                        "type": "done",
                        "result": {
                            "answer": safe_answer,
                            "sources": sources,
                            "steps": steps_log,
                            "intent": intent,
                            "classification_reason": reason,
                            "context": context,
                            "plan": plan.get(
                                "steps",
                                [],
                            ),
                            "status": "insufficient_evidence",
                            "replans": replans,
                            "verification": verification,
                        },
                    }

                    return

                # ----------------------------------------------
                # START REPLAN
                # ----------------------------------------------

                replans += 1

                steps_log.append(
                    f"Dynamic replan {replans} started"
                )

                yield {
                    "type": "replan",
                    "attempt": replans,
                    "max_attempts": self.MAX_REPLANS,
                    "reason": replan_reason,
                }

                yield {
                    "type": "step",
                    "message": (
                        f"🔄 Replanning attempt "
                        f"{replans}/{self.MAX_REPLANS}..."
                    ),
                }

                yield {
                    "type": "step",
                    "message": (
                        f"Reason: {replan_reason}"
                    ),
                }

                # ----------------------------------------------
                # CREATE NEW PLAN
                # ----------------------------------------------

                try:

                    new_plan = self.planner.replan(
                        question=question,
                        intent=intent,
                        previous_plan=plan,
                        feedback=verification,
                    )

                except Exception as e:

                    yield {
                        "type": "step",
                        "message": (
                            f"Replanning failed: {e}"
                        ),
                    }

                    new_plan = self.planner._fallback_plan(
                        question,
                        intent,
                    )

                plan = new_plan

                steps = plan.get(
                    "steps",
                    [],
                )

                steps_log.append(
                    f"Created replacement "
                    f"{len(steps)}-step plan"
                )

                yield {
                    "type": "step",
                    "message": (
                        f"New execution plan ready: "
                        f"{len(steps)} step(s)"
                    ),
                }

                yield {
                    "type": "plan",
                    "goal": plan.get(
                        "goal",
                        "",
                    ),
                    "complexity": plan.get(
                        "complexity",
                        "single_path",
                    ),
                    "steps": steps,
                    "final_output": plan.get(
                        "final_output",
                        "answer",
                    ),
                    "replan_count": replans,
                }

                # Reset evidence before retrying.
                context = ""

                sources = []

                verification = {}

                # Restart the execution loop
                continue

            # Safety fallback.
            break

        # ======================================================
        # UNEXPECTED TERMINATION
        # ======================================================

        yield {
            "type": "done",
            "result": {
                "answer": (
                    "The agent could not complete "
                    "the requested task."
                ),
                "sources": sources,
                "steps": steps_log,
                "intent": intent,
                "classification_reason": reason,
                "context": context,
                "plan": plan.get(
                    "steps",
                    [],
                ),
                "status": "failed",
                "replans": replans,
                "verification": verification,
            },
        }

    # ==========================================================
    # JSON HELPER
    # ==========================================================

    @staticmethod
    def _parse_json(
        raw: str,
    ) -> Dict[str, Any] | None:

        if not raw:
            return None

        raw = raw.strip()

        try:

            data = json.loads(
                raw
            )

            if isinstance(
                data,
                dict,
            ):
                return data

        except json.JSONDecodeError:
            pass

        # Handle markdown JSON.
        if "```" in raw:

            for part in raw.split(
                "```"
            ):

                part = part.strip()

                if part.startswith(
                    "json"
                ):
                    part = part[4:].strip()

                try:

                    data = json.loads(
                        part
                    )

                    if isinstance(
                        data,
                        dict,
                    ):
                        return data

                except json.JSONDecodeError:
                    continue

        # Extract JSON object.
        start = raw.find("{")
        end = raw.rfind("}")

        if (
            start != -1
            and end != -1
            and end > start
        ):

            try:

                data = json.loads(
                    raw[
                        start : end + 1
                    ]
                )

                if isinstance(
                    data,
                    dict,
                ):
                    return data

            except json.JSONDecodeError:
                pass

        return None