import json
from typing import Dict, Any


class Planner:

    ALLOWED_TOOLS = {
        "rag_search",
        "generate_response",
    }

    MAX_STEPS = 5

    def __init__(self, llm_call):
        self.llm_call = llm_call

    # ==========================================================
    # FALLBACK PLAN
    # ==========================================================

    def _fallback_plan(
        self,
        question: str,
        intent: str,
    ) -> Dict[str, Any]:

        if intent == "rag":
            return {
                "goal": "Retrieve local evidence and answer the question.",
                "complexity": "single_path",
                "steps": [
                    {
                        "id": 1,
                        "tool": "rag_search",
                        "action": "search_knowledge_base",
                        "params": {
                            "query": question
                        },
                        "description": (
                            "Search local industrial documents "
                            "for relevant evidence."
                        ),
                    },
                    {
                        "id": 2,
                        "tool": "generate_response",
                        "action": "synthesize_grounded_answer",
                        "params": {},
                        "description": (
                            "Generate an answer using only "
                            "verified local evidence."
                        ),
                    },
                ],
                "final_output": "grounded_answer",
            }

        return {
            "goal": "Answer the user request directly.",
            "complexity": "single_path",
            "steps": [
                {
                    "id": 1,
                    "tool": "generate_response",
                    "action": "direct_answer",
                    "params": {},
                    "description": (
                        "Generate a direct response "
                        "without local retrieval."
                    ),
                }
            ],
            "final_output": "answer",
        }

    # ==========================================================
    # INITIAL PLAN
    # ==========================================================

    def create_plan(
        self,
        question: str,
        intent: str,
    ) -> Dict[str, Any]:

        # Deterministic planning for the current supported tools.
        # This avoids unnecessary planning latency.
        return self._fallback_plan(
            question,
            intent,
        )

    # ==========================================================
    # DYNAMIC REPLAN
    # ==========================================================

    def replan(
        self,
        question: str,
        intent: str,
        previous_plan: Dict[str, Any],
        feedback: Dict[str, Any],
    ) -> Dict[str, Any]:

        if intent != "rag":
            return self._fallback_plan(
                question,
                intent,
            )

        reason = str(
            feedback.get(
                "reason",
                "Retrieved evidence was insufficient.",
            )
        )

        missing_points = feedback.get(
            "missing_points",
            [],
        )

        if not isinstance(
            missing_points,
            list,
        ):
            missing_points = []

        previous_query = question

        previous_steps = previous_plan.get(
            "steps",
            [],
        )

        for step in previous_steps:
            if step.get("tool") == "rag_search":
                params = step.get(
                    "params",
                    {},
                )

                previous_query = params.get(
                    "query",
                    question,
                )

                break

        prompt = f"""
You are the replanning component of a private,
on-premise industrial AI agent.

The initial retrieval attempt did not provide sufficient evidence.

Your job is to create a NEW retrieval strategy.

Do NOT answer the user's question.

Do NOT use internet or external knowledge.

Available tools:

- rag_search
  Params:
  {{
      "query": "search query"
  }}

- generate_response
  Params:
  {{}}

The new plan MUST attempt a meaningfully different retrieval query.

Do not simply repeat the previous query.

Use the verifier feedback to identify what information
the next retrieval should target.

USER QUESTION:
{question}

PREVIOUS SEARCH QUERY:
{previous_query}

VERIFIER REASON:
{reason}

MISSING INFORMATION:
{json.dumps(missing_points)}

Return ONLY valid JSON:

{{
  "goal": "Retrieve better local evidence for the request.",
  "complexity": "single_path",
  "steps": [
    {{
      "id": 1,
      "tool": "rag_search",
      "action": "refined_knowledge_search",
      "params": {{
        "query": "new and more precise search query"
      }},
      "description": "Search local documents using a refined query."
    }},
    {{
      "id": 2,
      "tool": "generate_response",
      "action": "synthesize_grounded_answer",
      "params": {{}},
      "description": "Generate an answer using verified local evidence."
    }}
  ],
  "final_output": "grounded_answer"
}}
"""

        try:
            raw = self.llm_call(
                prompt,
                temperature=0.0,
            )

            data = self._parse_json(raw)

            validated = self.validate_plan(
                data,
                intent,
            )

            if validated:
                # Make sure the new plan isn't simply repeating
                # the exact previous search query.
                new_query = self._get_rag_query(
                    validated,
                )

                if (
                    new_query
                    and new_query.strip().lower()
                    != previous_query.strip().lower()
                ):
                    return validated

        except Exception:
            pass

        # Safe deterministic fallback if the planner fails.
        fallback_query = self._build_fallback_requery(
            question,
            previous_query,
            missing_points,
        )

        return {
            "goal": (
                "Retry local retrieval using a refined "
                "query strategy."
            ),
            "complexity": "single_path",
            "steps": [
                {
                    "id": 1,
                    "tool": "rag_search",
                    "action": "refined_knowledge_search",
                    "params": {
                        "query": fallback_query
                    },
                    "description": (
                        "Retry the local knowledge search "
                        "with a refined query."
                    ),
                },
                {
                    "id": 2,
                    "tool": "generate_response",
                    "action": "synthesize_grounded_answer",
                    "params": {},
                    "description": (
                        "Generate an answer only if "
                        "the new evidence is sufficient."
                    ),
                },
            ],
            "final_output": "grounded_answer",
        }

    # ==========================================================
    # PLAN VALIDATION
    # ==========================================================

    def validate_plan(
        self,
        plan: Any,
        intent: str,
    ) -> Dict[str, Any] | None:

        if not isinstance(
            plan,
            dict,
        ):
            return None

        steps = plan.get(
            "steps",
            [],
        )

        if not (
            isinstance(steps, list)
            and 1 <= len(steps) <= self.MAX_STEPS
        ):
            return None

        clean_steps = []

        for index, step in enumerate(
            steps,
            start=1,
        ):

            if not isinstance(
                step,
                dict,
            ):
                return None

            tool = str(
                step.get(
                    "tool",
                    "",
                )
            ).strip().lower()

            if tool not in self.ALLOWED_TOOLS:
                return None

            params = step.get(
                "params",
                {},
            )

            if not isinstance(
                params,
                dict,
            ):
                params = {}

            # Validate RAG parameters.
            if tool == "rag_search":
                query = str(
                    params.get(
                        "query",
                        "",
                    )
                ).strip()

                if not query:
                    return None

                params = {
                    "query": query
                }

            clean_steps.append(
                {
                    "id": index,
                    "tool": tool,
                    "action": str(
                        step.get(
                            "action",
                            "",
                        )
                    ).strip(),
                    "params": params,
                    "description": str(
                        step.get(
                            "description",
                            "",
                        )
                    ).strip(),
                }
            )

        # A valid RAG plan must eventually generate a response.
        if intent == "rag":

            has_rag = any(
                step["tool"] == "rag_search"
                for step in clean_steps
            )

            has_response = any(
                step["tool"] == "generate_response"
                for step in clean_steps
            )

            if not has_rag or not has_response:
                return None

        return {
            "goal": str(
                plan.get(
                    "goal",
                    "Execute request",
                )
            ),
            "complexity": str(
                plan.get(
                    "complexity",
                    "single_path",
                )
            ),
            "steps": clean_steps,
            "final_output": str(
                plan.get(
                    "final_output",
                    "answer",
                )
            ),
        }

    # ==========================================================
    # HELPERS
    # ==========================================================

    @staticmethod
    def _get_rag_query(
        plan: Dict[str, Any],
    ) -> str:

        for step in plan.get(
            "steps",
            [],
        ):

            if step.get("tool") == "rag_search":

                return str(
                    step.get(
                        "params",
                        {},
                    ).get(
                        "query",
                        "",
                    )
                ).strip()

        return ""

    @staticmethod
    def _build_fallback_requery(
        question: str,
        previous_query: str,
        missing_points: list,
    ) -> str:

        missing_text = " ".join(
            str(item)
            for item in missing_points
            if str(item).strip()
        )

        if missing_text:
            return (
                f"{question} "
                f"Focus specifically on: {missing_text}"
            )

        return (
            f"{question} "
            "Search for specific procedures, "
            "requirements, limits, responsibilities, "
            "and operational details."
        )

    @staticmethod
    def _parse_json(
        raw: str,
    ) -> Dict[str, Any] | None:

        if not raw:
            return None

        raw = raw.strip()

        try:
            data = json.loads(raw)

            if isinstance(data, dict):
                return data

        except json.JSONDecodeError:
            pass

        if "```" in raw:

            parts = raw.split("```")

            for part in parts:

                part = part.strip()

                if part.startswith("json"):
                    part = part[4:].strip()

                try:
                    data = json.loads(part)

                    if isinstance(data, dict):
                        return data

                except json.JSONDecodeError:
                    continue

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

                if isinstance(data, dict):
                    return data

            except json.JSONDecodeError:
                pass

        return None