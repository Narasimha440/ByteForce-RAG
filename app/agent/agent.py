import json
import requests

from app.rag import LocalRAG
from app.config import OLLAMA_URL, OLLAMA_MODEL


class Agent:

    def __init__(self):
        self.rag = LocalRAG()

    def call_llm(self, prompt, temperature=0.1):
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are the reasoning agent for a "
                            "private, on-premise industrial AI workbench."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def call_llm_stream(self, prompt, temperature=0.1):
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are the reasoning agent for a "
                            "private, on-premise industrial AI workbench."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": True,
                "options": {"temperature": temperature},
            },
            stream=True,
            timeout=300,
        )

        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue

            data = json.loads(line)

            if "message" in data:
                token = data["message"].get("content", "")
                if token:
                    yield token

            if data.get("done", False):
                break

    def classify_task(self, question):
        prompt = f"""
Classify the user's request into exactly ONE of these intents:

rag
general

Use "rag" when the question asks about:
- company information
- industrial procedures
- SOPs
- inspection requirements
- safety requirements
- internal documents
- organization-specific information
- information that should come from uploaded documents

Use "general" when the question can be answered without
company-specific documents or private knowledge.

Return ONLY valid JSON.

Format:
{{
    "intent": "rag",
    "reason": "short explanation"
}}

USER REQUEST:
{question}
"""

        raw = self.call_llm(prompt)

        try:
            result = json.loads(raw)
            intent = result.get("intent", "rag").lower()

            if intent not in ["rag", "general"]:
                intent = "rag"

            return {
                "intent": intent,
                "reason": result.get("reason", ""),
            }

        except (json.JSONDecodeError, TypeError):
            return {
                "intent": "rag",
                "reason": "Classification failed; using secure RAG fallback.",
            }

    def build_rag_prompt(self, question, context):
        return f"""
You are operating inside a private industrial AI system.

Answer the user's question using ONLY the retrieved
local knowledge-base evidence.

Rules:
- Do not use outside knowledge.
- Do not invent facts.
- Do not invent procedures.
- Do not invent safety requirements.
- Do not invent measurements or engineering values.
- If the evidence is insufficient, clearly state that
  the local knowledge base does not contain enough information.

USER QUESTION:
{question}

RETRIEVED LOCAL EVIDENCE:
{context}

Provide a concise and professional answer.
"""

    def build_general_prompt(self, question):
        return f"""
Answer the user's question directly.

This request does not require information
from the private industrial knowledge base.

Do not pretend that information comes from
company documents.

USER QUESTION:
{question}

Provide a clear and professional answer.
"""

    def run_stream(self, question):
        """
        Generator that exposes observable agent execution events.

        Event format:
        {
            "type": "step" | "answer_start" | "token" | "sources" | "done",
            "message": ...,
            ...
        }

        This intentionally exposes execution events, not private
        model chain-of-thought.
        """

        if not question or not question.strip():
            yield {
                "type": "step",
                "message": "Please provide a question or task.",
            }
            yield {
                "type": "done",
                "result": {
                    "answer": "Please provide a question or task.",
                    "sources": [],
                    "steps": ["Received empty user request"],
                    "intent": "unknown",
                },
            }
            return

        steps = []

        steps.append("Received user request")
        yield {"type": "step", "message": "Received user request"}

        yield {"type": "step", "message": "Classifying task..."}

        classification = self.classify_task(question)
        intent = classification["intent"]
        reason = classification["reason"]

        steps.append(f"Classified request as {intent.upper()}")

        yield {
            "type": "classification",
            "intent": intent,
            "reason": reason,
        }

        yield {
            "type": "step",
            "message": f"Intent detected: {intent.upper()}",
        }

        if intent == "rag":
            yield {
                "type": "step",
                "message": "Knowledge-based request detected",
            }

            yield {
                "type": "step",
                "message": "Searching local knowledge base...",
            }

            rag_result = self.rag.search(question)

            if not rag_result["found"]:
                steps.append("No relevant evidence found")

                yield {
                    "type": "step",
                    "message": "No relevant local evidence found",
                }

                result = {
                    "answer": (
                        "The local knowledge base does not contain "
                        "enough information to answer this."
                    ),
                    "sources": [],
                    "steps": steps,
                    "intent": intent,
                    "classification_reason": reason,
                    "context": "",
                }

                yield {"type": "done", "result": result}
                return

            context = rag_result["context"]
            sources = rag_result["sources"]

            steps.append(
                f"Retrieved {len(sources)} relevant evidence sources"
            )

            yield {
                "type": "retrieval",
                "sources": sources,
                "context": context,
            }

            yield {
                "type": "step",
                "message": f"Retrieved {len(sources)} relevant evidence sources",
            }

            yield {
                "type": "step",
                "message": "Evidence assembled for reasoning model",
            }

            prompt = self.build_rag_prompt(question, context)

            yield {
                "type": "step",
                "message": "Sending grounded evidence to Qwen3...",
            }

        else:
            steps.append("Skipped knowledge-base retrieval")

            yield {
                "type": "step",
                "message": "General request detected",
            }

            yield {
                "type": "step",
                "message": "RAG not required — skipping knowledge base",
            }

            prompt = self.build_general_prompt(question)

        steps.append("Generating response with Qwen3")

        yield {
            "type": "step",
            "message": "Generating response with Qwen3...",
        }

        yield {"type": "answer_start"}

        answer_parts = []

        for token in self.call_llm_stream(prompt):
            answer_parts.append(token)
            yield {
                "type": "token",
                "content": token,
            }

        answer = "".join(answer_parts)

        steps.append("Response generation completed")

        result = {
            "answer": answer,
            "sources": sources if intent == "rag" else [],
            "steps": steps,
            "intent": intent,
            "classification_reason": reason,
            "context": context if intent == "rag" else "",
        }

        yield {"type": "done", "result": result}

    def run(self, question):
        """
        Backwards-compatible non-streaming interface.
        """
        result = None

        for event in self.run_stream(question):
            if event["type"] == "done":
                result = event["result"]

        return result
