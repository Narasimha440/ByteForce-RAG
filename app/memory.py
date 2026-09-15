"""
Conversational Memory & Multi-Turn Query Rewriting for SIH 26117.

Enables multi-turn conversations where engineers can ask follow-up questions
without restating full context every time. Query condensation rewrites vague
follow-ups into self-contained retrieval queries using the local Ollama LLM.

Example:
    Turn 1: "What is the trip limit for PT-101?"
    Turn 2: "And what about FT-204?"
    → Rewritten: "What is the trip limit for FT-204?"   ← sent to RAG

100% local/air-gapped. No external APIs.
"""

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Lightweight session-scoped conversation history buffer.

    Stores the last N turns (question + answer + sources) so multi-turn
    follow-up questions can be resolved with full context.
    """

    def __init__(self, max_turns: int = 6):
        """
        Args:
            max_turns: Maximum number of conversation turns to retain.
                       Older turns are evicted in FIFO order.
        """
        self.max_turns = max_turns
        self._history: List[Dict] = []

    def add_turn(
        self,
        question: str,
        answer: str,
        sources: Optional[List[Dict]] = None,
    ) -> None:
        """Record a completed question-answer turn."""
        self._history.append(
            {
                "question": question.strip(),
                "answer": answer.strip()[:600],  # Truncate to save tokens
                "sources": [
                    s.get("filename", "") for s in (sources or []) if s.get("filename")
                ],
            }
        )
        # Evict oldest turns beyond the window
        if len(self._history) > self.max_turns:
            self._history = self._history[-self.max_turns :]

    def get_context_window(self) -> List[Dict]:
        """Return the current history window (read-only copy)."""
        return list(self._history)

    def get_summary_string(self, max_turns: int = 3) -> str:
        """
        Return a compact text summary of the last N turns for use in prompts.
        Format: Q: ... | A: ... (first 200 chars)
        """
        recent = self._history[-max_turns:]
        parts = []
        for turn in recent:
            q = turn["question"]
            a = turn["answer"][:200].replace("\n", " ")
            sources_str = ", ".join(turn["sources"][:2]) if turn["sources"] else ""
            line = f"Q: {q}\nA: {a}"
            if sources_str:
                line += f"\n(Sources: {sources_str})"
            parts.append(line)
        return "\n---\n".join(parts)

    def is_empty(self) -> bool:
        """Return True if no conversation history is stored."""
        return len(self._history) == 0

    def clear(self) -> None:
        """Reset conversation history."""
        self._history = []
        logger.info("Conversation memory cleared.")

    def __len__(self) -> int:
        return len(self._history)


def is_followup_question(question: str, history: List[Dict]) -> bool:
    """
    Heuristic check for whether a question is a follow-up that requires
    context from prior turns to be answerable on its own.

    Triggers on:
    - Pronouns: "it", "that", "they", "this", "those"
    - Relative phrases: "what about", "and for", "same for", "how about"
    - Missing subject: very short questions (< 5 words) with no equipment tag
    """
    if not history:
        return False

    q_lower = question.strip().lower()
    words = q_lower.split()

    # Short, vague question without a self-contained equipment tag
    has_tag = bool(
        re.search(
            r"\b(?:[A-Z]{2,6}-[A-Z0-9]{2,6}|PT|FT|LT|XV|FV|ESD)\b",
            question,
            re.IGNORECASE,
        )
    )

    followup_triggers = {
        "it", "that", "they", "this", "those", "its",
        "what about", "how about", "and for", "same for",
        "what is it", "what does it", "and what", "also",
    }

    if len(words) <= 4 and not has_tag:
        return True

    q_clean = re.sub(r"[^\w\s]", " ", q_lower)
    for trigger in followup_triggers:
        if trigger in q_clean:
            return True

    return False


def condense_followup_query(
    history: List[Dict],
    new_question: str,
    ollama_url: str = "http://localhost:11434/api/chat",
    model: str = "qwen2.5:1.5b",
    timeout: int = 20,
) -> str:
    """
    Rewrite a follow-up question into a fully self-contained retrieval query
    using the local Ollama LLM and conversation history as context.

    Returns the rewritten query (or original question if rewriting fails).

    Example:
        history = [{"question": "What is the trip limit for PT-101?", ...}]
        new_question = "And what about FT-204?"
        → Returns: "What is the trip limit for FT-204?"
    """
    if not history or not new_question.strip():
        return new_question

    history_text = ""
    for turn in history[-3:]:  # Use last 3 turns for condensation context
        history_text += f"Human: {turn['question']}\nAssistant: {turn['answer'][:300]}\n\n"

    condensation_prompt = (
        f"Given this conversation history:\n\n{history_text}\n"
        f"Rewrite this follow-up question as a single, fully self-contained question "
        f"that can be answered without seeing the conversation history. "
        f"Preserve all technical terms, equipment tags, and numbers exactly. "
        f"Only output the rewritten question, nothing else.\n\n"
        f"Follow-up: {new_question}\n"
        f"Rewritten question:"
    )

    try:
        import requests

        response = requests.post(
            ollama_url,
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a query rewriter for an industrial plant knowledge base. "
                            "Rewrite follow-up questions to be self-contained. Be precise and concise."
                        ),
                    },
                    {"role": "user", "content": condensation_prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=timeout,
        )
        response.raise_for_status()
        rewritten = response.json()["message"]["content"].strip()

        # Safety: reject if rewritten is too long or empty (use original)
        if rewritten and len(rewritten) < 300:
            logger.info(f"Query condensed: '{new_question}' → '{rewritten}'")
            return rewritten

    except Exception as exc:
        logger.debug(f"Query condensation failed: {exc}. Using original question.")

    return new_question
