import json
import logging
import re
from typing import Dict, List, Optional, Any

import requests

from .config import OLLAMA_URL, OLLAMA_MODEL
from .domain_expansion import extract_query_equipment_tags
from .graph_rag import get_graph_rag
from .rag import LocalRAG
from .vision import analyze_engineering_drawing

logger = logging.getLogger(__name__)


class IndustrialAgent:
    """
    Agentic Multi-Step Reasoning (ReAct) Engine for SIH 26117.
    """
    
    def __init__(self):
        self.rag = LocalRAG()
        self.graph = get_graph_rag()
        self.max_steps = 5
        
    def _critique_context(self, question: str, context: str) -> bool:
        """Self-RAG: Critique if the context actually answers the question."""
        prompt = f"Does the following context contain enough information to answer the question? Answer YES or NO.\nQuestion: {question}\nContext: {context[:2000]}"
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0}
        }
        try:
            res = requests.post(OLLAMA_URL.replace("/chat", "/generate"), json=payload, timeout=30)
            text = res.json().get("response", "").strip().upper()
            return "YES" in text
        except Exception:
            return True  # Fallback to accepting it if Ollama fails

    def _run_tool(self, tool_name: str, tool_input: str, original_question: str = "") -> str:
        try:
            if tool_name == "search_documents":
                hits = self.rag.retrieve(tool_input, top_k=4)
                context, _ = self.rag.build_context(hits, question=tool_input)
                
                if context.strip():
                    if original_question and not self._critique_context(original_question, context):
                        return f"Retrieved documents for '{tool_input}', but they do not seem to answer the main question. Try rewriting the search query to be more specific or use different keywords."
                    return context
                return "No relevant documents found."
                
            elif tool_name == "query_graph":
                tags = extract_query_equipment_tags(tool_input)
                if not tags:
                    # try split
                    tags = [t.strip().upper() for t in tool_input.split(",")]
                subgraph = self.graph.get_subgraph_context(tags, depth=2)
                return subgraph if subgraph.strip() else f"No knowledge graph entities found for {tags}."
                
            elif tool_name == "analyze_image":
                desc = analyze_engineering_drawing(tool_input.strip())
                return desc if desc.strip() else "Failed to analyze image or empty description returned."
                
            else:
                return f"Error: Unknown tool {tool_name}"
        except Exception as e:
            return f"Error executing {tool_name}: {e}"

    def run(self, question: str) -> str:
        """Execute a ReAct loop to answer the question."""
        prompt = f"""You are an Industrial Agentic AI Assistant for a confidential refinery (MRPL).
Solve the user's problem by reasoning step-by-step. You have access to the following tools:

- search_documents: Searches the local RAG database for SOPs, manuals, and documents. Input should be a search query.
- query_graph: Queries the local Knowledge Graph for equipment relationships. Input should be a comma-separated list of equipment tags (e.g., PT-101, XV-301).
- analyze_image: Analyzes an engineering drawing or photo (e.g., P&ID). Input should be the absolute file path to the image.

Use the following format strictly:

Thought: you should always think about what to do next
Action: the action to take, should be one of [search_documents, query_graph, analyze_image]
Action Input: the input to the action
Observation: the result of the action (provided by the system)
... (this Thought/Action/Action Input/Observation can repeat multiple times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

USER QUESTION: {question}
"""
        messages = [{"role": "system", "content": "You are a ReAct agent."}, {"role": "user", "content": prompt}]
        
        for step in range(self.max_steps):
            payload = {
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.1, "stop": ["Observation:"]}
            }
            
            try:
                response = requests.post(OLLAMA_URL, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()["message"]["content"]
                
                messages.append({"role": "assistant", "content": result})
                
                if "Final Answer:" in result:
                    return result.split("Final Answer:")[-1].strip()
                    
                # Parse Action and Action Input
                action_match = re.search(r"Action:\s*(.*?)\n", result)
                input_match = re.search(r"Action Input:\s*(.*)", result)
                
                if action_match and input_match:
                    action = action_match.group(1).strip()
                    action_input = input_match.group(1).strip()
                    
                    logger.info(f"Agent Step {step+1}: Tool={action}, Input={action_input}")
                    observation = self._run_tool(action, action_input, original_question=question)
                    
                    messages.append({"role": "user", "content": f"Observation: {observation}\nThought:"})
                else:
                    return f"Agent stopped following formatting rules. Last output: {result}"
                    
            except Exception as e:
                logger.error(f"Agent loop failed: {e}")
                return f"Agent error: {e}"
                
        return "Agent reached maximum steps without a Final Answer."
