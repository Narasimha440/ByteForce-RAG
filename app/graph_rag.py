import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import requests

from .config import DATA_DIR, OLLAMA_GENERATE_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)

GRAPH_FILE_PATH = DATA_DIR / "knowledge_graph.graphml"


class GraphRAG:
    """
    Local Knowledge Graph Extractor and Retriever for SIH 26117.
    Extracts entity relationships from raw text using the local LLM.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self._load_graph()

    def _load_graph(self):
        if GRAPH_FILE_PATH.exists():
            try:
                self.graph = nx.read_graphml(str(GRAPH_FILE_PATH))
                logger.info(f"Loaded GraphRAG with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
            except Exception as e:
                logger.error(f"Failed to load graph: {e}")

    def save_graph(self):
        try:
            nx.write_graphml(self.graph, str(GRAPH_FILE_PATH))
            logger.info("GraphRAG saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")

    def extract_triples_from_text(self, text: str) -> List[Dict[str, str]]:
        """
        Uses the local LLM to extract (Subject, Relation, Object) triples from a text chunk.
        """
        prompt = (
            "You are an industrial knowledge extractor. Extract key entity relationships from the text below.\n"
            "Identify components (valves, pumps, pipes, limits, chemicals) and their relationships.\n"
            "Return the relationships STRICTLY as a JSON array of objects with 'subject', 'relation', and 'object' keys.\n"
            "Example:\n"
            "[{ \"subject\": \"PT-101\", \"relation\": \"measures_pressure_for\", \"object\": \"Main Air Blower\" }]\n\n"
            f"TEXT:\n{text}\n\n"
            "JSON OUTPUT:"
        )

        try:
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.0}
            }
            
            response = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            content = data.get("response", "").strip()
            
            if content:
                triples = json.loads(content)
                if isinstance(triples, list):
                    return triples
            return []

        except Exception as e:
            logger.warning(f"Failed to extract triples: {e}")
            return []

    def add_triples(self, triples: List[Dict[str, str]], source: str = "unknown"):
        """Add extracted triples to the NetworkX graph."""
        added = 0
        for t in triples:
            sub = t.get("subject", "").strip().upper()
            rel = t.get("relation", "").strip().upper()
            obj = t.get("object", "").strip().upper()
            
            if sub and rel and obj:
                self.graph.add_node(sub)
                self.graph.add_node(obj)
                self.graph.add_edge(sub, obj, relation=rel, source=source)
                added += 1
                
        if added > 0:
            self.save_graph()
            
    def get_subgraph_context(self, entities: List[str], depth: int = 1) -> str:
        """
        Given a list of entities (e.g., ['PT-101']), retrieves their 1-hop or 2-hop neighborhood
        and formats it as text to be injected into the RAG context.
        """
        context_lines = []
        visited = set()
        
        for entity in entities:
            entity_upper = entity.upper()
            if entity_upper in self.graph:
                # Get neighbors up to `depth`
                edges = nx.bfs_edges(self.graph, source=entity_upper, depth_limit=depth)
                for u, v in edges:
                    if (u, v) not in visited:
                        edge_data = self.graph.get_edge_data(u, v)
                        relation = edge_data.get("relation", "RELATED_TO")
                        context_lines.append(f"- {u} [{relation}] {v}")
                        visited.add((u, v))
                        
                # Also get in-edges
                in_edges = list(self.graph.in_edges(entity_upper, data=True))
                for u, v, data in in_edges:
                    if (u, v) not in visited:
                        relation = data.get("relation", "RELATED_TO")
                        context_lines.append(f"- {u} [{relation}] {v}")
                        visited.add((u, v))

        if context_lines:
            return "[KNOWLEDGE GRAPH CONTEXT]\n" + "\n".join(context_lines)
        return ""

# Global singleton
_graph_rag_instance = None

def get_graph_rag() -> GraphRAG:
    global _graph_rag_instance
    if _graph_rag_instance is None:
        _graph_rag_instance = GraphRAG()
    return _graph_rag_instance
