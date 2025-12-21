"""
Knowledge Graph - Shared Context and Memory for the Agentic System

This module implements a knowledge graph that serves as:
- Shared memory between agents
- Context storage for paper and code understanding
- Relationship mapping between concepts, code, and documentation
- Query interface for intelligent information retrieval
"""

import json
import hashlib
from datetime import datetime
from typing import Optional, Any, Dict, List, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import asyncio
from collections import defaultdict

try:
    import networkx as nx
except ImportError:
    nx = None


class NodeType(Enum):
    """Types of nodes in the knowledge graph."""
    # Paper-related
    PAPER = "paper"
    SECTION = "section"
    CONCEPT = "concept"
    EQUATION = "equation"
    FIGURE = "figure"
    TABLE = "table"
    CITATION = "citation"
    AUTHOR = "author"
    ALGORITHM = "algorithm"
    
    # Code-related
    REPOSITORY = "repository"
    FILE = "file"
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    VARIABLE = "variable"
    DEPENDENCY = "dependency"
    
    # Execution-related
    ENVIRONMENT = "environment"
    TEST = "test"
    RESULT = "result"
    VISUALIZATION = "visualization"
    ERROR = "error"
    
    # Meta
    AGENT = "agent"
    ACTION = "action"
    INSIGHT = "insight"


class EdgeType(Enum):
    """Types of relationships in the knowledge graph."""
    # Structural
    CONTAINS = "contains"
    PART_OF = "part_of"
    REFERENCES = "references"
    DEPENDS_ON = "depends_on"
    IMPORTS = "imports"
    EXTENDS = "extends"
    IMPLEMENTS = "implements"
    
    # Semantic
    RELATED_TO = "related_to"
    SIMILAR_TO = "similar_to"
    EXPLAINS = "explains"
    DEMONSTRATES = "demonstrates"
    VALIDATES = "validates"
    CONTRADICTS = "contradicts"
    
    # Temporal
    FOLLOWS = "follows"
    PRECEDES = "precedes"
    TRIGGERS = "triggers"
    
    # Agent actions
    CREATED_BY = "created_by"
    ANALYZED_BY = "analyzed_by"
    MODIFIED_BY = "modified_by"
    USES = "uses"
    TESTS = "tests"
    GENERATES = "generates"


@dataclass
class KnowledgeNode:
    """A node in the knowledge graph."""
    id: str
    node_type: NodeType
    name: str
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[List[float]] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = ""
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "node_type": self.node_type.value,
            "name": self.name,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source": self.source,
            "confidence": self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeNode":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            node_type=NodeType(data["node_type"]),
            name=data["name"],
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            source=data.get("source", ""),
            confidence=data.get("confidence", 1.0)
        )


@dataclass
class KnowledgeEdge:
    """An edge in the knowledge graph."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    created_by: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "created_by": self.created_by
        }


class KnowledgeGraph:
    """
    Shared knowledge graph for the agentic system.
    
    Provides:
    - Multi-agent memory and context sharing
    - Semantic relationship mapping
    - Query and retrieval capabilities
    - Persistence and serialization
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        if nx:
            self.graph = nx.MultiDiGraph()
        else:
            self.graph = None
        self._nodes: Dict[str, KnowledgeNode] = {}
        self._edges: List[KnowledgeEdge] = []
        self._index: Dict[NodeType, Set[str]] = defaultdict(set)
        self._content_index: Dict[str, Set[str]] = defaultdict(set)
        self._lock = asyncio.Lock()
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
    
    def _generate_id(self, node_type: NodeType, name: str, content: str = "") -> str:
        """Generate a unique ID for a node."""
        hash_input = f"{node_type.value}:{name}:{content[:100]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    async def add_node(
        self,
        node_type: NodeType,
        name: str,
        content: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "",
        node_id: Optional[str] = None
    ) -> str:
        """Add a node to the knowledge graph."""
        async with self._lock:
            if node_id is None:
                node_id = self._generate_id(node_type, name, content)
            
            if node_id in self._nodes:
                # Update existing node
                node = self._nodes[node_id]
                node.content = content or node.content
                node.metadata.update(metadata or {})
                node.updated_at = datetime.now().isoformat()
            else:
                # Create new node
                node = KnowledgeNode(
                    id=node_id,
                    node_type=node_type,
                    name=name,
                    content=content,
                    metadata=metadata or {},
                    source=source
                )
                self._nodes[node_id] = node
                self._index[node_type].add(node_id)
                
                # Add to networkx graph
                if self.graph is not None:
                    self.graph.add_node(node_id, **node.to_dict())
                
                # Update content index
                for word in name.lower().split() + content.lower().split()[:50]:
                    if len(word) > 2:
                        self._content_index[word].add(node_id)
            
            self.updated_at = datetime.now().isoformat()
            return node_id
    
    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        created_by: str = ""
    ) -> bool:
        """Add an edge between nodes."""
        async with self._lock:
            if source_id not in self._nodes or target_id not in self._nodes:
                return False
            
            edge = KnowledgeEdge(
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                weight=weight,
                metadata=metadata or {},
                created_by=created_by
            )
            
            self._edges.append(edge)
            
            if self.graph is not None:
                self.graph.add_edge(
                    source_id,
                    target_id,
                    key=edge_type.value,
                    **edge.to_dict()
                )
            
            self.updated_at = datetime.now().isoformat()
            return True
    
    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[KnowledgeNode]:
        """Get all nodes of a specific type."""
        return [self._nodes[nid] for nid in self._index.get(node_type, set())]
    
    def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[EdgeType] = None,
        direction: str = "both"
    ) -> List[Tuple[str, KnowledgeNode, EdgeType]]:
        """Get neighboring nodes."""
        if node_id not in self._nodes:
            return []
        
        neighbors = []
        
        if self.graph is not None:
            if direction in ("out", "both"):
                for _, target, key, data in self.graph.out_edges(node_id, keys=True, data=True):
                    et = EdgeType(data.get("edge_type", key))
                    if edge_type is None or et == edge_type:
                        if target in self._nodes:
                            neighbors.append((key, self._nodes[target], et))
            
            if direction in ("in", "both"):
                for source, _, key, data in self.graph.in_edges(node_id, keys=True, data=True):
                    et = EdgeType(data.get("edge_type", key))
                    if edge_type is None or et == edge_type:
                        if source in self._nodes:
                            neighbors.append((key, self._nodes[source], et))
        
        return neighbors
    
    def search(
        self,
        query: str,
        node_types: Optional[List[NodeType]] = None,
        limit: int = 10
    ) -> List[Tuple[KnowledgeNode, float]]:
        """Search for nodes matching a query."""
        query_words = set(query.lower().split())
        candidates: Dict[str, float] = defaultdict(float)
        
        for word in query_words:
            for node_id in self._content_index.get(word, set()):
                if node_types is None or self._nodes[node_id].node_type in node_types:
                    candidates[node_id] += 1.0
        
        # Normalize scores
        if query_words:
            for node_id in candidates:
                candidates[node_id] /= len(query_words)
        
        # Sort and return
        sorted_results = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return [(self._nodes[nid], score) for nid, score in sorted_results[:limit]]
    
    def find_path(
        self,
        source_id: str,
        target_id: str
    ) -> Optional[List[str]]:
        """Find shortest path between nodes."""
        if self.graph is None:
            return None
        
        try:
            return nx.shortest_path(self.graph, source_id, target_id)
        except (nx.NetworkXNoPath, nx.NetworkXError):
            return None
    
    def get_subgraph(
        self,
        center_node_id: str,
        depth: int = 2,
        edge_types: Optional[List[EdgeType]] = None
    ) -> "KnowledgeGraph":
        """Extract a subgraph centered on a node."""
        if center_node_id not in self._nodes:
            return KnowledgeGraph(f"{self.name}_subgraph")
        
        visited = {center_node_id}
        frontier = {center_node_id}
        
        for _ in range(depth):
            new_frontier = set()
            for node_id in frontier:
                for _, neighbor, et in self.get_neighbors(node_id):
                    if edge_types is None or et in edge_types:
                        if neighbor.id not in visited:
                            visited.add(neighbor.id)
                            new_frontier.add(neighbor.id)
            frontier = new_frontier
        
        subgraph = KnowledgeGraph(f"{self.name}_subgraph")
        subgraph._nodes = {nid: self._nodes[nid] for nid in visited}
        if self.graph is not None:
            subgraph.graph = self.graph.subgraph(visited).copy()
        
        return subgraph
    
    def get_paper_code_connections(self) -> List[Dict[str, Any]]:
        """Get all connections between paper concepts and code elements."""
        connections = []
        paper_types = {NodeType.CONCEPT, NodeType.ALGORITHM, NodeType.EQUATION}
        code_types = {NodeType.CLASS, NodeType.FUNCTION, NodeType.MODULE}
        
        for edge in self._edges:
            source = self._nodes.get(edge.source_id)
            target = self._nodes.get(edge.target_id)
            
            if source and target:
                if (source.node_type in paper_types and target.node_type in code_types) or \
                   (source.node_type in code_types and target.node_type in paper_types):
                    connections.append({
                        "paper_element": source.to_dict() if source.node_type in paper_types else target.to_dict(),
                        "code_element": target.to_dict() if target.node_type in code_types else source.to_dict(),
                        "relationship": edge.edge_type.value,
                        "weight": edge.weight,
                        "metadata": edge.metadata
                    })
        
        return connections
    
    def to_dict(self) -> Dict[str, Any]:
        """Export graph to dictionary."""
        return {
            "name": self.name,
            "nodes": [node.to_dict() for node in self._nodes.values()],
            "edges": [edge.to_dict() for edge in self._edges],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "stats": {
                "node_count": len(self._nodes),
                "edge_count": len(self._edges),
                "node_types": {nt.value: len(self._index[nt]) for nt in NodeType if self._index[nt]}
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        """Create graph from dictionary."""
        graph = cls(data.get("name", "imported"))
        
        for node_data in data.get("nodes", []):
            node = KnowledgeNode.from_dict(node_data)
            graph._nodes[node.id] = node
            graph._index[node.node_type].add(node.id)
            if graph.graph is not None:
                graph.graph.add_node(node.id, **node.to_dict())
        
        for edge_data in data.get("edges", []):
            edge = KnowledgeEdge(
                source_id=edge_data["source_id"],
                target_id=edge_data["target_id"],
                edge_type=EdgeType(edge_data["edge_type"]),
                weight=edge_data.get("weight", 1.0),
                metadata=edge_data.get("metadata", {}),
                created_by=edge_data.get("created_by", "")
            )
            graph._edges.append(edge)
            if graph.graph is not None:
                graph.graph.add_edge(
                    edge.source_id,
                    edge.target_id,
                    key=edge.edge_type.value,
                    **edge.to_dict()
                )
        
        graph.created_at = data.get("created_at", datetime.now().isoformat())
        graph.updated_at = data.get("updated_at", datetime.now().isoformat())
        
        return graph
    
    def save(self, filepath: str):
        """Save graph to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "KnowledgeGraph":
        """Load graph from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def clear(self):
        """Clear all nodes and edges."""
        self._nodes.clear()
        self._edges.clear()
        self._index.clear()
        self._content_index.clear()
        if self.graph is not None:
            self.graph.clear()
        self.updated_at = datetime.now().isoformat()


# Global instance management
_global_graph: Optional[KnowledgeGraph] = None


def get_global_graph() -> KnowledgeGraph:
    """Get or create the global knowledge graph instance."""
    global _global_graph
    if _global_graph is None:
        _global_graph = KnowledgeGraph(name="global")
    return _global_graph


def set_global_graph(graph: KnowledgeGraph):
    """Set the global knowledge graph instance."""
    global _global_graph
    _global_graph = graph


def reset_global_graph():
    """Reset the global knowledge graph."""
    global _global_graph
    _global_graph = None