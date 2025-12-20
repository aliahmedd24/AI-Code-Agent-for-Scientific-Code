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
import networkx as nx
from collections import defaultdict
import asyncio


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
    source: str = ""  # paper, repo, agent, etc.
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
    created_by: str = ""  # Agent that created this edge
    
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
        self.graph = nx.MultiDiGraph()
        self._nodes: Dict[str, KnowledgeNode] = {}
        self._edges: List[KnowledgeEdge] = []
        self._index: Dict[NodeType, Set[str]] = defaultdict(set)
        self._content_index: Dict[str, Set[str]] = defaultdict(set)  # word -> node_ids
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
        """
        Add a node to the knowledge graph.
        
        Args:
            node_type: Type of the node
            name: Name/title of the node
            content: Full content/description
            metadata: Additional metadata
            source: Source of this knowledge
            node_id: Optional custom ID
        
        Returns:
            The node ID
        """
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
                self.graph.add_node(node_id, **node.to_dict())
                self._index[node_type].add(node_id)
                
                # Index content for search
                words = set(name.lower().split() + content.lower().split()[:100])
                for word in words:
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
        """
        Add an edge between two nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of relationship
            weight: Edge weight (importance)
            metadata: Additional metadata
            created_by: Agent that created this edge
        
        Returns:
            True if edge was added, False if nodes don't exist
        """
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
        """
        Get neighboring nodes.
        
        Args:
            node_id: The node to get neighbors for
            edge_type: Filter by edge type
            direction: 'in', 'out', or 'both'
        
        Returns:
            List of (edge_key, neighbor_node, edge_type) tuples
        """
        if node_id not in self._nodes:
            return []
        
        neighbors = []
        
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
        """
        Search for nodes matching a query.
        
        Args:
            query: Search query
            node_types: Filter by node types
            limit: Maximum results
        
        Returns:
            List of (node, relevance_score) tuples
        """
        query_words = set(query.lower().split())
        scores: Dict[str, float] = defaultdict(float)
        
        for word in query_words:
            # Exact matches
            if word in self._content_index:
                for node_id in self._content_index[word]:
                    scores[node_id] += 2.0
            
            # Partial matches
            for indexed_word, node_ids in self._content_index.items():
                if word in indexed_word or indexed_word in word:
                    for node_id in node_ids:
                        scores[node_id] += 1.0
        
        # Filter by type and sort by score
        results = []
        for node_id, score in sorted(scores.items(), key=lambda x: -x[1]):
            node = self._nodes.get(node_id)
            if node:
                if node_types is None or node.node_type in node_types:
                    results.append((node, score))
                    if len(results) >= limit:
                        break
        
        return results
    
    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5
    ) -> Optional[List[str]]:
        """Find the shortest path between two nodes."""
        try:
            return nx.shortest_path(self.graph, source_id, target_id)
        except nx.NetworkXNoPath:
            return None
    
    def get_subgraph(
        self,
        center_node_id: str,
        depth: int = 2,
        edge_types: Optional[List[EdgeType]] = None
    ) -> "KnowledgeGraph":
        """
        Extract a subgraph centered on a node.
        
        Args:
            center_node_id: The center node
            depth: How many hops to include
            edge_types: Filter by edge types
        
        Returns:
            A new KnowledgeGraph with the subgraph
        """
        if center_node_id not in self._nodes:
            return KnowledgeGraph(f"{self.name}_subgraph")
        
        # BFS to find nodes within depth
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
        
        # Create subgraph
        subgraph = KnowledgeGraph(f"{self.name}_subgraph")
        subgraph._nodes = {nid: self._nodes[nid] for nid in visited}
        subgraph.graph = self.graph.subgraph(visited).copy()
        
        return subgraph
    
    def get_paper_code_connections(self) -> List[Dict[str, Any]]:
        """
        Get all connections between paper concepts and code elements.
        
        Returns:
            List of connection dictionaries
        """
        connections = []
        
        paper_types = {NodeType.PAPER, NodeType.SECTION, NodeType.CONCEPT, NodeType.EQUATION}
        code_types = {NodeType.REPOSITORY, NodeType.FILE, NodeType.MODULE, NodeType.CLASS, NodeType.FUNCTION}
        
        for edge in self._edges:
            source = self._nodes.get(edge.source_id)
            target = self._nodes.get(edge.target_id)
            
            if source and target:
                # Check if this connects paper to code
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
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        type_counts = defaultdict(int)
        for node in self._nodes.values():
            type_counts[node.node_type.value] += 1
        
        edge_type_counts = defaultdict(int)
        for edge in self._edges:
            edge_type_counts[edge.edge_type.value] += 1
        
        return {
            "name": self.name,
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "node_types": dict(type_counts),
            "edge_types": dict(edge_type_counts),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "density": nx.density(self.graph) if len(self._nodes) > 0 else 0,
            "is_connected": nx.is_weakly_connected(self.graph) if len(self._nodes) > 0 else True
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the graph to a dictionary."""
        return {
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "nodes": [node.to_dict() for node in self._nodes.values()],
            "edges": [edge.to_dict() for edge in self._edges],
            "statistics": self.get_statistics()
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize the graph to JSON."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        """Deserialize from a dictionary."""
        graph = cls(data.get("name", "default"))
        graph.created_at = data.get("created_at", graph.created_at)
        graph.updated_at = data.get("updated_at", graph.updated_at)
        
        # Add nodes
        for node_data in data.get("nodes", []):
            node = KnowledgeNode.from_dict(node_data)
            graph._nodes[node.id] = node
            graph.graph.add_node(node.id, **node.to_dict())
            graph._index[node.node_type].add(node.id)
        
        # Add edges
        for edge_data in data.get("edges", []):
            edge = KnowledgeEdge(
                source_id=edge_data["source_id"],
                target_id=edge_data["target_id"],
                edge_type=EdgeType(edge_data["edge_type"]),
                weight=edge_data.get("weight", 1.0),
                metadata=edge_data.get("metadata", {}),
                created_at=edge_data.get("created_at", ""),
                created_by=edge_data.get("created_by", "")
            )
            graph._edges.append(edge)
            graph.graph.add_edge(
                edge.source_id,
                edge.target_id,
                key=edge.edge_type.value,
                **edge.to_dict()
            )
        
        return graph
    
    @classmethod
    def from_json(cls, json_str: str) -> "KnowledgeGraph":
        """Deserialize from JSON."""
        return cls.from_dict(json.loads(json_str))
    
    def visualize_to_html(self) -> str:
        """Generate an HTML visualization of the graph."""
        # Create a simple D3.js visualization
        nodes_data = []
        for node in self._nodes.values():
            nodes_data.append({
                "id": node.id,
                "name": node.name[:30],
                "type": node.node_type.value,
                "group": list(NodeType).index(node.node_type)
            })
        
        edges_data = []
        for edge in self._edges:
            edges_data.append({
                "source": edge.source_id,
                "target": edge.target_id,
                "type": edge.edge_type.value,
                "weight": edge.weight
            })
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Knowledge Graph: {self.name}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ margin: 0; font-family: Arial, sans-serif; background: #0a0a0f; }}
        svg {{ width: 100vw; height: 100vh; }}
        .node {{ cursor: pointer; }}
        .node text {{ font-size: 10px; fill: #fff; }}
        .link {{ stroke-opacity: 0.6; }}
        .tooltip {{
            position: absolute; padding: 8px 12px;
            background: rgba(0,0,0,0.8); color: #fff;
            border-radius: 4px; font-size: 12px;
            pointer-events: none; opacity: 0;
        }}
        #legend {{
            position: fixed; top: 20px; right: 20px;
            background: rgba(0,0,0,0.8); padding: 15px;
            border-radius: 8px; color: #fff;
        }}
    </style>
</head>
<body>
    <div id="legend"></div>
    <div class="tooltip"></div>
    <svg></svg>
    <script>
        const nodes = {json.dumps(nodes_data)};
        const links = {json.dumps(edges_data)};
        
        const colors = d3.scaleOrdinal(d3.schemeCategory10);
        
        const svg = d3.select("svg");
        const width = window.innerWidth;
        const height = window.innerHeight;
        
        const simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links).id(d => d.id).distance(80))
            .force("charge", d3.forceManyBody().strength(-200))
            .force("center", d3.forceCenter(width / 2, height / 2));
        
        const link = svg.append("g")
            .selectAll("line")
            .data(links)
            .enter().append("line")
            .attr("class", "link")
            .attr("stroke", "#666")
            .attr("stroke-width", d => Math.sqrt(d.weight));
        
        const node = svg.append("g")
            .selectAll("g")
            .data(nodes)
            .enter().append("g")
            .attr("class", "node")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));
        
        node.append("circle")
            .attr("r", 8)
            .attr("fill", d => colors(d.group));
        
        node.append("text")
            .attr("dx", 12)
            .attr("dy", 4)
            .text(d => d.name);
        
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
        }});
        
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x; d.fy = d.y;
        }}
        function dragged(event, d) {{ d.fx = event.x; d.fy = event.y; }}
        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null; d.fy = null;
        }}
        
        // Legend
        const types = [...new Set(nodes.map(n => n.type))];
        const legend = d3.select("#legend");
        legend.append("h4").text("Node Types").style("margin", "0 0 10px 0");
        types.forEach((type, i) => {{
            const item = legend.append("div").style("margin", "5px 0");
            item.append("span")
                .style("display", "inline-block")
                .style("width", "12px")
                .style("height", "12px")
                .style("background", colors(i))
                .style("margin-right", "8px")
                .style("border-radius", "50%");
            item.append("span").text(type);
        }});
    </script>
</body>
</html>
"""
        return html


# Global knowledge graph instance
_global_graph: Optional[KnowledgeGraph] = None


def get_global_graph() -> KnowledgeGraph:
    """Get the global knowledge graph instance."""
    global _global_graph
    if _global_graph is None:
        _global_graph = KnowledgeGraph("global")
    return _global_graph


def set_global_graph(graph: KnowledgeGraph):
    """Set the global knowledge graph instance."""
    global _global_graph
    _global_graph = graph
