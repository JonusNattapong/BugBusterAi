import networkx as nx
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
from ..ast_parser.parser import ASTParser

class KnowledgeGraphBuilder:
    """Builds a knowledge graph with persistent storage and incremental updates."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.graph = nx.DiGraph()
        self.ast_parser = ASTParser()
        self.db_path = db_path or ":memory:"
        self._init_db()
        
    def _init_db(self):
        """Initialize SQLite database for persistent storage."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS graph_nodes (
                id TEXT PRIMARY KEY,
                type TEXT,
                data TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS graph_edges (
                source TEXT,
                target TEXT,
                relationship TEXT,
                PRIMARY KEY (source, target, relationship),
                FOREIGN KEY (source) REFERENCES graph_nodes(id),
                FOREIGN KEY (target) REFERENCES graph_nodes(id)
            )
        """)

    def build_from_source(self, source_code: str, incremental: bool = False) -> nx.DiGraph:
        """Build knowledge graph with optional incremental updates."""
        self.ast_parser.parse(source_code)
        
        if incremental:
            self._load_from_db()
            
        self._add_functions(incremental)
        self._add_variable_flows(incremental)
        self._add_control_flows(incremental)
        self._add_data_flows()
        
        if incremental:
            self._save_to_db()
            
        return self.graph
        
    def _add_functions(self, incremental: bool = False):
        """Add function definitions with incremental support."""
        for func in self.ast_parser.get_function_definitions():
            if not incremental or func['name'] not in self.graph:
                self.graph.add_node(
                    func['name'],
                    type='function',
                    **func,
                    signature=self._get_function_signature(func)
                )
            
    def _add_variable_flows(self, incremental: bool = False):
        """Track variable flows with type information."""
        for var, data in self.ast_parser.var_assignments.items():
            if not incremental or var not in self.graph:
                self.graph.add_node(
                    var,
                    type='variable',
                    **data,
                    usage_count=len(data.get('lines', []))
                )
                
                # Connect to functions with scope awareness
                if data['scope'] != 'global':
                    self.graph.add_edge(
                        var,
                        data['scope'],
                        relationship='scoped_in'
                    )
                    
    def _add_control_flows(self, incremental: bool = False):
        """Add enhanced control flow relationships."""
        for func, flows in self.ast_parser.control_flows.items():
            for flow in flows:
                self.graph.add_node(
                    f"flow_{func}_{flow['lineno']}",
                    type='control_flow',
                    **flow
                )
                self.graph.add_edge(
                    func,
                    f"flow_{func}_{flow['lineno']}",
                    relationship='has_flow'
                )

    def _add_data_flows(self):
        """Analyze data flows between variables and functions."""
        for caller, calls in self.ast_parser.function_calls.items():
            for call in calls:
                if 'param_map' in call:
                    for param, arg in call['param_map'].items():
                        if isinstance(arg, str):  # Variable passing
                            self.graph.add_edge(
                                arg,
                                call['callee'],
                                relationship=f'param:{param}',
                                lineno=call['lineno']
                            )

    def _get_function_signature(self, func: Dict) -> str:
        """Generate function signature string."""
        params = ', '.join([
            f"{p}: {self.ast_parser.function_params[func['name']].get(p, {}).get('type', 'Any')}"
            for p in func['args']
        ])
        return f"{func['name']}({params})"

    def _save_to_db(self):
        """Save current graph to SQLite database."""
        with self.conn:
            # Save nodes
            for node, data in self.graph.nodes(data=True):
                self.conn.execute(
                    "INSERT OR REPLACE INTO graph_nodes VALUES (?, ?, ?)",
                    (node, data.get('type'), str(data))
                )
            
            # Save edges
            for src, dst, data in self.graph.edges(data=True):
                self.conn.execute(
                    "INSERT OR REPLACE INTO graph_edges VALUES (?, ?, ?)",
                    (src, dst, data.get('relationship', ''))
                )

    def _load_from_db(self):
        """Load graph from SQLite database."""
        # Load nodes
        for row in self.conn.execute("SELECT id, type, data FROM graph_nodes"):
            self.graph.add_node(row[0], type=row[1], **eval(row[2]))
        
        # Load edges
        for row in self.conn.execute("SELECT source, target, relationship FROM graph_edges"):
            self.graph.add_edge(row[0], row[1], relationship=row[2])

    def visualize(self):
        """Generate enhanced visualization with types and flows."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph, k=0.5, iterations=50)
        
        # Color nodes by type
        node_colors = {
            'function': 'lightblue',
            'variable': 'lightgreen',
            'control_flow': 'pink'
        }
        colors = [node_colors.get(data.get('type'), 'gray')
                 for _, data in self.graph.nodes(data=True)]
        
        nx.draw(
            self.graph, pos,
            with_labels=True,
            node_color=colors,
            font_size=8,
            edge_color='gray'
        )
        plt.show()