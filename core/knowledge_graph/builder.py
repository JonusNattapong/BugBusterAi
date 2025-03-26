import networkx as nx
from typing import Dict, List
from ..ast_parser.parser import ASTParser

class KnowledgeGraphBuilder:
    """Builds a knowledge graph representing code relationships."""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.ast_parser = ASTParser()
        
    def build_from_source(self, source_code: str) -> nx.DiGraph:
        """Build knowledge graph from source code."""
        self.ast_parser.parse(source_code)
        self._add_functions()
        self._add_variable_flows()
        self._add_control_flows()
        return self.graph
        
    def _add_functions(self):
        """Add function definitions to the graph."""
        for func in self.ast_parser.get_function_definitions():
            self.graph.add_node(func['name'], type='function', **func)
            
    def _add_variable_flows(self):
        """Track how variables flow between functions."""
        var_usage = self.ast_parser.get_variable_usage()
        for var, lines in var_usage.items():
            self.graph.add_node(var, type='variable', lines=lines)
            
            # Connect variables to functions where they're used
            for func in self.ast_parser.get_function_definitions():
                if any(line in range(func['lineno'], func['lineno'] + 10) for line in lines):  # Simple heuristic
                    self.graph.add_edge(var, func['name'], relationship='used_in')
                    
    def _add_control_flows(self):
        """Add control flow relationships between functions."""
        # This would be enhanced with actual call graph analysis
        for func in self.ast_parser.get_function_definitions():
            func_source = func['source']
            for called_func in self.ast_parser.get_function_definitions():
                if called_func['name'] in func_source and called_func['name'] != func['name']:
                    self.graph.add_edge(
                        func['name'], 
                        called_func['name'], 
                        relationship='calls'
                    )

    def visualize(self):
        """Generate a visualization of the knowledge graph."""
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True)
        plt.show()