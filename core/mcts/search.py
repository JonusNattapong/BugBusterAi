import numpy as np
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple
from ..knowledge_graph.builder import KnowledgeGraphBuilder
from ..neural_nets.models import PolicyNetwork, ValueNetwork

class MCTSBugSearch:
    """Enhanced MCTS with neural guidance and parallel simulations."""
    
    def __init__(self, knowledge_graph: nx.DiGraph,
                 policy_net: PolicyNetwork,
                 value_net: ValueNetwork,
                 max_workers: int = 4):
        self.graph = knowledge_graph
        self.policy_net = policy_net
        self.value_net = value_net
        self.tree = {}  # Search tree with state caching
        self.simulation_count = 0
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.state_cache = {}
        
    def search(self, max_iterations: int = 1000,
               parallel_sims: int = 4) -> List[Dict]:
        """Perform MCTS with parallel simulations and neural guidance."""
        root_state = self._get_initial_state()
        self.tree[root_state] = {
            'visits': 0,
            'value': 0,
            'children': [],
            'state_vector': self._state_to_vector(root_state)
        }
        
        for _ in range(max_iterations // parallel_sims):
            # Parallel selection and expansion
            states = []
            for _ in range(parallel_sims):
                state = self._select(root_state)
                if not self._is_terminal(state):
                    state = self._expand(state)
                states.append(state)
            
            # Parallel simulation
            futures = [self.executor.submit(self._simulate, state)
                      for state in states]
            rewards = [f.result() for f in futures]
            
            # Backpropagation
            for state, reward in zip(states, rewards):
                self._backpropagate(state, reward)
        
        return self._get_best_bugs(root_state)
        
    def _select(self, state: str) -> str:
        """Select node using UCB1 algorithm."""
        if not self.tree[state]['children']:
            return state
            
        log_n = np.log(self.tree[state]['visits'])
        best_child = None
        best_score = -np.inf
        
        for child in self.tree[state]['children']:
            exploit = self.tree[child]['value'] / self.tree[child]['visits']
            explore = np.sqrt(2 * log_n / self.tree[child]['visits'])
            score = exploit + explore
            
            if score > best_score:
                best_score = score
                best_child = child
                
        return self._select(best_child)
        
    def _expand(self, state: str) -> str:
        """Expand the search tree."""
        new_state = self._get_next_state(state)
        self.tree[new_state] = {
            'visits': 0,
            'value': 0,
            'children': []
        }
        self.tree[state]['children'].append(new_state)
        return new_state
        
    def _simulate(self, state: str) -> float:
        """Simulate with neural network evaluation."""
        self.simulation_count += 1
        
        if state in self.state_cache:
            state_vec = self.state_cache[state]
        else:
            state_vec = self._state_to_vector(state)
            self.state_cache[state] = state_vec
        
        # Get neural network predictions
        with torch.no_grad():
            bug_prob = self.policy_net(state_vec)
            severity = self.value_net(state_vec)
        
        # Reward based on bug probability and severity
        reward = -(bug_prob * severity[2])  # Negative reward weighted by critical severity
        return float(reward)
        
    def _backpropagate(self, state: str, reward: float):
        """Backpropagate simulation results."""
        while state in self.tree:
            self.tree[state]['visits'] += 1
            self.tree[state]['value'] += reward
            state = self._get_parent_state(state)
            
    def _state_to_vector(self, state: str) -> torch.Tensor:
        """Convert state to feature vector for neural networks."""
        # Extract features from knowledge graph
        features = []
        
        # Node centrality features
        if state in self.graph:
            features.extend([
                self.graph.degree(state),
                nx.clustering(self.graph, state),
                len(list(self.graph.successors(state))),
                len(list(self.graph.predecessors(state)))
            ])
            
            # Node type specific features
            node_data = self.graph.nodes[state]
            if node_data.get('type') == 'function':
                features.append(len(node_data.get('args', [])))
            elif node_data.get('type') == 'variable':
                features.append(node_data.get('usage_count', 0))
        else:
            features.extend([0]*4)
        
        return torch.FloatTensor(features)
        
    def _get_initial_state(self) -> str:
        """Get initial state from knowledge graph."""
        return "root"
        
    def _get_next_state(self, current_state: str) -> str:
        """Generate next state to explore."""
        return f"state_{self.simulation_count}"
        
    def _is_terminal(self, state: str) -> bool:
        """Check if state is terminal."""
        return False
        
    def _get_parent_state(self, state: str) -> Optional[str]:
        """Get parent state in search tree."""
        for parent, data in self.tree.items():
            if state in data['children']:
                return parent
        return None
        
    def _get_best_bugs(self, root_state: str) -> List[Dict]:
        """Extract bugs with highest probability and severity."""
        bugs = []
        
        # Get top states by visit count
        states = sorted(
            self.tree.keys(),
            key=lambda s: self.tree[s]['visits'],
            reverse=True
        )[:10]
        
        for state in states:
            if state in self.graph:
                state_data = self.graph.nodes[state]
                if state_data.get('type') in ['function', 'variable']:
                    with torch.no_grad():
                        state_vec = self._state_to_vector(state)
                        bug_prob = self.policy_net(state_vec)
                        severity = self.value_net(state_vec)
                    
                    bugs.append({
                        'location': state,
                        'type': state_data.get('type'),
                        'probability': float(bug_prob),
                        'severity': {
                            'minor': float(severity[0]),
                            'moderate': float(severity[1]),
                            'critical': float(severity[2])
                        },
                        'visits': self.tree[state]['visits']
                    })
        
        return sorted(bugs, key=lambda x: x['probability'] * x['severity']['critical'], reverse=True)