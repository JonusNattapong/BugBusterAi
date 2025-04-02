import numpy as np
import networkx as nx
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from ..knowledge_graph.builder import KnowledgeGraphBuilder
from ..neural_nets.models import PolicyNetwork, ValueNetwork

if TYPE_CHECKING:
    from ..neural_nets.models import BugBusterModel

class MCTSBugSearch:
    """Enhanced MCTS with neural guidance, parallel simulations and learning capabilities."""
    
    def __init__(self, knowledge_graph: nx.DiGraph,
                 policy_net: PolicyNetwork,
                 value_net: ValueNetwork,
                 bug_buster_model: Optional['BugBusterModel'] = None,
                 max_workers: int = 4):
        self.graph = knowledge_graph
        self.policy_net = policy_net
        self.value_net = value_net
        self.bug_buster_model = bug_buster_model
        self.tree = {}  # Search tree with state caching
        self.simulation_count = 0
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.state_cache = {}
        self.error_log = []
        self.experience_log = []
        
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
            
        log_n = np.log(self.tree[state]['visits'] + 1e-6)  # Avoid log(0)
        best_child = None
        best_score = -np.inf
        
        for child in self.tree[state]['children']:
            # Enhanced selection with heuristic factors
            exploit = self.tree[child]['value'] / (self.tree[child]['visits'] + 1)
            explore = np.sqrt(2 * log_n / (self.tree[child]['visits'] + 1))
            
            # Additional heuristic - prefer nodes with higher complexity
            complexity = len(list(self.graph.successors(child))) if child in self.graph else 1
            heuristic = np.log(complexity + 1) * 0.1
            
            score = exploit + explore + heuristic
            
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
        """Simulate with neural network evaluation and learning.
        
        Uses enhanced reward calculation with:
        - Bug probability and severity
        - Code complexity factor
        - Usage frequency factor
        """
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
        
        # Enhanced reward calculation with multiple factors
        severity_score = severity[2]  # Critical severity
        complexity = len(list(self.graph.successors(state))) if state in self.graph else 1
        frequency = self.graph.nodes[state].get('usage_count', 1) if state in self.graph else 1
        
        # Reward formula combining multiple heuristics
        reward = -(
            bug_prob * severity_score * 0.7 +  # Bug severity impact
            (1 - 1/complexity) * 0.2 +        # Code complexity factor
            (1 - 1/frequency) * 0.1           # Usage frequency factor
        )
        
        # Record for learning if model is available
        if self.bug_buster_model:
            self.experience_log.append({
                'state': state_vec,
                'action': 0,  # Placeholder for actual action
                'reward': reward
            })
            
        return float(reward)
        
    def _backpropagate(self, state: str, reward: float):
        """Backpropagate simulation results with learning integration.
        
        Updates:
        - Node visit counts
        - Node values
        - Learning model if available
        """
        original_state = state
        total_reward = reward
        
        while state in self.tree:
            self.tree[state]['visits'] += 1
            self.tree[state]['value'] += reward
            
            # Apply learning decay for deeper nodes
            reward *= 0.9
            state = self._get_parent_state(state)
        
        # Record full episode for learning
        if self.bug_buster_model and original_state in self.state_cache:
            self.bug_buster_model.record_experience(
                [self.state_cache[original_state]],
                [0],  # Placeholder action
                [total_reward]
            )
            
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
                    
                    bug_info = {
                        'location': state,
                        'type': state_data.get('type'),
                        'probability': float(bug_prob),
                        'severity': {
                            'minor': float(severity[0]),
                            'moderate': float(severity[1]),
                            'critical': float(severity[2])
                        },
                        'visits': self.tree[state]['visits']
                    }
                    bugs.append(bug_info)
                    
                    # Record for error learning if model is available
                    if self.bug_buster_model and 'actual' in state_data:
                        self.bug_buster_model.record_error(
                            state_vec,
                            {'bug': bug_prob, 'severity': severity.argmax()},
                            state_data['actual']
                        )
        
        # Perform learning if model is available
        if self.bug_buster_model and self.experience_log:
            states = [e['state'] for e in self.experience_log]
            actions = [e['action'] for e in self.experience_log]
            rewards = [e['reward'] for e in self.experience_log]
            self.bug_buster_model.record_experience(states, actions, rewards)
            self.bug_buster_model.self_improve()
        
        return sorted(bugs, key=lambda x: x['probability'] * x['severity']['critical'], reverse=True)