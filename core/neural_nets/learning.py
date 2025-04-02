import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import deque
import random
import numpy as np
from typing import List, Dict, Tuple

class ExperienceReplay:
    """Store and sample experiences for reinforcement learning."""
    
    def __init__(self, capacity: int = 10000):
        self.memory = deque(maxlen=capacity)
        
    def push(self, state: torch.Tensor, action: int, reward: float, 
             next_state: torch.Tensor, done: bool):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> List[Tuple]:
        """Randomly sample a batch of experiences."""
        return random.sample(self.memory, min(batch_size, len(self.memory)))
        
    def __len__(self):
        return len(self.memory)

class ErrorLearning:
    """Learn from mistakes by analyzing error patterns."""
    
    def __init__(self, policy_net, value_net, learning_rate: float = 1e-4):
        self.policy_net = policy_net
        self.value_net = value_net
        self.error_memory = deque(maxlen=1000)
        self.optimizer = torch.optim.Adam(
            list(policy_net.parameters()) + list(value_net.parameters()),
            lr=learning_rate
        )
        
    def record_error(self, state: torch.Tensor, 
                    predicted: Dict, actual: Dict):
        """Record prediction errors for later analysis."""
        self.error_memory.append({
            'state': state,
            'predicted': predicted,
            'actual': actual
        })
        
    def analyze_errors(self, batch_size: int = 32):
        """Analyze recent errors and update models."""
        if len(self.error_memory) < batch_size:
            return
            
        errors = random.sample(self.error_memory, batch_size)
        states = torch.stack([e['state'] for e in errors])
        
        # Policy loss - difference between predicted and actual bug locations
        policy_loss = F.binary_cross_entropy(
            self.policy_net(states),
            torch.FloatTensor([e['actual']['bug'] for e in errors])
        )
        
        # Value loss - difference in severity assessment
        value_loss = F.cross_entropy(
            self.value_net(states),
            torch.LongTensor([e['actual']['severity'] for e in errors])
        )
        
        # Combined loss
        loss = policy_loss + value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

class SelfLearningDataset(Dataset):
    """Dataset for self-learning from generated data."""
    
    def __init__(self, data: List[Dict]):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            item['state'],
            item['bug_label'],
            item['severity_label'],
            item['fix_sequence']
        )

class SelfLearning:
    """Enable models to learn from their own predictions."""
    
    def __init__(self, policy_net, value_net, fix_gen,
                learning_rate: float = 1e-4):
        self.policy_net = policy_net
        self.value_net = value_net
        self.fix_gen = fix_gen
        self.self_memory = deque(maxlen=10000)
        self.optimizer = torch.optim.Adam(
            list(policy_net.parameters()) + 
            list(value_net.parameters()) +
            list(fix_gen.parameters()),
            lr=learning_rate
        )
        
    def record_episode(self, states: List[torch.Tensor], 
                      actions: List[int], rewards: List[float]):
        """Record a complete episode for self-learning."""
        for state, action, reward in zip(states, actions, rewards):
            self.self_memory.append({
                'state': state,
                'action': action,
                'reward': reward
            })
            
    def self_improve(self, batch_size: int = 32):
        """Improve models using self-generated data."""
        if len(self.self_memory) < batch_size:
            return
            
        batch = random.sample(self.self_memory, batch_size)
        states = torch.stack([b['state'] for b in batch])
        actions = torch.LongTensor([b['action'] for b in batch])
        rewards = torch.FloatTensor([b['reward'] for b in batch])
        
        # Policy gradient update
        log_probs = torch.log_softmax(self.policy_net(states), dim=-1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1))
        policy_loss = -torch.mean(selected_log_probs * rewards)
        
        # Value update
        value_loss = F.mse_loss(
            self.value_net(states).sum(dim=1),
            rewards
        )
        
        # Combined loss
        loss = policy_loss + value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()