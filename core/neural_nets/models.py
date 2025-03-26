import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    """Predicts likely bug locations in code."""
    
    def __init__(self, input_dim=256, hidden_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.locator = nn.Linear(hidden_dim, 1)  # Predicts bug probability
        
    def forward(self, x):
        features = self.encoder(x)
        return torch.sigmoid(self.locator(features))

class ValueNetwork(nn.Module):
    """Assesses severity of detected bugs."""
    
    def __init__(self, input_dim=256, hidden_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.assessor = nn.Linear(hidden_dim, 3)  # [minor, moderate, critical]
        
    def forward(self, x):
        features = self.encoder(x)
        return F.softmax(self.assessor(features), dim=-1)

class FixGenerator(nn.Module):
    """Generates potential fixes for detected bugs."""
    
    def __init__(self, vocab_size=10000, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.generator = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, max_length=20):
        embedded = self.embedding(x)
        _, (hidden, cell) = self.encoder(embedded)
        
        # Start with SOS token
        inputs = torch.zeros(embedded.size(0), 1).long().to(x.device)
        outputs = []
        
        for _ in range(max_length):
            decoder_embedded = self.embedding(inputs)
            out, (hidden, cell) = self.decoder(decoder_embedded, (hidden, cell))
            logits = self.generator(out.squeeze(1))
            outputs.append(logits)
            inputs = logits.argmax(-1).unsqueeze(1)
            
        return torch.stack(outputs, dim=1)

class BugBusterModel(nn.Module):
    """Combined model integrating all neural components."""
    
    def __init__(self):
        super().__init__()
        self.policy_net = PolicyNetwork()
        self.value_net = ValueNetwork()
        self.fix_gen = FixGenerator()
        
    def forward(self, code_representation):
        bug_probs = self.policy_net(code_representation)
        severities = self.value_net(code_representation)
        fixes = self.fix_gen(code_representation)
        return bug_probs, severities, fixes