import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig

class CodeTransformer(nn.Module):
    """Shared transformer encoder for code representation."""
    
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = BertConfig(
                vocab_size=30000,
                hidden_size=768,
                num_hidden_layers=6,
                num_attention_heads=12,
                intermediate_size=3072,
                max_position_embeddings=512
            )
        self.encoder = BertModel(config)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state

class PolicyNetwork(nn.Module):
    """Transformer-based bug location predictor."""
    
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
        self.locator = nn.Linear(768, 1)  # Predicts bug probability
        self.attention = nn.MultiheadAttention(768, 8)
        
    def forward(self, input_ids, attention_mask=None):
        features = self.transformer(input_ids, attention_mask)
        attn_output, _ = self.attention(
            features, features, features
        )
        return torch.sigmoid(self.locator(attn_output.mean(dim=1)))

class ValueNetwork(nn.Module):
    """Transformer-based bug severity assessor."""
    
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
        self.assessor = nn.Linear(768, 3)  # [minor, moderate, critical]
        
    def forward(self, input_ids, attention_mask=None):
        features = self.transformer(input_ids, attention_mask)
        return F.softmax(self.assessor(features.mean(dim=1)), dim=-1)

class FixGenerator(nn.Module):
    """Transformer-based fix generator."""
    
    def __init__(self, transformer, vocab_size=30000):
        super().__init__()
        self.transformer = transformer
        self.generator = nn.Linear(768, vocab_size)
        
    def forward(self, input_ids, attention_mask=None, max_length=20):
        encoder_outputs = self.transformer(input_ids, attention_mask)
        
        # Start with SOS token
        inputs = torch.zeros(input_ids.size(0), 1).long().to(input_ids.device)
        outputs = []
        
        for _ in range(max_length):
            decoder_outputs = self.transformer(inputs)
            logits = self.generator(decoder_outputs.mean(dim=1))
            outputs.append(logits)
            inputs = logits.argmax(-1).unsqueeze(1)
            
        return torch.stack(outputs, dim=1)

class BugBusterModel(nn.Module):
    """End-to-end transformer-based bug detection and fixing."""
    
    def __init__(self, pretrained_path=None):
        super().__init__()
        self.transformer = CodeTransformer()
        
        if pretrained_path:
            self.transformer.load_state_dict(torch.load(pretrained_path))
        
        self.policy_net = PolicyNetwork(self.transformer)
        self.value_net = ValueNetwork(self.transformer)
        self.fix_gen = FixGenerator(self.transformer)
        
    def forward(self, input_ids, attention_mask=None):
        shared_features = self.transformer(input_ids, attention_mask)
        
        bug_probs = self.policy_net(input_ids, attention_mask)
        severities = self.value_net(input_ids, attention_mask)
        fixes = self.fix_gen(input_ids, attention_mask)
        
        return bug_probs, severities, fixes
    
    def save_pretrained(self, path):
        """Save pretrained transformer weights."""
        torch.save(self.transformer.state_dict(), path)