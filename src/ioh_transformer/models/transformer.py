import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, _ = q.size()
        
        # Linear projections
        q = self.w_q(q).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, v)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.w_o(output), attention_weights

class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward network"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output, attention_weights = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attention_weights

class MAPPeriFormer(nn.Module):
    """Main MAP-PeriFormer model architecture"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Feature dimensions
        self.static_dim = config['static_dim']
        self.dynamic_dim = config['dynamic_dim']
        self.medication_dim = config['medication_dim']
        self.d_model = config['d_model']
        
        # Embedding layers
        self.static_embedding = nn.Linear(self.static_dim, self.d_model)
        self.dynamic_embedding = nn.Linear(self.dynamic_dim, self.d_model)
        self.medication_embedding = nn.Linear(self.medication_dim, self.d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.d_model, config['max_seq_len'])
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=self.d_model,
                n_heads=config['n_heads'],
                d_ff=config['d_ff'],
                dropout=config['dropout']
            ) for _ in range(config['n_layers'])
        ])
        
        # Output heads
        self.continuous_head = nn.Linear(self.d_model, 3)  # 3,5,10-min predictions
        self.binary_head = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(64, 3),  # 3,5,10-min hypotension probabilities
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(config['dropout'])
        
    def forward(self, static_features, dynamic_features, medication_features):
        batch_size, seq_len, _ = dynamic_features.size()
        
        # Embed features
        static_emb = self.static_embedding(static_features).unsqueeze(1)
        dynamic_emb = self.dynamic_embedding(dynamic_features)
        medication_emb = self.medication_embedding(medication_features)
        
        # Combine embeddings
        x = dynamic_emb + medication_emb
        x = torch.cat([static_emb.expand(-1, seq_len, -1), x], dim=1)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Transformer encoding
        attention_weights = []
        for layer in self.encoder_layers:
            x, attn_weights = layer(x)
            attention_weights.append(attn_weights)
        
        # Use the last time step for prediction
        last_hidden = x[:, -1, :]
        
        # Output predictions
        continuous_pred = self.continuous_head(last_hidden)
        binary_pred = self.binary_head(last_hidden)
        
        return {
            'continuous': continuous_pred,  # [batch_size, 3]
            'binary': binary_pred,          # [batch_size, 3]
            'attention_weights': attention_weights
        }

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)
