"""
Small Transformer-based classifier for text classification (from-scratch training).
Reduced to 0.5M-10M parameters for the fine-tuning vs from-scratch comparison lab.

Based on "Attention Is All You Need" (Vaswani et al., 2017).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Positional encoding using sine and cosine functions.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query, key, value: (batch_size, seq_len, d_model)
            mask: Optional mask (batch_size, seq_len, seq_len) or (seq_len, seq_len)

        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        batch_size = query.size(0)

        # Linear projections and reshape to (batch_size, num_heads, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, V)

        # Reshape and apply output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)

        return output


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerDecoderLayer(nn.Module):
    """
    Single Transformer decoder layer with self-attention and feed-forward.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: Causal mask (seq_len, seq_len)

        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # Pre-LayerNorm: normalize BEFORE attention (more stable training)
        # Self-attention with residual connection
        x_norm = self.norm1(x)
        attn_output = self.self_attn(x_norm, x_norm, x_norm, mask)
        x = x + self.dropout1(attn_output)

        # Pre-LayerNorm: normalize BEFORE feed-forward
        # Feed-forward with residual connection
        x_norm = self.norm2(x)
        ff_output = self.feed_forward(x_norm)
        x = x + self.dropout2(ff_output)

        return x


class SmallTransformerClassifier(nn.Module):
    """
    Small Transformer-based classifier for text classification.

    Designed to be 0.5M-10M parameters for from-scratch training.
    Uses last token's hidden state for classification.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        d_ff: int = 512,
        max_seq_len: int = 256,
        dropout: float = 0.3,
        pad_token_id: int = 0,
    ):
        """
        Initialize Small Transformer classifier.

        Args:
            vocab_size: Size of vocabulary
            num_classes: Number of classification classes
            d_model: Model dimension (embedding dimension)
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Dimension of feed-forward network
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            pad_token_id: ID of padding token
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Final layer normalization
        self.final_norm = nn.LayerNorm(d_model)

        # Classification head (simple linear layer)
        self.classifier = nn.Linear(d_model, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights with improved strategy for better training.
        Based on GPT-2 initialization.
        """
        # Initialize embedding layers
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        # Initialize all linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Special scaled init for residual projections (GPT-2 style)
        for layer in self.layers:
            # Scale attention output projection
            nn.init.normal_(
                layer.self_attn.W_o.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * self.num_layers)
            )
            # Scale feed-forward output projection
            nn.init.normal_(
                layer.feed_forward.linear2.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * self.num_layers)
            )

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generate causal mask to prevent attending to future tokens.

        Args:
            sz: Sequence length

        Returns:
            Causal mask of shape (sz, sz)
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return ~mask  # Invert: 1 means can attend, 0 means cannot

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for classification.

        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Optional attention mask (batch_size, seq_len)

        Returns:
            logits: Classification logits (batch_size, num_classes)
        """
        seq_len = input_ids.size(1)
        device = input_ids.device

        # Create causal mask
        causal_mask = self._generate_square_subsequent_mask(seq_len).to(device)

        # Embed tokens and add positional encoding
        # (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, causal_mask)

        # Final layer normalization
        x = self.final_norm(x)

        # Classification: use the last token's representation
        # (batch_size, seq_len, d_model) -> (batch_size, d_model)
        if attention_mask is not None:
            # Find the last non-padding token for each sequence
            seq_lengths = attention_mask.sum(dim=1) - 1  # -1 to get last valid index
            batch_indices = torch.arange(x.size(0), device=device)
            last_hidden_states = x[batch_indices, seq_lengths]
        else:
            # Use the last token in the sequence
            last_hidden_states = x[:, -1, :]

        # Project to class logits
        logits = self.classifier(last_hidden_states)

        return logits

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_small_classifier(vocab_size: int, num_classes: int, size: str = "medium") -> SmallTransformerClassifier:
    """
    Create a small transformer classifier with predefined size configurations.

    Args:
        vocab_size: Size of vocabulary
        num_classes: Number of classification classes
        size: One of "tiny" (~0.5M), "small" (~2M), "medium" (~5M), "large" (~10M)

    Returns:
        SmallTransformerClassifier instance
    """
    configs = {
        "tiny": {
            "d_model": 128,
            "num_heads": 4,
            "num_layers": 2,
            "d_ff": 512,
            "dropout": 0.3,
        },
        "small": {
            "d_model": 192,
            "num_heads": 4,
            "num_layers": 3,
            "d_ff": 768,
            "dropout": 0.3,
        },
        "medium": {
            "d_model": 256,
            "num_heads": 4,
            "num_layers": 3,
            "d_ff": 1024,
            "dropout": 0.3,
        },
        "large": {
            "d_model": 320,
            "num_heads": 4,
            "num_layers": 4,
            "d_ff": 1280,
            "dropout": 0.3,
        },
    }

    if size not in configs:
        raise ValueError(f"Size must be one of {list(configs.keys())}, got {size}")

    config = configs[size]
    model = SmallTransformerClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        **config
    )

    return model


if __name__ == "__main__":
    # Test different model sizes
    vocab_size = 10000
    num_classes = 4
    batch_size = 8
    seq_len = 128

    print("=" * 80)
    print("SMALL TRANSFORMER CLASSIFIER - PARAMETER COUNT TEST")
    print("=" * 80)

    for size in ["tiny", "small", "medium", "large"]:
        model = create_small_classifier(vocab_size, num_classes, size=size)

        # Test forward pass
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)

        logits = model(input_ids, attention_mask)

        num_params = model.get_num_parameters()

        print(f"\n{size.upper()} Model:")
        print(f"  Config: d_model={model.d_model}, layers={model.num_layers}, "
              f"heads={model.num_heads}")
        print(f"  Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Output shape: {logits.shape}")

        # Verify output shape
        assert logits.shape == (batch_size, num_classes), \
            f"Expected shape ({batch_size}, {num_classes}), got {logits.shape}"

    print("\n" + "=" * 80)
    print("All model sizes tested successfully!")
    print("=" * 80)
