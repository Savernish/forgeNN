"""Transformer layers scaffold for forgeNN.

Safe stubs that don't break imports and clearly guide users until implementation lands.
"""

from __future__ import annotations
from typing import Optional, List
import numpy as np

from ..layers import Layer, Dense, Dropout, LayerNorm, Embedding
from ..core.tensor import Tensor


class TransformerLayer(Layer):  # pragma: no cover - scaffold base
    """Base class for transformer layers.

    Subclasses should implement forward(x: Tensor) -> Tensor.
    This base class integrates with the Layer API: parameters(), train()/eval().
    """

    def parameters(self) -> List[Tensor]:  # no params by default
        return []


class MHA(TransformerLayer):  # pragma: no cover - stub
    """A Multi-Head Self-Attention module.

    Attention is all you need!
    """

    def _get_causal_mask(self, T: int) -> Tensor:
        """Return a cached (1,1,T,T) causal mask tensor with 0 on/below diag and -1e9 above.

        Grows the cache if a larger T is requested; smaller T returns a slice view.
        """
        if getattr(self, "_mask_np", None) is None or getattr(self, "_mask_T", 0) < T:
            mask_np = np.triu(np.ones((T, T), dtype=np.float32), k=1) * -1e9
            # Cache full-sized numpy mask and its T
            self._mask_np = mask_np.reshape(1, 1, T, T)
            self._mask_T = T
        # Return a Tensor wrapping the appropriate slice (no grad)
        m = self._mask_np[:, :, :T, :T]
        return Tensor(m, requires_grad=False)

    def _attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Scaled dot-product attention."""
        d_k = q.shape[-1]
        scores = (q @ k.transpose(0, 1, 3, 2)) / (d_k ** 0.5)
        # Apply cached causal mask so position i cannot attend to future positions (> i)
        T = q.shape[-2]
        scores = scores + self._get_causal_mask(T)

        weights = scores.softmax(axis=-1)
        # Attention dropout
        if hasattr(self, 'attn_drop'):
            weights = self.attn_drop(weights)
        return weights @ v

    def __init__(self, embed_dim: int, num_heads: int, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        """
        Args:
            embed_dim (int): The total dimension of the embedding.
            num_heads (int): The number of attention heads.
            attn_dropout (float): Dropout rate applied to attention weights.
            proj_dropout (float): Dropout rate applied after the output projection.
        """
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = Dense(embed_dim, embed_dim)
        self.k_proj = Dense(embed_dim, embed_dim)
        self.v_proj = Dense(embed_dim, embed_dim)
        self.out_proj = Dense(embed_dim, embed_dim)
        # Dropouts
        self.attn_drop = Dropout(attn_dropout) if attn_dropout and attn_dropout > 0.0 else Dropout(0.0)
        self.proj_drop = Dropout(proj_dropout) if proj_dropout and proj_dropout > 0.0 else Dropout(0.0)
        # Mask cache
        self._mask_np = None
        self._mask_T = 0
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Performs the forward pass of the Multi-Head Attention.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
        
        Returns:
            Tensor: Output tensor of the same shape.
        """
        B, T, C = x.shape  # Batch size, sequence length, embedding dimension
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)  # (B, num_heads, T, head_dim)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)  # (B, num_heads, T, head_dim)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)  # (B, num_heads, T, head_dim)

        context = self._attention(q, k, v)  # (B, num_heads, T, head_dim)
        context = context.transpose(0, 2, 1, 3).reshape(B, T, C)  # (B, T, embed_dim)
        output = self.out_proj(context)
        # Output projection dropout
        output = self.proj_drop(output)
        return output
    
    def parameters(self):
        return (
            self.q_proj.parameters()
            + self.k_proj.parameters()
            + self.v_proj.parameters()
            + self.out_proj.parameters()
        )

    def train(self, flag: bool = True) -> "MHA":
        self.training = bool(flag)
        # Propagate to sub-layers (dropouts and linears)
        for sub in (self.q_proj, self.k_proj, self.v_proj, self.out_proj, self.attn_drop, self.proj_drop):
            if hasattr(sub, 'train'):
                sub.train(flag)
        return self

# Common alias to match naming in literature and other libs
MultiHeadAttention = MHA

class TransformerBlock(TransformerLayer):  # pragma: no cover - implemented
    """Pre-LN Transformer block: LN -> MHA -> residual, then LN -> FFN -> residual.

    Args:
        embed_dim: Model dimension (channels per token)
        num_heads: Number of attention heads
        mlp_ratio: Expansion for FFN hidden size (default 4.0)
        attn_dropout: Dropout rate on attention weights
        proj_dropout: Dropout rate on attention output projection
        ffn_dropout: Dropout rate applied after FFN second linear
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
    ):
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)

        # Layers
        self.ln1 = LayerNorm(self.embed_dim)
        self.attn = MHA(self.embed_dim, self.num_heads, attn_dropout=attn_dropout, proj_dropout=proj_dropout)
        self.ln2 = LayerNorm(self.embed_dim)
        hidden_dim = int(self.embed_dim * self.mlp_ratio)
        self.fc1 = Dense(hidden_dim, in_features=self.embed_dim)
        self.fc2 = Dense(self.embed_dim, in_features=hidden_dim)
        self.ffn_drop = Dropout(ffn_dropout) if ffn_dropout and ffn_dropout > 0.0 else Dropout(0.0)

    def forward(self, x: Tensor) -> Tensor:
        # Self-attention block (pre-LN)
        y = self.attn(self.ln1(x))
        x = x + y
        # FFN block (pre-LN)
        y = self.ln2(x)
        y = self.fc1(y).gelu()
        y = self.fc2(y)
        y = self.ffn_drop(y)
        x = x + y
        return x

    def parameters(self) -> List[Tensor]:
        params: List[Tensor] = []
        params.extend(self.ln1.parameters())
        params.extend(self.attn.parameters())
        params.extend(self.ln2.parameters())
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        return params

    def train(self, flag: bool = True) -> "TransformerBlock":
        self.training = bool(flag)
        for sub in (self.ln1, self.attn, self.ln2, self.fc1, self.fc2, self.ffn_drop):
            if hasattr(sub, 'train'):
                sub.train(flag)
        return self

class PositionalEncoding(TransformerLayer):  # pragma: no cover - stub
    """Sinusoidal positional encoding (stub)."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "PositionalEncoding (sinusoidal) is not implemented yet. "
            "Targeted for v2.0.0 — see README.md."
        )


class PositionalEmbedding(TransformerLayer):  # pragma: no cover - stub
    """Learnable positional embedding (stub)."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "PositionalEmbedding (learnable) is not implemented yet. "
            "Targeted for v2.0.0 — see README.md."
        )
