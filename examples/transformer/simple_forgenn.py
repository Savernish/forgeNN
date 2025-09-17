import os
import math
import numpy as np
import forgeNN as fnn

# Hyperparameters (roughly mirroring the torch example)
block_size = 256
learning_rate = 9e-4
iterations = 5000
eval_interval = 300
batch_size = 64
embeds_size = 192  # make divisible by num_heads
num_heads = 6
num_layers = 5
drop_prob = 0.15

# -------------------------
# Data loading (input.txt)
# -------------------------
_script_dir = os.path.dirname(__file__)
_candidates = [
    os.path.join(_script_dir, 'input.txt'),
    os.path.join(os.getcwd(), 'input.txt'),
]
_input_path = next((p for p in _candidates if os.path.exists(p)), None)
if _input_path is None:
    raise FileNotFoundError(
        "input.txt not found. Place it next to simple_forgenn.py or in the current working directory.\n"
        f"Tried: {os.path.join(_script_dir, 'input.txt')} and {os.path.join(os.getcwd(), 'input.txt')}"
    )

with open(_input_path, 'r', encoding='utf-8') as fp:
    text = fp.read()

if len(text) <= 1:
    raise ValueError("input.txt is too short (<=1 char). Provide a longer corpus.")

# Clamp block_size to avoid small-corpus errors
orig_block_size = block_size
block_size = min(block_size, max(2, len(text) - 1))
if block_size != orig_block_size:
    print(f"[info] block_size clamped from {orig_block_size} to {block_size} based on corpus length {len(text)}")

# Vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}

def encode(s: str):
    return [stoi[x] for x in s]

def decode(e: list[int]):
    return ''.join([itos[x] for x in e])

# Tokenized full corpus (NumPy int64)
data = np.asarray(encode(text), dtype=np.int64)

# Train/val split
split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]


def get_batch(split='train', block_size: int = block_size, batch_size: int = batch_size):
    arr = train_data if split == 'train' else val_data
    max_start = len(arr) - block_size
    if max_start <= 0:
        raise ValueError(
            f"block_size={block_size} must be < split length={len(arr)}. "
            "Consider reducing block_size or using a longer corpus."
        )
    ix = fnn.randint(0, max_start, size=(batch_size,))
    x = np.stack([arr[i : i + block_size] for i in ix], axis=0).astype(np.int64)
    y = np.stack([arr[i + 1 : i + block_size + 1] for i in ix], axis=0).astype(np.int64)
    return x, y


# -------------------------
# Model using forgeNN
# -------------------------
class CharTransformer:
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int, block_size: int, dropout: float = 0.0):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.block_size = block_size

        # Embeddings
        self.tok_emb = fnn.Embedding(vocab_size, embed_dim)
        self.pos_emb = fnn.Embedding(block_size, embed_dim)
        self.drop = fnn.Dropout(dropout)

        # Transformer blocks
        self.blocks = [fnn.TransformerBlock(embed_dim, num_heads, attn_dropout=dropout, proj_dropout=dropout, ffn_dropout=dropout) for _ in range(num_layers)]
        self.ln_f = fnn.LayerNorm(embed_dim)
        self.lm_head = fnn.Dense(vocab_size, in_features=embed_dim)
        self.training = True

    def __call__(self, idx: np.ndarray) -> fnn.Tensor:
        return self.forward(idx)

    def forward(self, idx: np.ndarray) -> fnn.Tensor:
        # idx: (B, T) int indices
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"T={T} exceeds block_size={self.block_size}")
        # Token + positional embeddings
        tok = self.tok_emb(idx)              # (B, T, C)
        pos_idx = np.broadcast_to(np.arange(T, dtype=np.int64), (B, T))
        pos = self.pos_emb(pos_idx)          # (B, T, C)
        x = tok + pos
        x = self.drop(x)                     # (B, T, C)
        # Blocks
        for blk in self.blocks:
            x = blk(x)                       # (B, T, C)
        x = self.ln_f(x)
        logits = self.lm_head(x)             # (B, T, V)
        return logits

    def parameters(self):
        params = []
        params.extend(self.tok_emb.parameters())
        params.extend(self.pos_emb.parameters())
        for blk in self.blocks:
            params.extend(blk.parameters())
        params.extend(self.ln_f.parameters())
        params.extend(self.lm_head.parameters())
        return params

    def train(self, flag: bool = True):
        self.training = bool(flag)
        self.drop.train(flag)
        for blk in self.blocks:
            blk.train(flag)
        self.ln_f.train(flag)
        return self

    def eval(self):
        return self.train(False)

    def generate(self, idx: np.ndarray, max_new_tokens: int = 100) -> np.ndarray:
        # idx: (B, T0)
        self.eval()
        out = idx.copy()
        for _ in range(max_new_tokens):
            idx_cond = out[:, -self.block_size:]
            logits = self.forward(idx_cond)          # (B, T, V)
            last = logits.data[:, -1, :]             # (B, V) np.ndarray
            # Softmax with numerical stability
            last = last - last.max(axis=1, keepdims=True)
            probs = np.exp(last)
            probs /= probs.sum(axis=1, keepdims=True)
            # Sample next token per batch
            next_idx = np.array([np.random.choice(self.vocab_size, p=probs[i]) for i in range(probs.shape[0])], dtype=np.int64)
            next_idx = next_idx.reshape(-1, 1)
            out = np.concatenate([out, next_idx], axis=1)
        return out


# Instantiate model and optimizer
model = CharTransformer(vocab_size, embeds_size, num_heads, num_layers, block_size, dropout=drop_prob)
opt = fnn.Adam(model.parameters(), lr=learning_rate)


def loss_and_logits(x_np: np.ndarray, y_np: np.ndarray):
    logits = model(x_np)                       # (B, T, V)
    B, T, V = logits.data.shape
    logits_flat = logits.reshape(B * T, V)
    y_flat = y_np.reshape(B * T)
    loss = fnn.cross_entropy_loss(logits_flat, y_flat)
    return loss, logits


def generate(n_chars: int = 200) -> str:
    sample = np.zeros((1, 1), dtype=np.int64)
    out = model.generate(sample, max_new_tokens=n_chars)
    return decode(out[0].tolist())


# Warm-up sample before training
print(generate(200))

# -------------------------
# Training loop
# -------------------------
for step in range(1, iterations + 1):
    X, y = get_batch('train')
    loss, _ = loss_and_logits(X, y)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % eval_interval == 0:
        # Switch to eval for generation snapshot
        model.eval()
        print(generate(200))
        model.train(True)
        print(f"step {step}: loss={float(loss.data):.4f}")

# Final sample
print(generate(500))
