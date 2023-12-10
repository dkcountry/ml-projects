import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 32
block_size = 128
max_iters = 5000
eval_interval = 300
learning_rate = 5e-4
device = 'cpu'
eval_iters = 200
n_embd = 192
n_heads = 6
torch.manual_seed(1337)
dropout = 0.2

# Load dataset
with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Get all unique chars
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Tokenize at char level
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(encode('hello world'))
print(decode(encode('hello world')))

# Tokenize whole dataset
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)

# Split data into train v. validation
n = int(.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# Get Train or Val
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


# Begin transformer modules
class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        weights = q @ k.transpose(-2, -1) * C**-0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=1)
        weights = self.dropout(weights)
        v = self.value(x)
        return weights @ v


class MultiHead(nn.Module):
    """Stack multiple self-attention heads together"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection(out)
        return self.dropout(out)


class SimpleNN(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHead(n_heads, head_size)
        self.ffwd = SimpleNN(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# GPT
class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            TransformerBlock(n_embd, n_heads),
            TransformerBlock(n_embd, n_heads),
            TransformerBlock(n_embd, n_heads),
            TransformerBlock(n_embd, n_heads),
            nn.LayerNorm(n_embd)
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == "__main__":
    model = GPT(vocab_size)
    m = model.to(device)

    xb, yb = get_batch('train')
    xb, yb = xb.to(device), yb.to(device)
    logits, loss = m(xb, yb)

    print(logits.shape)
    print(loss)

    # Train the model
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    for steps in range(10000):
        xb, yb = get_batch('train')
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    xb, yb = get_batch('val')
    _, val_loss = m(xb, yb)
    print(f'train loss: {loss.item()}', f'val loss: {val_loss.item()}')

    # Generate sample output
    idx = torch.zeros((1, 1), dtype=torch.long)
    out = m.generate(idx, 500)[0].tolist()
    print(decode(out))