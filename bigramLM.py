from train import get_batch, device, eval_iters, n_embd, block_size, n_heads, dropout, n_layers, max_iters, learning_rate
from prepare import decode, encode, vocab_size
import torch

class Head(torch.nn.Module):
    ''' single attention head'''

    def __init__(self, n_embd):
        super().__init__()
        self.key = torch.nn.Linear(n_embd, n_embd)
        self.query = torch.nn.Linear(n_embd, n_embd)
        self.value = torch.nn.Linear(n_embd, n_embd)
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        # get the key, query, and value
        k = self.key(x) # (B T C)
        q = self.query(x) # (B T C)
        v = self.value(x) # (B T C)
        # compute the dot product attention
        weights = q @ k.transpose(-1, -2) * C**-0.5 # (B T T)
        weights = weights.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        weights = torch.nn.functional.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        y = weights @ v # (B T C)
        return y
    
class multiHeadAttention(torch.nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = torch.nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = torch.nn.Linear(n_embd, n_embd)
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        # apply all the heads
        y = torch.cat([h(x) for h in self.heads], dim=-1)
        y = self.dropout(self.proj(y))
        return y
    
class FFN(torch.nn.Module):
# RELU non linearity
    def __init__(self, n_embd):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_embd, 4 * n_embd),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * n_embd, n_embd),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(torch.nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.multiHeadAttention = multiHeadAttention(n_heads, n_embd//n_heads)
        self.FFN = FFN(n_embd)
        self.norm1 = torch.nn.LayerNorm(n_embd)
        self.norm2 = torch.nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.multiHeadAttention(self.norm1(x))
        x = x + self.FFN(self.norm2(x))
        return x
    
class BigramLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, n_embd) # b t c
        self.sa_head = Head(n_embd)
        self.multiHeadAttention = multiHeadAttention(n_heads, n_embd//n_heads)
        self.FFN = FFN(n_embd)
        self.lm_head = torch.nn.Linear(n_embd, vocab_size)
        self.blocks = torch.nn.Sequential(
            *[Block(n_embd, n_heads) for _ in range(n_layers)],
            torch.nn.LayerNorm(n_embd)
        )
        # position embedding table
        self.positional_embeddings = torch.nn.Embedding(block_size, n_embd)

    def forward(self, idx, target =None):
        B, T = idx.shape

        # get the positional embeddings
        pos_embd = self.positional_embeddings(torch.arange(T, device=device)) # (T C)
        tok_embd = self.embedding(idx) # (B T C)
        x = tok_embd + pos_embd # (B T C)
        x = self.blocks(x) # (B T C)
        logits = self.lm_head(x) # (B T V)
        # v != c && v = vocab_size

        if target is not None:
            loss = torch.nn.CrossEntropyLoss()(logits.view(-1, vocab_size), target.view(-1))
        else:
            loss = None
            
        return logits, loss
    
    def generate(self, idx, max):
        for _ in range(max):
            idx_croped = idx[:, -block_size:] # (B T)
            logits, _ = self(idx_croped) # get the logits (B T C)
            # drop the time sequence dimension
            logits = logits[:, -1, :] # (B C)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # sample from the distribution
            next_idx = torch.multinomial(probs, num_samples=1) # (B 1)
            # append to the sequence
            idx = torch.cat([idx, next_idx], dim=1) # (B T+1)
        return idx
    
model = BigramLM()
model = model.to(device)

@torch.no_grad()
def evaluate_Loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(split)
            _ , loss = model(x, y)
            losses[i] = loss
        out[split] = losses.mean().item()
    model.train()
    return out

# train the model
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for i in range(max_iters):

    if i % 100 == 0:
        losses = evaluate_Loss()
        print(f"train loss: {losses['train']:.4f} val loss: {losses['val']:.4f}")

    # sample a batch of data
    x, y = get_batch('train')

    # forward pass
    logits, loss = model(x, y)
    # backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()




print(decode(model.generate(torch.zeros((1,1), dtype=torch.long, device=device), 10000)[0].tolist())) # generate some 10,000 char shakespeare