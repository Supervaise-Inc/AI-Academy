// Week 8 · Transformer architecture. Follows GA Lab 3, Part 2 (backend/content/week-08.md, notebooks/src/week-08.py).
export default {
  week: 8,
  title: 'Tiny Transformer lab',
  lede: 'Build positional encoding and causal multi-head attention, train a small next-token model, then remove the mask and see what breaks.',
  lab: 'GA Lab 3, Part 2',
  chips: ['3 hours', 'CPU is enough', 'NumPy and PyTorch', 'No pretrained model'],
  exercises: [
    {
      tab: '1 · positions',
      title: 'Sinusoidal positional encoding',
      min: 20,
      kind: 'code',
      goal: 'Build the sine and cosine position table in NumPy and check one row by hand.',
      tasks: [
        'Run the cell and read the shape and the first values of rows 0 and 1.',
        'Work out PE(position 1) for d_model = 4 by hand, then check with positional_encoding(3, 4).round(4).',
        'Describe in two sentences how the columns change from left to right (d_model = 16, 24 positions).',
      ],
      code: `import numpy as np
def positional_encoding(n_pos, d_model):
    pos = np.arange(n_pos)[:, None]
    i = np.arange(0, d_model, 2)[None, :]
    angle = pos / (10000 ** (i / d_model))
    pe = np.zeros((n_pos, d_model))
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle)
    return pe
pe = positional_encoding(24, 16)
print(pe.shape, pe[0, :4], pe[1, :4].round(4))
print(positional_encoding(3, 4)[1].round(4))`,
      run: 'Run cell',
      out: '(24, 16) [0. 1. 0. 1.] [0.8415 0.5403 0.311  0.9504]\n[0.8415 0.5403 0.01   1.    ]',
      expect: 'Row 0 starts [0, 1, 0, 1]. For d_model = 4, row 1 is [sin 1, cos 1, sin 0.01, cos 0.01] = [0.8415, 0.5403, 0.01, 1.0]. The left columns change fast from one position to the next; the right columns change slowly.',
    },
    {
      tab: '2 · attention',
      title: 'Causal multi-head attention',
      min: 40,
      kind: 'code',
      goal: 'Run the multi-head attention layer and check the shapes, the causal mask and the parameter count.',
      tasks: [
        'Run the cell and read the two shapes, the row sum and the masked sum.',
        'Write the shape of q, k, v, scores and w at each step in (batch, heads, seq, d_head) form.',
        'Compute the parameter count by formula for d_model = 32 with biases (4 x 32^2 + 4 x 32) and for d_model = 512, and confirm the first against the printed number.',
      ],
      code: `import math, torch, torch.nn as nn
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.h, self.dh = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
    def forward(self, x, causal=True):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.reshape(B, T, self.h, self.dh).transpose(1, 2) for t in (q, k, v)]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        if causal:
            keep = torch.tril(torch.ones(T, T, device=x.device)).bool()
            scores = scores.masked_fill(~keep, float("-inf"))
        w = scores.softmax(dim=-1)
        y = (w @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(y), w
mha = MultiHeadAttention(d_model=32, n_heads=4)
y, w = mha(torch.randn(2, 5, 32))
print(y.shape, w.shape, w.sum(-1)[0, 0], w[0, 0].triu(1).abs().sum())
print(sum(p.numel() for p in mha.parameters()))`,
      run: 'Run cell',
      out: 'torch.Size([2, 5, 32]) torch.Size([2, 4, 5, 5]) tensor(1., ...) tensor(0., ...)\n4224',
      expect: 'Shapes [2, 5, 32] and [2, 4, 5, 5]; a row of weights sums to 1; everything above the diagonal is 0 (the mask); 4 x 32^2 + 4 x 32 = 4096 + 128 = 4224 parameters.',
    },
    {
      tab: '3 · block + model',
      title: 'A Transformer block and a tiny language model',
      min: 30,
      kind: 'code',
      goal: 'Assemble the block (attention, residual, layer norm, feed-forward) and the tiny model around it.',
      tasks: [
        'Run the cell and read the output shape.',
        'Draw the block as boxes and arrows and label each box with the corresponding line of code.',
        'Say where the two residual connections are.',
      ],
      code: `class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.attn = MultiHeadAttention(d, h)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, x):
        x = self.ln1(x + self.attn(x)[0])
        return self.ln2(x + self.ff(x))
class TinyLM(nn.Module):
    def __init__(self, vocab, d=64, h=4, n_ctx=16):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        pe = torch.tensor(positional_encoding(n_ctx, d), dtype=torch.float32)
        self.register_buffer("pe", pe)
        self.block = Block(d, h)
        self.head = nn.Linear(d, vocab)
    def forward(self, idx):
        x = self.emb(idx) + self.pe[: idx.size(1)]
        return self.head(self.block(x))

print(TinyLM(vocab=11)(torch.zeros(2, 16, dtype=torch.long)).shape)`,
      run: 'Run cell',
      out: 'torch.Size([2, 16, 11])',
      expect: 'One score per character in the vocabulary at every position: 2 sequences, 16 positions, 11 characters. The residual connections are the "x +" parts inside each ln1(...) and ln2(...).',
    },
    {
      tab: '4 · train',
      title: 'Train on a repeated string',
      min: 40,
      kind: 'code',
      goal: 'Train the model for 400 steps on "the cat sat on the mat. " repeated 40 times and watch the loss fall.',
      tasks: [
        'Compute ln 11 (the string has 11 different characters) and compare it with the loss at step 0.',
        'Run the cell and note how the loss falls. Store the values if you want to plot the curve.',
        'Explain why the targets y are the input shifted by one.',
      ],
      code: `text = "the cat sat on the mat. " * 40
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text])
T = 16
def get_batch(bs=32):
    ix = torch.randint(0, len(data) - T - 1, (bs,))
    x = torch.stack([data[i:i + T] for i in ix])
    y = torch.stack([data[i + 1:i + T + 1] for i in ix])
    return x, y
torch.manual_seed(0)
model = TinyLM(len(chars), n_ctx=T)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
for step in range(401):
    x, y = get_batch()
    loss = nn.functional.cross_entropy(model(x).reshape(-1, len(chars)), y.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0:
        print(step, round(loss.item(), 3))`,
      run: 'Run cell',
      out: 'Nine lines, for steps 0, 50, 100, ... 400. The exact numbers depend on the seed and your run.\n- Step 0: near ln 11 = 2.40 (it can be somewhat higher or lower, because the weights start random).\n- Then the loss falls steadily towards zero, because the string repeats exactly.\n- If it is not low after 400 steps, run more steps.',
      expect: 'A uniform guess among 11 characters has loss ln 11 = 2.398, so the first loss should be near 2.4. y is the input shifted by one because every position learns to predict the character that comes after it.',
    },
    {
      tab: '5 · generate',
      title: 'Generate text from a prompt',
      min: 10,
      kind: 'code',
      goal: 'Use the trained model to continue "the cat s" one character at a time.',
      tasks: [
        'Run the cell and read the generated text.',
        'Does it continue the pattern "the cat sat on the mat. "?',
        'Why does the loop keep only the last T = 16 characters as input?',
      ],
      code: `model.eval()
prompt = [stoi[c] for c in "the cat s"]
ids = torch.tensor([prompt])
with torch.no_grad():
    for _ in range(30):
        nxt = model(ids[:, -T:])[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
print("".join(chars[i] for i in ids[0]))`,
      run: 'Run cell',
      out: 'The text starts with "the cat s" and usually continues the pattern "the cat sat on the mat. ". The exact string depends on your training run.',
      expect: 'The positional table only has n_ctx = T = 16 rows, so a longer sequence would index past it. Cutting to the last T tokens avoids that mistake.',
    },
    {
      tab: '6 · break it',
      title: 'Remove the mask and compare',
      min: 25,
      kind: 'prompt',
      goal: 'Predict what happens when the causal mask is removed, and explain what the model learns to do instead of predicting.',
      tasks: [
        'In Block, change self.attn(x)[0] to self.attn(x, causal=False)[0]. Rebuild and retrain the model.',
        'Compare the loss curve and the generated text with the masked run.',
        'Ask the tutor to explain what you see. Say which change you made and what you observed.',
      ],
      presets: [
        { l: 'Vague question', u: 'Why is my model bad?' },
        {
          l: 'No mask',
          u: 'I trained the tiny Transformer with causal=False. The loss drops almost immediately but the generated text is poor. Why?',
        },
        {
          l: 'Mask needed',
          u: 'Why is the causal mask needed in training but automatic in generation?',
        },
      ],
      rules: [
        {
          u: 'needed in training|automatic|in generation|why .*mask',
          out: 'In training the whole string is fed in at once, so every position could look at the characters after it, including the one it must predict. The mask stops that: position t only sees positions up to t. In generation the future characters do not exist yet, because we add one new character at a time, so there is nothing to hide and the model only ever sees the past. The mask is set to minus infinity before the softmax so the weights become exactly 0.',
          note: 'matches the lab reflection',
          ok: true,
        },
        {
          u: 'causal\\s*=\\s*false|no mask|without (the )?mask|remov\\w* (the )?mask|unmasked',
          out: 'Without the mask, position t can attend to position t+1, which holds exactly the character it is supposed to predict. The model learns to copy it, so the training loss falls quickly without any real prediction. At generation time the future characters do not exist yet, so the copying trick has nothing to copy and the output is poor. The mask is what forces the model to learn the pattern from the past only.',
          note: 'matches the lab: the model reads the answer',
          ok: true,
        },
      ],
      fallback: {
        out: '"Bad" can mean a high loss, poor text or an error message. Tell me what you changed in the model, what the loss curve did and what the generated text looks like, and I can explain it.',
        note: 'vague prompt, vague answer',
        ok: false,
      },
    },
    {
      tab: '7 · reflection',
      title: 'Reflection',
      min: 15,
      kind: 'note',
      goal: 'Answer the three reflection questions from the lab.',
      tasks: ['Answer (a), (b) and (c) in a few sentences each.', 'Press Save card.'],
      note: '## Week 8 reflection\n\n### (a) Why the feed-forward network sees each token separately\n-\n\n### (b) What would happen to sequence order if I removed the positional encoding\n-\n\n### (c) Why the mask is needed in training but is automatic in generation\n-\n\n### Exercises I still find hard\n-',
    },
  ],
};
