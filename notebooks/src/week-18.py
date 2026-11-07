# %% [markdown]
# # Week 18 — Scaling and Deploying AI: lab notebook
#
# Built for **Google Colab**. Before you start, choose **File → Save a copy in Drive**, so that you work in your own copy and your answers are kept.
#
# **Runtime:** a normal CPU runtime is enough for everything. If you choose a **GPU runtime** (Runtime → Change runtime type), the notebook also measures on the GPU wherever that is interesting; nothing breaks without one. **Time:** about 3 hours for this notebook; the longest cells run for under a minute.
#
# This lab is called **Imagine the warehouse and the delivery trucks**. A company ships goods from one warehouse with one truck. The warehouse is **too small** for all the goods (that is the **memory** limit of one GPU), and the truck is **too slow** to deliver before the deadline (that is the **time** limit). You can build several warehouses and run many trucks (parallelism), pack the goods in smaller boxes (fewer bits per number), and organise the deliveries so that no truck waits (batching). *How much memory and time does a model really need, and what does each trick buy, on the computer in front of you?* You measure everything, with a fixed random seed, and every part says what it measured **on this machine**: on a laptop CPU some of the tricks help and some do not, and a GPU would give different timings. Memory sizes, in contrast, are counted exactly and do not depend on the machine.
#
# ### What you will do
#
# 1. **Part 1. Imagine the warehouse: memory and time** (25 min): build a small Transformer, count its parameters, measure the memory of training and of activations, build the memory table for 7 B and 70 B models and estimate training time.
# 2. **Part 2. Data parallelism by hand and for real** (30 min): split a batch into shards, average the gradients, prove that it equals the big batch, run real multi-process training with `torch.distributed`, and measure the cost of the all-reduce.
# 3. **Part 3. Tensor and pipeline parallelism** (30 min): split matrices across simulated devices, split a model into stages, simulate the GPipe schedule and check the bubble formula.
# 4. **Part 4. Mixed precision** (30 min): write the number formats from scratch, see overflow and underflow, train with loss scaling, and time the real `torch.autocast`.
# 5. **Part 5. Quantization** (30 min): write symmetric quantization, measure error against bits, quantise a trained network and time it.
# 6. **Part 6. Serving: latency, throughput and the KV cache** (30 min): measure batching, simulate a server, and build and measure a KV cache.
# 7. **Part 7. Plan the deployment and reflect** (20 min): a capacity planner, a deployment checklist and your findings table.
#
# ### How this notebook works
#
# Run the cells from top to bottom with the ▶ button. Every part starts with **✏️ Imagine first**: write a prediction before you run the code, then see how close you were. **✏️ Your turn** cells (the exercises) are for you: write the answer in the starter cell. A **Check** cell tells you whether you are right (✓ or ✗ with a hint). Open **Show a possible answer** only after you tried. Counts, sizes and formulas must match exactly; **timings depend on your computer**, so the checks test properties (equality, signs, formulas) and not exact times.
#
# %% [markdown]
# ## Setup
#
# Run the cell below once. It loads the libraries, fixes the random seed `SEED = 42`, finds out **what hardware you have** (a CPU, or also a GPU), and defines the small helpers that all the parts share: a timer, byte counters and a bandwidth probe.
#
# **Read the hardware report that it prints.** Scaling and deployment are about hardware, so the numbers in this lab depend on the machine: on a laptop CPU, half precision can be *slower* than 32-bit, while on a GPU with Tensor Cores it is usually much faster. Every part says what it measured **on this machine**, and whenever a GPU is available the notebook also measures on the GPU. Memory sizes, on the other hand, are counted exactly and do not depend on the machine.

# %%
import os, re, io, gc, math, time, json, random, platform, subprocess, sys, warnings
from collections import Counter, defaultdict
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
HAS_CUDA = torch.cuda.is_available()
device = "cuda" if HAS_CUDA else "cpu"
QUICK = bool(os.environ.get("NB_TEST"))   # lets the author shorten long loops when testing; leave it alone

def time_ms(fn, repeat=5, warmup=1):
    """Average time of fn() in milliseconds over `repeat` runs after `warmup` runs. On a GPU it waits for the GPU to finish before it stops the clock."""
    for _ in range(warmup): fn()
    if HAS_CUDA: torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(repeat): fn()
    if HAS_CUDA: torch.cuda.synchronize()
    return (time.perf_counter() - t) / repeat * 1000

def n_params(model):
    """Number of parameters of a model."""
    return sum(p.numel() for p in model.parameters())

def param_bytes(model):
    """Bytes that the parameters of a model occupy (number of values times bytes per value, counted exactly)."""
    return sum(p.numel() * p.element_size() for p in model.parameters())

def state_mb(model):
    """Size in megabytes of the model's state dict when saved (what a checkpoint file would weigh)."""
    b = io.BytesIO(); torch.save(model.state_dict(), b); return len(b.getvalue()) / 1e6

def human_bytes(x):
    """Write a number of bytes as B, KB, MB or GB (powers of 1000)."""
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(x) < 1000 or u == "TB": return f"{x:.0f} {u}" if u == "B" else f"{x:.2f} {u}"
        x /= 1000

def _cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"): return line.split(":", 1)[1].strip()
    except Exception: pass
    return platform.processor() or platform.machine()

# ---- What hardware do we have, and is half precision fast on it? ----
_a32 = torch.randn(512, 512); _b32 = torch.randn(512, 512)
_t32 = time_ms(lambda: _a32 @ _b32, repeat=3)
_a16 = _a32.bfloat16(); _b16 = _b32.bfloat16()
_t16 = time_ms(lambda: _a16 @ _b16, repeat=2)
CPU_BF16_RATIO = _t16 / _t32      # above 1: bfloat16 matrix products are slower than 32-bit on this CPU
print("Python", platform.python_version(), "| PyTorch", torch.__version__, "| threads:", torch.get_num_threads())
print("CPU:", _cpu_name())
if HAS_CUDA:
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, {props.total_memory / 1e9:.1f} GB of memory, compute capability {props.major}.{props.minor}")
else:
    print("GPU: none found (CUDA is not available); the GPU measurements are skipped and the text says what to expect.")
print(f"A 512x512 matrix product on this CPU: float32 {_t32:.2f} ms, bfloat16 {_t16:.2f} ms ({CPU_BF16_RATIO:.1f}x {'slower' if CPU_BF16_RATIO > 1 else 'faster'}).")
print("Quantization engines available:", torch.backends.quantized.supported_engines)


# %% [markdown]
# ### Two small extra helpers
#
# `check(...)` prints a tick or a cross with a hint, so you can test your own answers. `RESULTS` is a dictionary where you keep your key numbers; Part 7 prints it as your summary.

# %%
if "RESULTS" not in globals():
    RESULTS = {}   # a place to keep your key numbers and answers; the last part prints them together

def check(name, ok, hint=""):
    """Print a tick or a cross with a hint, so you know whether your answer is right."""
    print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))

# %% [markdown]
# ### Smoke test
#
# Before the first part, check the things that this lab needs: the timer, the byte counters, the process tools for Part 2 (a child Python process and the `gloo` communication backend) and a free network port on your own computer. Nothing is downloaded.

# %%
import socket
import torch.distributed as dist
smoke_t = time_ms(lambda: torch.randn(256, 256) @ torch.randn(256, 256), repeat=3)
smoke_model = nn.Linear(100, 10)
smoke_child = subprocess.run([sys.executable, "-c", "print(6*7)"], capture_output=True, text=True, timeout=60)
with socket.socket() as _s:
    _s.bind(("127.0.0.1", 0)); smoke_port = _s.getsockname()[1]
print("device:", device, "| threads:", torch.get_num_threads(), "| CPU_BF16_RATIO: %.1f" % CPU_BF16_RATIO)
print("a 256 x 256 matrix product takes %.2f ms here" % smoke_t)
print("Linear(100, 10): %d parameters = %d bytes (%s)" % (n_params(smoke_model), param_bytes(smoke_model), human_bytes(param_bytes(smoke_model))))
print("child Python process answered:", smoke_child.stdout.strip(), "| torch.distributed available:", dist.is_available(), "| gloo:", dist.is_available() and dist.is_gloo_available())
print("a free local port was found:", smoke_port > 0)
check("setup works", n_params(smoke_model) == 1010 and param_bytes(smoke_model) == 4040 and smoke_child.stdout.strip() == "42" and dist.is_available() and dist.is_gloo_available() and smoke_port > 0,
      "Run the setup cell first. If torch.distributed is missing, Part 2 shows what to do.")

# %% [markdown]
# ## Part 1. Imagine the warehouse: memory and time (25 min)
#
# *Lesson sections: "Why one GPU is not enough" and "Serving a model" (the memory budget).* Picture the company again. The **warehouse** holds everything that must be at hand at the same moment: the goods themselves (the **weights**), the notes on how to change each good (the **gradients**), two ledgers per good that the optimiser keeps (**Adam's** running averages), and the half-finished work lying on the benches (the **activations**). The **truck** is the speed of the computer: how many arithmetic operations it can do each second. A model fails to train for one of two reasons: ==the warehouse is too small, or the truck is too slow==. In this part you build a small Transformer, count what its warehouse holds **exactly** (bytes taken from the real tensors, not from a formula), measure how the benches fill up when the batch or the text gets longer, scale the numbers up to 7-billion and 70-billion-parameter models, and finish with the truck: how long would training take?

# %% [markdown]
# ### ✏️ Imagine first
#
# (1) A Transformer decoder with d = 256, 4 layers and a vocabulary of 4,000 tokens: how many parameters does it have, to the nearest million? (2) Training it with AdamW in 32-bit numbers: how many **bytes per parameter** does the optimiser step need in total (weights, gradients and the two Adam ledgers)? (3) You double the batch size. By what factor does the memory of the **activations** grow? (4) How many 80 GB GPUs are needed just to *hold* the training state of a 70-billion-parameter model? Write your guesses in the block below (replace each `None`), then run it. You compare with Steps 1.1, 1.2, 1.3 and 1.4.

# %%
# Imagine first: the warehouse
my_guess_params_m = None   # parameters of the small model, in millions (a number)
my_guess_bytes_per_param = None   # bytes per parameter during an AdamW step in 32-bit (a whole number)
my_guess_act_factor = None   # growth factor of the activation memory when the batch doubles (a number)
my_guess_gpus_70b = None   # number of 80 GB GPUs to hold the training state of a 70 B model (a whole number)

# %% [markdown]
# ### Step 1.1. Build a small Transformer and count what it has
#
# The block below writes a decoder-only Transformer by hand, in the same shape as the models of Weeks 7 and 8: token and position embeddings, `L` blocks (layer norm, causal self-attention, layer norm, feed-forward network with a 4 d hidden layer, each with a residual connection), a final layer norm and an output layer that gives one score per word of the vocabulary. The attention is split into small modules (`P1Scores`, `P1Mix`) so that the next steps can see every intermediate tensor. We use `d = 256`, 4 layers, 4 heads, a vocabulary of 4,000 and a context of up to 256 tokens, and set `P1_CTX`, `P1_V` and so on as the shared settings of this part.
#
# The formula of Week 8 says that a block has about 12 d² parameters (4 d² in attention, 8 d² in the feed-forward network), so the model should have about 12 L d² for the blocks plus the embeddings. We count the real parameters and compare. The exact count also contains the small biases and layer-norm parameters: 13 d per block.

# %%
P1_D, P1_L, P1_H, P1_V, P1_CTX = 256, 4, 4, 4000, 256

class P1Scores(nn.Module):
    """Attention probabilities: softmax(q k^T / sqrt(dh)) with a causal mask. Output shape (batch, heads, T, T)."""
    def forward(self, q, k):
        T = q.shape[-2]
        att = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), 1)
        return att.masked_fill(mask, float("-inf")).softmax(-1)

class P1Mix(nn.Module):
    """Mix the value vectors with the attention probabilities."""
    def forward(self, p, v):
        return p @ v

class P1Attn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.scores, self.mix = P1Scores(), P1Mix()
    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        o = self.mix(self.scores(q, k), v)
        return self.proj(o.transpose(1, 2).reshape(B, T, d))

class P1Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.attn, self.ln2 = nn.LayerNorm(d), P1Attn(d, h), nn.LayerNorm(d)
        self.fc1, self.act, self.fc2 = nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.fc2(self.act(self.fc1(self.ln2(x))))

class P1LM(nn.Module):
    def __init__(self, V=P1_V, d=P1_D, L=P1_L, h=P1_H, ctx=P1_CTX):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([P1Block(d, h) for _ in range(L)])
        self.lnf, self.head = nn.LayerNorm(d), nn.Linear(d, V, bias=False)
    def forward(self, ids):
        x = self.tok(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
        for b in self.blocks: x = b(x)
        return self.head(self.lnf(x))

torch.manual_seed(SEED)
p1_model = P1LM()
p1_n = n_params(p1_model)
p1_approx_blocks = 12 * P1_L * P1_D ** 2
p1_embed = 2 * P1_V * P1_D + P1_CTX * P1_D
print(f"parameters counted from the tensors: {p1_n:,}  ({p1_n / 1e6:.2f} million)")
print(f"12 L d^2 (blocks only):             {p1_approx_blocks:,}")
print(f"embeddings: tokens + output layer + positions = {p1_embed:,}")
print(f"12 L d^2 + embeddings:              {p1_approx_blocks + p1_embed:,}  ({100 * (p1_approx_blocks + p1_embed) / p1_n:.1f}% of the real count)")
print(f"bytes of the parameters in float32: {param_bytes(p1_model):,} = {human_bytes(param_bytes(p1_model))}")
assert abs(p1_approx_blocks + p1_embed - p1_n) / p1_n < 0.02

# %% [markdown]
# The model has **5.27 million parameters**. The formula 12 L d² gives 3.15 million for the four blocks, and the three embedding tables add 2.11 million: in this *small* model the embeddings and the output layer are **40 %** of everything, which is why the rule "12 L d²" only works for big models (in a 7-billion-parameter model the blocks dominate). The formula plus the embeddings is within 0.3 % of the real count; the remainder is the biases and the layer norms (13 d per block). At 4 bytes per float32 value the weights occupy **21.09 MB**.

# %% [markdown]
# ### ✏️ Your turn: Exercise 1.1. Count the parameters with a formula
#
# Your task: write `p1_count(V, d, L, ctx)` that returns the **exact** number of parameters of `P1LM` (with `h` heads: the number of heads does not change the count) from the sizes alone, without building the model. Use what Week 8 taught and the exact pieces of one block: the attention has `qkv` (3 d² + 3 d) and `proj` (d² + d), the feed-forward network has `fc1` (4 d² + 4 d) and `fc2` (4 d² + d), and each of the two layer norms has 2 d parameters. Then add the token table, the position table, the final layer norm and the output layer (no bias). Why it matters: you will use this function again to size models that are too big to build.
#
# Replace the starter, run it, then run the **Check** cell.

# %%
def p1_count(V, d, L, ctx):
    # your code here
    return None

# %%
if p1_count(P1_V, P1_D, P1_L, P1_CTX) is None:
    print("Write p1_count above, then run this cell again.")
else:
    check("exact count equals the real model", p1_count(P1_V, P1_D, P1_L, P1_CTX) == p1_n, f"you got {p1_count(P1_V, P1_D, P1_L, P1_CTX):,}, the model has {p1_n:,}; check the biases and the layer norms")
    _other = P1LM(V=500, d=64, L=2, h=2, ctx=32)
    check("also right for another size", p1_count(500, 64, 2, 32) == n_params(_other), "the formula must work for any V, d, L, ctx")
    RESULTS["p1_params"] = p1_n

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def p1_count(V, d, L, ctx):
#     attn = (3 * d * d + 3 * d) + (d * d + d)        # qkv and proj
#     ffn = (4 * d * d + 4 * d) + (4 * d * d + d)     # fc1 and fc2
#     block = attn + ffn + 2 * (2 * d)                # two layer norms (weight and bias each)
#     return V * d + ctx * d + L * block + 2 * d + d * V
# ```
#
# A block has 12 d² + 13 d parameters: the 12 d² is the part that Week 8 uses, and the 13 d is the small remainder. Because the count is exact, a model that is too big to build can be sized with the same function.
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the formula, so that the cells after it work without your answer.

# %%
def p1_count(V, d, L, ctx):
    attn = (3 * d * d + 3 * d) + (d * d + d)
    ffn = (4 * d * d + 4 * d) + (4 * d * d + d)
    block = attn + ffn + 2 * (2 * d)
    return V * d + ctx * d + L * block + 2 * d + d * V
assert p1_count(P1_V, P1_D, P1_L, P1_CTX) == p1_n
print("exact formula:", f"{p1_count(P1_V, P1_D, P1_L, P1_CTX):,}", "= real count", f"{p1_n:,}")

# %% [markdown]
# ### Step 1.2. Measure the memory of one training step
#
# Training keeps four things per parameter. The **weights** `p`, the **gradients** `p.grad` (one per weight), and the two ledgers of **AdamW**, `exp_avg` (a running average of the gradient) and `exp_avg_sq` (a running average of its square). We run one real AdamW step on a batch and then add up the bytes of **every tensor** that the optimiser holds: `numel() * element_size()`. Nothing is estimated.
#
# The lesson says "16 bytes per parameter". There are two accountings that both give 16. In **plain 32-bit training** it is 4 (weights) + 4 (gradients) + 8 (two Adam ledgers) = 16. In **mixed-precision training** (Part 4) the working copies are 16-bit and a 32-bit **master copy** is kept: 2 (16-bit weights) + 2 (16-bit gradients) + 4 (master weights) + 8 (Adam) = 16. Same total, different pieces. The activations come on top in both.

# %%
torch.manual_seed(SEED)
p1_m = P1LM()
p1_opt = torch.optim.AdamW(p1_m.parameters(), lr=1e-3)
p1_ids = torch.randint(0, P1_V, (8, 128))
p1_loss = F.cross_entropy(p1_m(p1_ids[:, :-1]).reshape(-1, P1_V), p1_ids[:, 1:].reshape(-1))
p1_loss.backward()
p1_opt.step()

p1_w = sum(p.numel() * p.element_size() for p in p1_m.parameters())
p1_g = sum(p.grad.numel() * p.grad.element_size() for p in p1_m.parameters())
p1_a = sum(s["exp_avg"].numel() * s["exp_avg"].element_size() for s in p1_opt.state.values())
p1_b = sum(s["exp_avg_sq"].numel() * s["exp_avg_sq"].element_size() for s in p1_opt.state.values())
p1_total = p1_w + p1_g + p1_a + p1_b
print(f"weights:               {p1_w:>12,} bytes = {p1_w / p1_n:.1f} per parameter")
print(f"gradients:             {p1_g:>12,} bytes = {p1_g / p1_n:.1f} per parameter")
print(f"Adam exp_avg:          {p1_a:>12,} bytes = {p1_a / p1_n:.1f} per parameter")
print(f"Adam exp_avg_sq:       {p1_b:>12,} bytes = {p1_b / p1_n:.1f} per parameter")
print(f"total training state:  {p1_total:>12,} bytes = {p1_total / p1_n:.1f} per parameter = {human_bytes(p1_total)}")
print("mixed-precision accounting: 2 + 2 + 4 + 8 =", 2 + 2 + 4 + 8, "bytes per parameter")
assert p1_total == 16 * p1_n

# %% [markdown]
# The optimiser step holds **84,369,408 bytes**, exactly **16 bytes per parameter** (84.37 MB for this model): 4 for the weights, 4 for the gradients and 4 + 4 for the two Adam ledgers. The state of training is therefore **four times** the size of the weights alone. Note what is *not* in this number: the activations, which depend on the batch and on the length of the text (Step 1.3), and temporary buffers.

# %% [markdown]
# ### ✏️ Your turn: Exercise 1.2. Measure the optimiser state yourself
#
# Your task: write `p1_state_bytes(model, opt)` that returns a dictionary with the four keys `"weights"`, `"grads"`, `"exp_avg"` and `"exp_avg_sq"`, each the total number of bytes (`numel() * element_size()`) of the matching tensors **after** one optimiser step. The optimiser's tensors are in `opt.state`, a dictionary from each parameter to a dictionary of tensors. Why it matters: when a training job runs out of memory, adding up the real tensors tells you which of the four is the problem.
#
# Replace the starter, run it, then run the **Check** cell.

# %%
def p1_state_bytes(model, opt):
    # your code here
    return {}

# %%
_d = p1_state_bytes(p1_m, p1_opt)
if not _d:
    print("Write p1_state_bytes above, then run this cell again.")
else:
    check("four keys", set(_d) == {"weights", "grads", "exp_avg", "exp_avg_sq"}, "use exactly the four key names")
    check("weights and gradients are right", _d.get("weights") == p1_w and _d.get("grads") == p1_g, "sum numel() * element_size() over the parameters, and over p.grad")
    check("both Adam ledgers are right", _d.get("exp_avg") == p1_a and _d.get("exp_avg_sq") == p1_b, "they are the values of opt.state[p]")
    check("the total is 16 bytes per parameter", sum(_d.values()) == 16 * p1_n, "4 + 4 + 4 + 4 bytes in float32")
    RESULTS["p1_bytes_per_param"] = sum(_d.values()) / p1_n if _d else None

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def p1_state_bytes(model, opt):
#     nb = lambda t: t.numel() * t.element_size()
#     params = list(model.parameters())
#     return {"weights": sum(nb(p) for p in params),
#             "grads": sum(nb(p.grad) for p in params),
#             "exp_avg": sum(nb(opt.state[p]["exp_avg"]) for p in params),
#             "exp_avg_sq": sum(nb(opt.state[p]["exp_avg_sq"]) for p in params)}
# ```
#
# Each of the four has the same number of values as the model has parameters, and each value is a 4-byte float32: that is why the total is exactly 16 bytes per parameter. (If a gradient is `None`, the optimiser has not run yet.)
#
# </details>

# %% [markdown]
# Below the exercise we run our version, so that the cells after it work without your answer.

# %%
def p1_state_bytes(model, opt):
    nb = lambda t: t.numel() * t.element_size()
    params = list(model.parameters())
    return {"weights": sum(nb(p) for p in params), "grads": sum(nb(p.grad) for p in params),
            "exp_avg": sum(nb(opt.state[p]["exp_avg"]) for p in params), "exp_avg_sq": sum(nb(opt.state[p]["exp_avg_sq"]) for p in params)}
RESULTS["p1_bytes_per_param"] = sum(p1_state_bytes(p1_m, p1_opt).values()) / p1_n
print(p1_state_bytes(p1_m, p1_opt))
RESULTS.setdefault("p1_params", p1_n)

# %% [markdown]
# ### Step 1.3. Measure the activations: the benches fill up with the batch
#
# The **activations** are the intermediate results of the forward pass that the backward pass needs again. We measure them with a **forward hook** on every small module: a hook is a function that PyTorch calls with the module's output, and ours adds up `numel() * element_size()`. Because we hooked only the *leaf* modules (those with no children) and our attention is split into leaves, nothing is counted twice. As a cross-check we also ask autograd itself: `torch.autograd.graph.saved_tensors_hooks` reports every tensor that is **saved for the backward pass**, and we add up the distinct memory blocks (without the weights).
#
# We then run a grid of batch sizes and sequence lengths and look at how the activation memory grows. Think about the formula first: a Transformer block stores 16 d numbers per token (the outputs of the layer norms, the attention projections, the feed-forward hidden layer twice and so on) plus, for every head, a table of T × T attention probabilities.

# %%
def p1_act_bytes(model, B, T):
    """Activation bytes of one forward pass: (sum of the outputs of all leaf modules, bytes of tensors that autograd saved for backward)."""
    total, hooks = [0], []
    for m in model.modules():
        if len(list(m.children())) == 0:
            hooks.append(m.register_forward_hook(lambda mod, inp, out: total.__setitem__(0, total[0] + out.numel() * out.element_size())))
    saved, wptr = {}, {p.untyped_storage().data_ptr() for p in model.parameters()}
    def pack(t):
        ptr = t.untyped_storage().data_ptr()
        if ptr not in wptr: saved[ptr] = t.untyped_storage().nbytes()
        return t
    ids = torch.randint(0, P1_V, (B, T))
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = model(ids)
    for h in hooks: h.remove()
    return total[0], sum(saved.values())

p1_grid = [(B, T) for B in (1, 2, 4, 8) for T in (32, 64, 128, 256)]
p1_rows = [(B, T) + p1_act_bytes(p1_m, B, T) for (B, T) in p1_grid]
print(" batch  tokens  hook bytes (MB)  autograd saved (MB)")
for B, T, hb, sb in p1_rows:
    print(f"{B:6d} {T:7d} {hb / 1e6:15.2f} {sb / 1e6:20.2f}")
p1_x = {(B, T): hb for B, T, hb, sb in p1_rows}
print("doubling the batch: 8 x 128 over 4 x 128 =", round(p1_x[(8, 128)] / p1_x[(4, 128)], 3))
print("doubling the length: 4 x 256 over 4 x 128 =", round(p1_x[(4, 256)] / p1_x[(4, 128)], 3))
assert abs(p1_x[(8, 128)] / p1_x[(4, 128)] - 2) < 0.01

# %% [markdown]
# Doubling the **batch** doubles the activation memory (a factor of 1.997; not exactly 2.000 only because the position embedding is the same for every sequence): every sequence in the batch has its own benches. Doubling the **length** more than doubles it (2.18): that is the T × T attention table growing with the square of the length. The two measurements are of the same size and grow in the same way: 205 MB for the module outputs against 172 MB for the tensors that autograd really saves, at the largest setting. The hook count is about 19 % higher because some outputs (the last output of each residual branch, the final scores) are not needed again by the backward pass; so the hook count is a slight over-estimate and autograd's count is the exact need. We fit the hook count. Now fit the formula to the measurements.

# %%
p1_Bs = np.array([r[0] for r in p1_rows], float); p1_Ts = np.array([r[1] for r in p1_rows], float); p1_y = np.array([r[2] for r in p1_rows], float)
p1_A = np.stack([p1_Bs * p1_Ts, p1_Bs * p1_Ts ** 2, p1_Ts], axis=1)
p1_coef, *_ = np.linalg.lstsq(p1_A, p1_y, rcond=None)
p1_theory = (4 * (P1_L * 16 * P1_D + 2 * P1_D + P1_V), 4 * P1_L * P1_H, 4 * P1_D)   # bytes per token, per (token x T) for the attention tables, per position of the position embedding
p1_fit = p1_A @ p1_coef
print("fitted:  bytes = %.0f x (B T) + %.0f x (B T^2) + %.0f x T" % tuple(p1_coef))
print("theory:  bytes = %.0f x (B T) + %.0f x (B T^2) + %.0f x T   [4 x (16 d L + 2 d + V), 4 x L x heads, 4 x d]" % p1_theory)
print("largest relative error of the fit: %.2e" % np.max(np.abs(p1_fit - p1_y) / p1_y))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for B in (1, 2, 4, 8):
    ax[0].plot([32, 64, 128, 256], [p1_x[(B, T)] / 1e6 for T in (32, 64, 128, 256)], "o-", label=f"batch {B}")
ax[0].set_xlabel("tokens per sequence T"); ax[0].set_ylabel("activation memory (MB)"); ax[0].set_title("grows with T, faster than a line"); ax[0].legend()
for T in (32, 64, 128, 256):
    ax[1].plot([1, 2, 4, 8], [p1_x[(B, T)] / 1e6 for B in (1, 2, 4, 8)], "o-", label=f"T = {T}")
ax[1].set_xlabel("batch size B"); ax[1].set_ylabel("activation memory (MB)"); ax[1].set_title("grows like a straight line in B"); ax[1].legend()
plt.tight_layout(); plt.show()
assert np.allclose(p1_coef, p1_theory, rtol=1e-6)

# %% [markdown]
# The fit reproduces the theory to six digits: the activation memory is **83,584 bytes for every token in the batch plus 64 bytes for every token times the length** (and a tiny 1,024 bytes per position for the position embedding, which does not depend on the batch), which is exactly what counting the layers predicts (83,584, 64 and 1,024). The first term is the linear part (16 d numbers per token and layer, plus 2 d and the output layer's V scores per token); the second is the attention tables (L layers × h heads × T numbers per token). For a long context the second term wins, which is why long contexts are expensive and why **FlashAttention** (Week 8) avoids storing the table.
#
# Compare with the weights: the optimiser state of this model is 84.37 MB, and one batch of 8 sequences of 256 tokens needs **205 MB** of activations, which is **2.4 times** as much. The weights and the optimiser state do not depend on the batch; the activations do. That is why a bigger batch is the first thing that fails when memory runs out, and why people trade time for memory by recomputing activations (named in the lesson as activation checkpointing).

# %%
if HAS_CUDA:
    torch.cuda.reset_peak_memory_stats()
    _g = P1LM().to("cuda"); _o = torch.optim.AdamW(_g.parameters(), lr=1e-3)
    _ids = torch.randint(0, P1_V, (8, 128), device="cuda")
    F.cross_entropy(_g(_ids[:, :-1]).reshape(-1, P1_V), _ids[:, 1:].reshape(-1)).backward(); _o.step()
    torch.cuda.synchronize()
    p1_peak = torch.cuda.max_memory_allocated()
    print(f"GPU peak memory of one training step: {p1_peak / 1e6:.1f} MB; weights + grads + Adam = {16 * p1_n / 1e6:.1f} MB; the rest is activations and temporary buffers")
    del _g, _o
else:
    print("No GPU found: this cell is skipped. On a GPU it prints torch.cuda.max_memory_allocated() for the same step;")
    print("expect a number a little above the optimiser state plus the activations of Step 1.3.")

# %% [markdown]
# ### Step 1.4. The memory table for big models
#
# Now the formula becomes a function. The weights of a model with `P` billion parameters at `b` bits take `P × 10⁹ × b / 8` bytes, which is `P × b / 8` gigabytes (with 1 GB = 10⁹ bytes, as in the lesson). Training needs 16 bytes **in total** per parameter (the weights are included in that number), plus the activations. Then the number of GPUs is the memory divided by the memory of one GPU, rounded **up**, because you cannot buy half a GPU. We assume GPUs with 80 GB; this is an example size, not a specification of any product.

# %% [markdown]
# ### ✏️ Your turn: Exercise 1.3. The memory function
#
# Your task: write three functions. `weights_gb(P, bits)` returns the weight memory in GB of a model with `P` billion parameters stored with `bits` bits per value. `train_gb(P)` returns the training state in GB at 16 bytes per parameter (without activations). `gpus_needed(gb, gpu_gb=80)` returns the number of GPUs of `gpu_gb` GB needed to hold `gb` GB (round up). The lesson's numbers must come out: 7 B at 32, 16, 8 and 4 bits is 28, 14, 7 and 3.5 GB, training 7 B is 112 GB, and for 70 B: 140 GB to hold (FP16) and 1,120 GB to train.
#
# Replace the starter, run it, then run the **Check** cell.

# %%
def weights_gb(P, bits):
    # your code here
    return None

def train_gb(P):
    # your code here
    return None

def gpus_needed(gb, gpu_gb=80):
    # your code here
    return None

# %%
if weights_gb(7, 16) is None or train_gb(7) is None or gpus_needed(100) is None:
    print("Write the three functions above, then run this cell again.")
else:
    check("7 B at 32, 16, 8, 4 bits", [weights_gb(7, b) for b in (32, 16, 8, 4)] == [28, 14, 7, 3.5], "P * bits / 8 gigabytes")
    check("training 7 B takes 112 GB", abs(train_gb(7) - 112) < 1e-9, "16 bytes per parameter: 16 * P gigabytes")
    check("70 B: 140 GB to hold in FP16, 1,120 GB to train", abs(weights_gb(70, 16) - 140) < 1e-9 and abs(train_gb(70) - 1120) < 1e-9, "same formulas with P = 70")
    check("GPUs: 2 hold 70 B in FP16 and 14 train it", gpus_needed(140) == 2 and gpus_needed(1120) == 14, "round up: -(-gb // gpu_gb), or math.ceil")
    check("rounds up, never down", gpus_needed(80) == 1 and gpus_needed(80.1) == 2, "80 GB fits one GPU exactly, 80.1 GB needs two")
    RESULTS["p1_gpus_70b_train"] = gpus_needed(train_gb(70))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def weights_gb(P, bits):
#     return P * bits / 8          # P billion values * bits / 8 bytes, in gigabytes
#
# def train_gb(P):
#     return 16 * P                # 16 bytes per parameter: P billion parameters -> 16 * P gigabytes
#
# def gpus_needed(gb, gpu_gb=80):
#     return math.ceil(gb / gpu_gb)
# ```
#
# The unit trick: a billion parameters at 1 byte each is 1 GB, so the number of gigabytes is just the number of billions times the bytes per parameter. The 14 GPUs for 70 B are a **lower bound**: activations, the KV cache and unused space on each GPU come on top, and real systems split the state unevenly.
#
# </details>

# %% [markdown]
# Below the exercise we run our version, so that the cells after it work. Then the lesson's table: for each model size and precision, the memory to hold the weights and the number of 80 GB GPUs; and the memory to train.

# %%
def weights_gb(P, bits): return P * bits / 8
def train_gb(P): return 16 * P
def gpus_needed(gb, gpu_gb=80): return math.ceil(gb / gpu_gb)

assert [weights_gb(7, b) for b in (32, 16, 8, 4)] == [28, 14, 7, 3.5]
assert train_gb(7) == 112 and weights_gb(70, 16) == 140 and train_gb(70) == 1120
assert [gpus_needed(weights_gb(70, b)) for b in (32, 16, 8, 4)] == [4, 2, 1, 1] and gpus_needed(train_gb(70)) == 14
print("model   precision   weights GB   80 GB GPUs to hold")
for P in (7, 70):
    for b, name in ((32, "FP32"), (16, "FP16"), (8, "INT8"), (4, "INT4")):
        print(f"{P:3d} B   {name:9s} {weights_gb(P, b):9.1f}   {gpus_needed(weights_gb(P, b)):10d}")
print()
for P in (7, 70):
    print(f"training {P:2d} B (16 bytes per parameter, no activations): {train_gb(P):7,.0f} GB -> at least {gpus_needed(train_gb(P))} GPUs of 80 GB")
RESULTS["p1_gpus_70b_train"] = gpus_needed(train_gb(70))

# %% [markdown]
# Read the table from the top. A 7 B model needs 14 GB in FP16, which fits one 80 GB GPU for *inference* but not for *training* (112 GB). A 70 B model needs 140 GB in FP16: 2 GPUs just to hold it, and **14** to train it (1,120 GB). Precision is the cheapest lever: 4-bit weights bring 70 B down to 35 GB, which fits one GPU. These are counts of bytes; they hold on every machine.

# %% [markdown]
# ### Step 1.5. The truck: the time that training takes
#
# The other limit is time. Training a model takes about **C ≈ 6 N D** floating-point operations (FLOPs) for `N` parameters and `D` training tokens (Week 9): 2 N for the forward pass and 4 N for the backward pass, per token. To turn FLOPs into seconds we need the speed of the machine, in FLOP/s. We **measure it**: a matrix product of two n × n matrices costs 2 n³ operations, so dividing by the measured time gives the FLOP/s that this computer reaches on the operation that dominates a Transformer. Then we time a real training step of our small model and compare it with the 6 N D prediction.

# %%
p1_n_mm = 1024
p1_a_mm, p1_b_mm = torch.randn(p1_n_mm, p1_n_mm), torch.randn(p1_n_mm, p1_n_mm)
p1_t_mm = time_ms(lambda: p1_a_mm @ p1_b_mm, repeat=5, warmup=2) / 1000
p1_flops_s = 2 * p1_n_mm ** 3 / p1_t_mm
print(f"1024 x 1024 matrix product: {p1_t_mm * 1000:.1f} ms -> {p1_flops_s / 1e9:.0f} GFLOP/s on this machine")

p1_m2 = P1LM(); p1_o2 = torch.optim.AdamW(p1_m2.parameters(), lr=1e-3)
p1_batch = torch.randint(0, P1_V, (8, 129))
def p1_step():
    p1_o2.zero_grad()
    F.cross_entropy(p1_m2(p1_batch[:, :-1]).reshape(-1, P1_V), p1_batch[:, 1:].reshape(-1)).backward(); p1_o2.step()
p1_t_step = time_ms(p1_step, repeat=3 if QUICK else 5, warmup=1) / 1000
p1_tokens = 8 * 128
p1_pred = 6 * p1_n * p1_tokens / p1_flops_s
print(f"one training step on {p1_tokens} tokens: {p1_t_step * 1000:.0f} ms measured; 6 N D / (matmul speed) predicts {p1_pred * 1000:.0f} ms")
print(f"-> the step reached {100 * 6 * p1_n * p1_tokens / p1_t_step / p1_flops_s:.0f}% of the matrix-product speed")

p1_C = 6 * 7e9 * 1e12
p1_sec_here = p1_C / p1_flops_s
p1_gpu_rate, p1_gpus = 200e12, 1024      # ASSUMED: a sustained 200 TFLOP/s per GPU and a cluster of 1,024 GPUs
print(f"\n7 B parameters on 1 T tokens: C = 6 N D = {p1_C:.2e} FLOPs")
print(f"  on this machine ({p1_flops_s / 1e9:.0f} GFLOP/s):   {p1_sec_here:.2e} s = {p1_sec_here / 3.156e7:,.0f} years")
print(f"  on 1 assumed GPU (200 TFLOP/s):   {p1_C / p1_gpu_rate:.2e} s = {p1_C / p1_gpu_rate / 3.156e7:.1f} years")
print(f"  on 1,024 assumed GPUs (perfect):  {p1_C / p1_gpu_rate / p1_gpus:.2e} s = {p1_C / p1_gpu_rate / p1_gpus / 86400:.1f} days")
assert abs(p1_C - 4.2e22) < 1e18

# %% [markdown]
# On the machine that ran this lab the matrix product reaches **137 GFLOP/s**. A real training step of our small model takes 346 ms, and the 6 N D formula, with the measured matrix speed, predicts 237 ms: the formula is a **floor**, because a real step also does the attention, layer norms, softmax and optimiser update, which are not matrix products and run slower. Training a 7-billion-parameter model on a trillion tokens needs 6 × 7 × 10⁹ × 10¹² = **4.2 × 10²² FLOPs**. At this machine's speed that is about **9,726 years**, which is absurd; at an *assumed* sustained 200 TFLOP/s one GPU would still need 6.7 years, and 1,024 such GPUs working perfectly together about **2.4 days**. The 200 TFLOP/s and the 1,024 GPUs are assumptions for the arithmetic, not facts about any product, and "perfectly together" is what Parts 2 and 3 show you must pay for.

# %% [markdown]
# ### Step 1.6. Memory or time? A decision
#
# Both limits can bind, one, or neither. Write a function that takes a scenario and says which. A scenario is a model of `P` billion parameters that is either **trained** on `D` billion tokens before a deadline, or only **held** for inference, on GPUs with 80 GB, at an assumed sustained 200 TFLOP/s. The rules: memory is a problem if the state (16 bytes per parameter for training, 2 bytes for FP16 inference) is above the GPU memory; time is a problem if one GPU would need more than the deadline (training only).

# %%
# The four scenarios, and your guesses (write them BEFORE you write the function)
SCENARIOS = [("train 7 B on 1,000 B tokens in 30 days", 7, "train", 1000, 30),
             ("serve 70 B in FP16", 70, "infer", 0, None),
             ("train 0.3 B on 300 B tokens in 7 days", 0.3, "train", 300, 7),
             ("fine-tune 1 B on 0.1 B tokens in 3 days", 1, "train", 0.1, 3)]
my_answers = [None, None, None, None]   # write "memory", "time", "both" or "neither" for each scenario

# %% [markdown]
# ### ✏️ Your turn: Exercise 1.4. Memory or time?
#
# Your task: write `which_limit(P, mode, D_b=0, days=None, gpu_gb=80, rate=200e12)` that returns `"memory"`, `"time"`, `"both"` or `"neither"`. `mode` is `"train"` or `"infer"`. For `"train"` use `train_gb(P)` for memory and the time `6 * P * 1e9 * D_b * 1e9 / rate` seconds against `days * 86400`. For `"infer"` use `weights_gb(P, 16)` and ignore time. Then classify the four scenarios in the list `my_answers` (replace each `None` by one of the four words) **before** you run the function on them, and compare. The four scenarios are in the cell above: (1) train a 7 B model on 1,000 B tokens in 30 days; (2) serve a 70 B model in FP16 for the district; (3) train a 0.3 B model on 300 B tokens in 7 days; (4) fine-tune a 1 B model on 0.1 B tokens in 3 days.

# %%
def which_limit(P, mode, D_b=0, days=None, gpu_gb=80, rate=200e12):
    # your code here
    return None

# %%
if which_limit(7, "train", 1000, 30) is None:
    print("Write which_limit above, then run this cell again.")
else:
    got = [which_limit(P, mode, D, days) for (_, P, mode, D, days) in SCENARIOS]
    check("scenario 1 needs both", got[0] == "both", "112 GB does not fit 80 GB, and 6.7 years is more than 30 days")
    check("scenario 2 is a memory problem", got[1] == "memory", "140 GB against 80 GB; inference has no training deadline here")
    check("scenario 3 is a time problem", got[2] == "time", "4.8 GB fits easily; 5.4e20 FLOPs at 2e14 FLOP/s is about 31 days > 7")
    check("scenario 4 is neither", got[3] == "neither", "16 GB fits; 6e17 FLOPs is under an hour")
    print("your guesses :", my_answers); print("the function :", got)
    RESULTS["p1_guess_hits"] = sum(a == b for a, b in zip(my_answers, got))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def which_limit(P, mode, D_b=0, days=None, gpu_gb=80, rate=200e12):
#     if mode == "train":
#         mem_bad = train_gb(P) > gpu_gb
#         time_bad = days is not None and 6 * P * 1e9 * D_b * 1e9 / rate > days * 86400
#     else:
#         mem_bad = weights_gb(P, 16) > gpu_gb
#         time_bad = False
#     return "both" if mem_bad and time_bad else "memory" if mem_bad else "time" if time_bad else "neither"
# ```
#
# Training a 0.3 B model on 300 B tokens is the interesting one: it fits in memory with room to spare (4.8 GB), but at one GPU's assumed speed it takes about 31 days, so the problem is the truck, not the warehouse. The fix is **data parallelism** (Part 2): several copies of the model, each reading a slice of the data.
#
# </details>

# %% [markdown]
# Below the exercise we run our version, so that the next cell works.

# %%
def which_limit(P, mode, D_b=0, days=None, gpu_gb=80, rate=200e12):
    if mode == "train":
        mem_bad = train_gb(P) > gpu_gb
        time_bad = days is not None and 6 * P * 1e9 * D_b * 1e9 / rate > days * 86400
    else:
        mem_bad = weights_gb(P, 16) > gpu_gb
        time_bad = False
    return "both" if mem_bad and time_bad else "memory" if mem_bad else "time" if time_bad else "neither"
p1_got = [which_limit(P, mode, D, days) for (_, P, mode, D, days) in SCENARIOS]
for (name, P, mode, D, days), g in zip(SCENARIOS, p1_got):
    t_days = 6 * P * 1e9 * D * 1e9 / 200e12 / 86400 if mode == "train" else 0
    print(f"{name:45s} memory {train_gb(P) if mode == 'train' else weights_gb(P, 16):7.1f} GB   one-GPU time {t_days:9.2f} days  ->  {g}")
assert p1_got == ["both", "memory", "time", "neither"]

# %% [markdown]
# The answers are **both**, **memory**, **time** and **neither**. Notice scenario 3: a small model can still be too slow, and scenario 2: a model can be fine in time and impossible in memory. The next parts attack each limit with a different tool. **Data parallelism** (Part 2) attacks time when the model fits; **tensor and pipeline parallelism** (Part 3) attack memory when it does not; **mixed precision and quantization** (Parts 4 and 5) shrink the bytes; **batching and the KV cache** (Part 6) decide what a server can hold.

# %% [markdown]
# ## Part 2. Data parallelism by hand and for real (30 min)
#
# *Lesson section: "Data parallelism".* The company's order book is too big for one truck to finish in time (the time limit of Part 1). So it buys **K identical trucks**, gives each a copy of the same plan (the model) and a **slice of the day's orders** (the batch). Each truck works out how the plan should change, they meet at the end of the day, **average** their notes, and everybody updates the plan in exactly the same way, so all the copies stay identical. That meeting is the **all-reduce**, and it is the price of ==data parallelism: the compute is shared, but every step must send the gradients around==. In this part you do the averaging by hand and prove it equals one big batch, then run **real** training in several processes with `torch.distributed`, and measure what the meeting costs on this machine.

# %% [markdown]
# ### ✏️ Imagine first
#
# (1) You split a batch of 32 examples into 4 shards of 8, compute the gradient of the mean loss on each shard and **average** the 4 gradients. How different is the result from the gradient on all 32 examples at once: not at all (up to rounding), a little, or a lot? (2) If the shards had **different sizes** (10, 20 and 34 examples) and you still took the plain average of the three gradients, would the result be right? (3) A model with 100 MB of gradients is averaged by 4 processes on this computer. The all-reduce moves 2(N − 1)/N × S bytes per process. Roughly how long does it take: under 1 ms, about 10 ms, about 100 ms, or about a second? Write your guesses in the block below (replace each `None`), then run it. You compare with Steps 2.1, 2.2 and 2.5.

# %%
# Imagine first: the trucks
my_guess_equal = None    # "same", "little" or "lot": equal shards, averaged gradients against the full-batch gradient
my_guess_unequal = None  # True if the plain average of unequal shards is still right, else False
my_guess_ms = None       # milliseconds for an all-reduce of 100 MB between 4 processes on this computer (a number)

# %% [markdown]
# ### Step 2.1. Split the batch, average the gradients
#
# We use a small regression network (8 inputs, 16 hidden units, 1 output) and a batch of 32 examples. `torch.autograd.grad(loss, params)` returns the gradient of a loss with respect to the parameters **without** storing it in `p.grad`, which is exactly what a shard needs. We compute the gradient of the mean squared error on the **whole batch**, then on each of K = 4 shards of 8 examples, average the four gradients and compare. The loss is a **mean** over the examples, so a gradient on a shard is the average of 8 per-example gradients, and the average of four such averages over equal shards is the average over all 32.

# %%
def p2_make(seed=SEED):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 1))

def p2_grads(model, X, Y):
    """Gradient of the mean squared error of (X, Y) with respect to every parameter, as a list of tensors."""
    params = list(model.parameters())
    return list(torch.autograd.grad(F.mse_loss(model(X), Y), params))

torch.manual_seed(SEED)
p2_X, p2_Y = torch.randn(32, 8), torch.randn(32, 1)
p2_net = p2_make()
p2_full = p2_grads(p2_net, p2_X, p2_Y)
p2_K = 4
p2_shards = [p2_grads(p2_net, p2_X[i::p2_K], p2_Y[i::p2_K]) for i in range(p2_K)]       # shard i takes every 4th example, starting at i
p2_mean = [sum(g[j] for g in p2_shards) / p2_K for j in range(len(p2_full))]
p2_maxdiff = max((a - b).abs().max().item() for a, b in zip(p2_mean, p2_full))
p2_scale = max(g.abs().max().item() for g in p2_full)
print("tensors in the gradient:", len(p2_full), "| numbers:", sum(g.numel() for g in p2_full))
print(f"largest gradient entry: {p2_scale:.4f}")
print(f"largest difference between the average of 4 shard gradients and the full-batch gradient: {p2_maxdiff:.2e}")
print("shard 0 vs shard 1 (first layer weight), first 3 entries:", p2_shards[0][0].flatten()[:3].tolist(), p2_shards[1][0].flatten()[:3].tolist())
assert p2_maxdiff < 1e-6

# %% [markdown]
# The average of the four shard gradients differs from the full-batch gradient by at most **6.0e-08**, while the gradient entries themselves are as large as 0.53: the difference is only the rounding of 32-bit numbers (adding in a different order). The individual shard gradients are quite different from one another (look at the first three entries printed above), but their **average is exact**. That is the whole idea of data parallelism: averaging gradients costs nothing in accuracy.

# %% [markdown]
# ### ✏️ Your turn: Exercise 2.1. Average the gradients of the shards
#
# Your task: write `average_grads(grad_lists, sizes=None)`. `grad_lists` is a list with one entry per shard, and each entry is a list of gradient tensors (one per parameter, as returned by `p2_grads`). Return a list with the averaged gradient of each parameter. When `sizes` (the number of examples in each shard) is given, use a **weighted** average with weights `size / sum(sizes)`; when it is `None`, give every shard the same weight. Why it matters: Step 2.2 shows that the weights are necessary when the shards are not equal.
#
# Replace the starter, run it, then run the **Check** cell.

# %%
def average_grads(grad_lists, sizes=None):
    # your code here
    return None

# %%
if average_grads(p2_shards) is None:
    print("Write average_grads above, then run this cell again.")
else:
    _a = average_grads(p2_shards)
    check("equal shards: matches the full-batch gradient", all(torch.allclose(x, y, atol=1e-6) for x, y in zip(_a, p2_full)), "add the shard gradients and divide by the number of shards")
    _g1, _g2 = p2_grads(p2_net, p2_X[:10], p2_Y[:10]), p2_grads(p2_net, p2_X[10:], p2_Y[10:])
    _w = average_grads([_g1, _g2], sizes=[10, 22])
    check("weighted: matches the full-batch gradient for shards of 10 and 22", all(torch.allclose(x, y, atol=1e-6) for x, y in zip(_w, p2_full)), "weight each shard by size / total size")
    check("returns one tensor per parameter", len(_a) == len(p2_full) and all(x.shape == y.shape for x, y in zip(_a, p2_full)), "keep the shapes of the parameters")
    RESULTS["p2_avg_grads_ok"] = True

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def average_grads(grad_lists, sizes=None):
#     K = len(grad_lists)
#     w = [1 / K] * K if sizes is None else [s / sum(sizes) for s in sizes]
#     return [sum(w[i] * grad_lists[i][j] for i in range(K)) for j in range(len(grad_lists[0]))]
# ```
#
# `j` runs over the parameters and `i` over the shards. The weights add up to 1, so the result is a weighted mean. In a real system this sum is what `all_reduce` computes (with the sum divided by the number of processes afterwards).
#
# </details>

# %% [markdown]
# Below the exercise we run our version, so that the next steps work without your answer.

# %%
def average_grads(grad_lists, sizes=None):
    K = len(grad_lists)
    w = [1 / K] * K if sizes is None else [s / sum(sizes) for s in sizes]
    return [sum(w[i] * grad_lists[i][j] for i in range(K)) for j in range(len(grad_lists[0]))]
assert all(torch.allclose(x, y, atol=1e-6) for x, y in zip(average_grads(p2_shards), p2_full))
print("average_grads works on the 4 equal shards")

# %% [markdown]
# ### Step 2.2. Unequal shards need a weighted average
#
# What if one truck gets more orders than another? Take shards of 10, 20 and 2 examples (the sizes of a batch of 32 that does not divide by 3). The plain average of the three shard gradients gives the small shard the same say as the big one, which is wrong. The weighted average with weights `size / total` is right. We measure both errors against the full-batch gradient.

# %%
p2_sizes = [10, 20, 2]
p2_edges = np.cumsum([0] + p2_sizes)
p2_uneq = [p2_grads(p2_net, p2_X[a:b], p2_Y[a:b]) for a, b in zip(p2_edges[:-1], p2_edges[1:])]
def p2_err(gl): return max((a - b).abs().max().item() for a, b in zip(gl, p2_full))
p2_err_plain, p2_err_w = p2_err(average_grads(p2_uneq)), p2_err(average_grads(p2_uneq, sizes=p2_sizes))
print("shard sizes:", p2_sizes)
print(f"plain average of the three shard gradients: largest error {p2_err_plain:.4f}")
print(f"weighted average (size / total):            largest error {p2_err_w:.2e}")
print("relative size of the plain-average error: %.0f%% of the largest gradient entry" % (100 * p2_err_plain / p2_scale))
assert p2_err_w < 1e-6 and p2_err_plain > 100 * p2_err_w

# %% [markdown]
# The plain average is wrong by **0.139**, which is 26 % of the largest gradient entry, while the weighted average is exact again (1e-07). In practice frameworks avoid the problem by giving every process the same number of examples, or by **dropping or padding** the last batch; but when you write the loss with `sum` instead of `mean`, or when sequences have different lengths, the same weighting question comes back.

# %% [markdown]
# ### Step 2.3. One step with averaged gradients equals one big-batch step
#
# The last piece: after the averaging, every copy takes **the same** SGD step, `p ← p − lr × gradient`. We take one step on a copy of the model with the full-batch gradient, and one step with the averaged shard gradients on another copy, and assert that the two models are the same afterwards. The consequence for the learning rate: with K copies each reading b examples, the **effective batch size** is K × b, so the data-parallel run is a big-batch run, and the common (heuristic, not a law) advice is to scale the learning rate with it and warm it up.

# %%
import copy
p2_lr = 0.1
p2_a, p2_b = copy.deepcopy(p2_net), copy.deepcopy(p2_net)
with torch.no_grad():
    for p, g in zip(p2_a.parameters(), p2_full): p -= p2_lr * g                           # one big-batch step
    for p, g in zip(p2_b.parameters(), average_grads(p2_shards)): p -= p2_lr * g          # one data-parallel step
p2_pd = max((x - y).abs().max().item() for x, y in zip(p2_a.parameters(), p2_b.parameters()))
p2_move = max((x - y).abs().max().item() for x, y in zip(p2_a.parameters(), p2_net.parameters()))
print(f"largest difference between the two models after one step: {p2_pd:.2e}")
print(f"largest change of any parameter during the step:           {p2_move:.2e}")
print(f"effective batch size: {p2_K} shards x 8 examples = {p2_K * 8}")
assert p2_pd < 1e-6

# %% [markdown]
# The two models differ by 1e-08, while one step moved some parameter by 0.053: they are the same model. This is why data parallelism is called **exact**: it is not an approximation of big-batch training, it *is* big-batch training, spread over several machines. What it does **not** change is that every copy holds the full weights, gradients and optimiser state (Part 1): data parallelism buys **time**, not memory.

# %% [markdown]
# ### Step 2.4. A real data-parallel run with `torch.distributed`
#
# Now the real thing: several **processes**, each with its own copy of the model, connected by `torch.distributed`. We use the **gloo** backend, which works on any CPU (on GPUs the usual backend is NCCL). `DistributedDataParallel` (DDP) wraps a model: after the backward pass it runs an all-reduce over the gradients in the background, so every process ends the step with the same averaged gradient.
#
# The code of one process is written to a small temporary file and started with `subprocess`, one process per rank, because a Jupyter kernel cannot be split into several processes. Each rank takes its own slice of the same batch, trains 5 steps with SGD, and prints a **checksum** (the sum of the absolute values of all parameters). We then train the same model on the whole batch in **this** process and compare. Two things must be true: all ranks agree with each other, and they agree with the single-process run. The cell is written defensively: if processes cannot be started on your platform, it catches the problem and says what to do.

# %%
import socket, tempfile, shutil
P2_DDP_SCRIPT = '''
import os, sys, json, time, datetime, torch, torch.nn as nn, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.set_num_threads(int(os.environ.get("THREADS", "1")))
dist.init_process_group("gloo", init_method="tcp://127.0.0.1:" + os.environ["PORT"], rank=rank, world_size=world, timeout=datetime.timedelta(seconds=60))
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(128, 1024), nn.Tanh(), nn.Linear(1024, 1024), nn.Tanh(), nn.Linear(1024, 1))
g = torch.Generator().manual_seed(1)
X = torch.randn(256, 128, generator=g); Y = torch.randn(256, 1, generator=g)
shard = slice(rank * 256 // world, (rank + 1) * 256 // world)
ddp = DDP(model)
opt = torch.optim.SGD(ddp.parameters(), lr=0.05)
def step(sync=True):
    opt.zero_grad()
    if sync: nn.functional.mse_loss(ddp(X[shard]), Y[shard]).backward()
    else:
        with ddp.no_sync(): nn.functional.mse_loss(ddp(X[shard]), Y[shard]).backward()
    opt.step()
for _ in range(5): step()
checksum = sum(float(p.detach().double().abs().sum()) for p in model.parameters())
def timeit(sync, n=20):
    step(sync); dist.barrier(); t = time.perf_counter()
    for _ in range(n): step(sync)
    dist.barrier(); return (time.perf_counter() - t) / n
t_local = timeit(False); t_sync = timeit(True)
print("RESULT " + json.dumps({"rank": rank, "checksum": checksum, "t_local": t_local, "t_sync": t_sync}))
dist.destroy_process_group()
'''
P2_AR_SCRIPT = '''
import os, json, time, datetime, torch, torch.distributed as dist
rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.set_num_threads(int(os.environ.get("THREADS", "1")))
dist.init_process_group("gloo", init_method="tcp://127.0.0.1:" + os.environ["PORT"], rank=rank, world_size=world, timeout=datetime.timedelta(seconds=60))
SIZES = [1000, 10000, 100000, 1000000, 10000000] + ([] if os.environ.get("NB_TEST") else [100000000])
res = {}
for nbytes in SIZES:
    x = torch.ones(nbytes // 4); reps = 30 if nbytes <= 1000000 else 5
    dist.all_reduce(x); dist.barrier(); t = time.perf_counter()
    for _ in range(reps): dist.all_reduce(x)
    res[nbytes] = (time.perf_counter() - t) / reps
if rank == 0: print("RESULT " + json.dumps(res))
dist.destroy_process_group()
'''

def p2_launch(script_text, world, timeout=120):
    """Run script_text in `world` processes (ranks 0..world-1) that talk through gloo on this computer. Returns the list of dictionaries that the ranks printed after 'RESULT ', or None (with a message) if something fails."""
    tmp = tempfile.mkdtemp(prefix="w18_")
    try:
        path = os.path.join(tmp, "worker.py")
        open(path, "w").write(script_text)
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
        threads = max(1, torch.get_num_threads() // world)
        procs = []
        for r in range(world):
            env = dict(os.environ, RANK=str(r), WORLD_SIZE=str(world), PORT=str(port), THREADS=str(threads), OMP_NUM_THREADS=str(threads))
            procs.append(subprocess.Popen([sys.executable, path], env=env, cwd=tmp, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        results = []
        try:
            for p in procs:
                out, err = p.communicate(timeout=timeout)
                if p.returncode != 0:
                    print("a worker failed:", err.strip().splitlines()[-1] if err.strip() else "no message"); return None
                results += [json.loads(l[7:]) for l in out.splitlines() if l.startswith("RESULT ")]
        except subprocess.TimeoutExpired:
            for p in procs: p.kill()
            print(f"the workers did not finish in {timeout} s and were stopped"); return None
        return results
    except Exception as e:
        print("could not start worker processes:", repr(e)); return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def p2_single(steps=5):
    """The same training in this process on the whole batch (what the ranks must reproduce together)."""
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(128, 1024), nn.Tanh(), nn.Linear(1024, 1024), nn.Tanh(), nn.Linear(1024, 1))
    g = torch.Generator().manual_seed(1)
    X = torch.randn(256, 128, generator=g); Y = torch.randn(256, 1, generator=g)
    o = torch.optim.SGD(m.parameters(), lr=0.05)
    def st():
        o.zero_grad(); F.mse_loss(m(X), Y).backward(); o.step()
    for _ in range(steps): st()
    return m, st

p2_ddp = {}
for W in (2, 4):
    t0 = time.perf_counter()
    r = p2_launch(P2_DDP_SCRIPT, W)
    if r is None:
        print(f"world size {W}: the multi-process run failed; see the note below the cell.")
    else:
        p2_ddp[W] = sorted(r, key=lambda d: d["rank"])
        p2_ranks = p2_ddp[W]
        p2_cs = [round(d["checksum"], 6) for d in p2_ranks]
        print(f"world size {W}: {len(r)} processes finished in {time.perf_counter() - t0:.1f} s; checksums of the ranks:", p2_cs)
p2_m1, p2_st1 = p2_single()
p2_check1 = sum(float(p.detach().double().abs().sum()) for p in p2_m1.parameters())
print(f"single process, whole batch: checksum {p2_check1:.6f}")
for W, r in p2_ddp.items():
    cs = [d["checksum"] for d in r]
    print(f"world size {W}: spread between ranks {max(cs) - min(cs):.2e}; difference from single process {abs(cs[0] - p2_check1):.2e} (relative {abs(cs[0] - p2_check1) / p2_check1:.1e})")
    assert max(cs) - min(cs) < 1e-6 * p2_check1 and abs(cs[0] - p2_check1) < 1e-5 * p2_check1

# %% [markdown]
# With 2 and with 4 processes, **all the ranks end with the same parameters** (the checksums agree), and they equal the parameters of ordinary training on the whole batch in one process, up to floating-point rounding: the relative difference is below one part in 100,000. That is what DDP guarantees, and what Step 2.1 proved on paper: averaged shard gradients are the full-batch gradient.
#
# *If this cell printed "the multi-process run failed":* on some platforms (certain Windows or notebook setups, or a firewall that blocks local ports) the processes cannot connect. Try running the cell again (a random free port is chosen each time), close other heavy programs, or run the notebook as a normal Python script. The rest of the lab does not depend on it, because the later steps use the numbers that you measured here only if they exist, and say so if they do not.

# %% [markdown]
# ### Step 2.5. How long does the meeting take? Measure the all-reduce
#
# The all-reduce is the communication of data parallelism, so we measure it. Each process holds a tensor of S bytes and `dist.all_reduce` replaces it by the **sum** over all processes. We time it for sizes from 1 KB to 100 MB with 2 and with 4 processes. In the ring algorithm each process sends and receives about **2(N − 1)/N × S** bytes (the lesson's formula), so the **effective bandwidth** is that traffic divided by the time. For tiny messages the time is dominated by a fixed **latency** (the cost of starting a message), for big ones by the bandwidth. We estimate the latency as the time of the 1 KB message and the bandwidth from the slope between the two largest sizes. We run the measurement twice and keep the faster time of each size, because other programs on your computer disturb timings.

# %%
p2_ar = {}
for W in (2, 4):
    runs = [p2_launch(P2_AR_SCRIPT, W, timeout=240) for _ in range(1 if QUICK else 2)]       # two launches; we keep the faster time of each size (less noise)
    runs = [r[0] for r in runs if r]
    if runs: p2_ar[W] = {int(k): min(float(r[k]) for r in runs) for k in runs[0]}
p2_bw, p2_alpha = {}, {}
if p2_ar:
    print(f"{'size':>9}", *[f"{'N=' + str(W) + ' time':>12}{'GB/s':>8}" for W in p2_ar])
    for nb in sorted(next(iter(p2_ar.values()))):
        row = [f"{human_bytes(nb):>9}"]
        for W in p2_ar:
            traffic = 2 * (W - 1) / W * nb
            row.append(f"{p2_ar[W][nb] * 1000:9.3f} ms{traffic / p2_ar[W][nb] / 1e9:8.2f}")
        print(*row)
    for W, d in p2_ar.items():
        sz = sorted(d); a, b = sz[-2], sz[-1]
        p2_bw[W] = 2 * (W - 1) / W * (b - a) / (d[b] - d[a])
        first = sz[0]
        p2_alpha[W] = d[first]
        print(f"N = {W}: latency about {p2_alpha[W] * 1e6:.0f} microseconds, bandwidth about {p2_bw[W] / 1e9:.1f} GB/s (from the two largest sizes)")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for W, d in p2_ar.items():
        sz = sorted(d)
        ax[0].loglog(sz, [d[s] * 1000 for s in sz], "o-", label=f"N = {W}")
        ax[1].semilogx(sz, [2 * (W - 1) / W * s / d[s] / 1e9 for s in sz], "o-", label=f"N = {W}")
    ax[0].set_xlabel("message size S (bytes)"); ax[0].set_ylabel("all-reduce time (ms)"); ax[0].set_title("flat for small S (latency), a line for large S"); ax[0].legend()
    ax[1].set_xlabel("message size S (bytes)"); ax[1].set_ylabel("effective bandwidth (GB/s)"); ax[1].set_title("small messages waste the link"); ax[1].legend()
    plt.tight_layout(); plt.show()
    assert all(p2_ar[W][max(p2_ar[W])] > p2_ar[W][min(p2_ar[W])] for W in p2_ar)
else:
    print("no all-reduce measurements: see the note below the cell")

# %% [markdown]
# On the machine that ran this lab, a message of 1 KB takes about 294 microseconds between 2 processes: that is **latency**, because the message is tiny. The time hardly changes up to 100 KB (0.29 ms for 1 KB, 0.35 ms for 100 KB) and after that it grows in proportion to the size. For the largest messages the effective bandwidth reaches about **1.8 GB/s with 2 processes** and **1.1 GB/s with 4** (the 4 processes share the same cores and memory, so they do not get faster). Two lessons. First, **many small messages are slow**: one big all-reduce beats a thousand small ones, which is why DDP groups the gradients into buckets of several megabytes. Second, these numbers are for processes that share one computer's memory; a real cluster link is faster inside a machine and much slower between machines, and its numbers have to be measured the same way. Your own numbers will differ from ours, and will vary a little from run to run: that is the point of measuring.
#
# Now we use the measured bandwidth in the lesson's formulas.

# %% [markdown]
# ### Step 2.6. Predict the cost of a step, then compare with the DDP run
#
# The lesson gives two formulas. The **traffic** per process is 2(N − 1)/N × S bytes, so the all-reduce time is about `latency + traffic / bandwidth`. The **efficiency** of a data-parallel step is compute ÷ (compute + communication), and the speed-up over one copy is N × efficiency (if the all-reduce does not overlap with the computation; real systems overlap part of it, which only helps). For the lesson's example, a 7-billion-parameter model has 14 GB of FP16 gradients; with N = 8 the traffic is 2 × 7/8 × 14 = **24.5 GB** per process, and at an assumed 100 GB/s that is **245 ms**.

# %% [markdown]
# ### ✏️ Your turn: Exercise 2.2. The data-parallel cost function
#
# Your task: write `dp_cost(Tc, S, N, bw, alpha=0.0)` that returns a dictionary with the keys `"traffic"` (bytes per process: 2(N − 1)/N × S), `"comm"` (seconds: `alpha + traffic / bw`), `"eff"` (efficiency: `Tc / (Tc + comm)`) and `"speedup"` (`N × eff`). `Tc` is the compute time of one step in seconds, `S` the size of the gradients in bytes, `bw` the bandwidth in bytes per second. Check your function on the lesson's example: Tc = 1 s, S = 14 GB, N = 8, bw = 100 GB/s.
#
# Replace the starter, run it, then run the **Check** cell.

# %%
def dp_cost(Tc, S, N, bw, alpha=0.0):
    # your code here
    return {}

# %%
_c = dp_cost(1.0, 14e9, 8, 100e9)
if not _c:
    print("Write dp_cost above, then run this cell again.")
else:
    check("traffic is 24.5 GB", abs(_c["traffic"] - 24.5e9) < 1e3, "2 * (N - 1) / N * S with N = 8 and S = 14e9")
    check("all-reduce takes 245 ms", abs(_c["comm"] - 0.245) < 1e-6, "traffic / bw")
    check("efficiency about 0.803 and speed-up about 6.42", abs(_c["eff"] - 1 / 1.245) < 1e-6 and abs(_c["speedup"] - 8 / 1.245) < 1e-5, "eff = Tc / (Tc + comm); speedup = N * eff")
    check("one process has no traffic", dp_cost(1.0, 14e9, 1, 100e9)["traffic"] == 0 and abs(dp_cost(1.0, 14e9, 1, 100e9)["speedup"] - 1) < 1e-12, "with N = 1 the formula gives 0 bytes and a speed-up of 1")
    RESULTS["p2_example_speedup"] = round(_c["speedup"], 2)

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def dp_cost(Tc, S, N, bw, alpha=0.0):
#     traffic = 2 * (N - 1) / N * S
#     comm = alpha + traffic / bw
#     eff = Tc / (Tc + comm)
#     return {"traffic": traffic, "comm": comm, "eff": eff, "speedup": N * eff}
# ```
#
# With N = 1 the traffic is 0 and, with the default `alpha = 0.0`, the communication time is 0 too: a single process does no all-reduce at all. The 245 ms on a 1 s step is a 20 % tax, which is why a fast link, or overlapping the all-reduce with the backward pass, matters.
#
# </details>

# %% [markdown]
# Below the exercise we run our version. Then we compare the prediction with the measured DDP runs of Step 2.4: the **compute** time is the time of a step without synchronisation (`no_sync`), the total is the normal step, and the difference is the communication. We predict that difference from the measured latency and bandwidth of Step 2.5, using the real gradient size of the DDP model (its parameters times 4 bytes).

# %%
def dp_cost(Tc, S, N, bw, alpha=0.0):
    traffic = 2 * (N - 1) / N * S
    comm = alpha + traffic / bw
    eff = Tc / (Tc + comm)
    return {"traffic": traffic, "comm": comm, "eff": eff, "speedup": N * eff}
RESULTS["p2_example_speedup"] = round(dp_cost(1.0, 14e9, 8, 100e9)["speedup"], 2)
assert abs(dp_cost(1.0, 14e9, 8, 100e9)["traffic"] - 24.5e9) < 1e3 and abs(dp_cost(1.0, 14e9, 8, 100e9)["comm"] - 0.245) < 1e-9

p2_S = sum(p.numel() for p in p2_m1.parameters()) * 4
p2_t_single = time_ms(p2_st1, repeat=10, warmup=2) / 1000     # one step of the whole batch in this process (all threads)
p2_rows = {}
print(f"gradient size of the DDP model: {human_bytes(p2_S)}; one step of the whole batch in one process: {p2_t_single * 1000:.1f} ms")
for W in p2_ddp:
    ranks = p2_ddp[W]
    tl = float(np.mean([d["t_local"] for d in ranks]))
    ts = float(np.mean([d["t_sync"] for d in ranks]))
    if W in p2_bw:
        pred = dp_cost(tl, p2_S, W, p2_bw[W], alpha=p2_alpha[W])
        p2_rows[W] = (tl, ts, ts - tl, pred["comm"], tl / ts, pred["eff"])
        print(f"N = {W}: compute {tl * 1000:6.2f} ms | step with all-reduce {ts * 1000:6.2f} ms | communication measured {(ts - tl) * 1000:6.2f} ms, predicted {pred['comm'] * 1000:6.2f} ms | efficiency measured {tl / ts:.2f}, predicted {pred['eff']:.2f}")
        print(f"       speed-up over one process on the whole batch: {p2_t_single / ts:.2f} x")

# %% [markdown]
# For the all-reduce of 4.73 MB the formula predicted 2.9 ms with 2 processes and 7.4 ms with 4; the measured difference between a synchronised and an unsynchronised step was 17.2 ms and 12.7 ms (a ratio of 5.9 and 1.7). The formula gives the right order of magnitude and the right trend (more processes, more communication), but it counts only the bytes on the wire: a real step also pays for splitting the gradients into buckets, for the Python side of DDP and for the processes competing for the same cores, so on this machine the measurement is above the prediction by a factor that is the **overhead** that the formula leaves out. The efficiency measured was 0.54 with 2 processes and 0.55 with 4, against 0.87 and 0.68 predicted.
#
# **And the honest result: there is no real speed-up here.** One process on the whole batch takes 9.4 ms; two processes take 37.2 ms and four 28.4 ms per step. All the "GPUs" of this experiment are the **same CPU**: the processes share the same cores (each got an equal share of the threads) and the same memory, so the total compute does not grow, while the all-reduce is extra work. On real separate GPUs each process has its own compute, and the speed-up approaches N when the communication is small, as in the next step.

# %% [markdown]
# ### Step 2.7. Speed-up against the number of GPUs, for three link speeds
#
# With your `dp_cost` function we can ask the question of the lesson: how far does data parallelism scale? Take the 7-billion-parameter example (14 GB of gradients), a compute time of 1 s per step on each GPU (assumed), and N from 1 to 64. We draw the speed-up for three **assumed** link speeds, 25, 100 and 400 GB/s, next to the ideal line, and for comparison a fourth curve with the bandwidth that **you measured** on this machine in Step 2.5. Because the traffic 2(N − 1)/N × S approaches 2 S as N grows, the communication time stops growing, so the efficiency levels off instead of falling to zero, as in Amdahl's law for a part of a program that does not get faster.

# %%
p2_Ns = np.arange(1, 65)
p2_links = {"25 GB/s (assumed, a slow link)": 25e9, "100 GB/s (assumed)": 100e9, "400 GB/s (assumed, a fast link)": 400e9}
if p2_bw: p2_links["measured on this machine (gloo, N = 2)"] = p2_bw[2]
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(p2_Ns, p2_Ns, "k--", label="ideal: N")
p2_sp = {}
for name, bw in p2_links.items():
    p2_sp[name] = np.array([dp_cost(1.0, 14e9, int(n), bw)["speedup"] for n in p2_Ns])
    ax.plot(p2_Ns, p2_sp[name], label=name)
ax.set_xlabel("number of GPUs N"); ax.set_ylabel("speed-up over 1 GPU"); ax.set_title("data parallelism: 14 GB of gradients, 1 s of compute per step")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
for name, v in p2_sp.items(): print(f"{name:42s}: speed-up at N = 8: {v[7]:5.2f}, at N = 64: {v[63]:5.2f}, efficiency at N = 64: {v[63] / 64:.2f}")
vals = list(p2_sp.values())
assert all(v[63] < 64 for v in vals) and np.all(p2_sp["25 GB/s (assumed, a slow link)"] <= p2_sp["100 GB/s (assumed)"] + 1e-12) and np.all(p2_sp["100 GB/s (assumed)"] <= p2_sp["400 GB/s (assumed, a fast link)"] + 1e-12)

# %% [markdown]
# With the assumed 400 GB/s link, 64 GPUs give a speed-up of **59.9** (efficiency 0.94); with 100 GB/s, **50.2**; with 25 GB/s, only **30.4**. At N = 8 the three links give 7.5, 6.4 and 4.0. The curves bend away from the ideal line and the slower the link the earlier they bend. The speed of the link, not the number of GPUs, sets the limit; and with the bandwidth measured on this machine the 14 GB of gradients would take far longer to average than to compute, so the speed-up at N = 8 would be only 0.55: a lesson in why a laptop's processes are not a cluster. Data parallelism is the right first tool when **the model fits on one GPU and you want speed**; when it does not fit, Part 3 splits the model itself.
#
# *Check your numbers against Step 2.6:* the speed-up formula uses the same `dp_cost` function that you wrote in Exercise 2.2. The link speeds of 25, 100 and 400 GB/s are assumptions chosen for the example, not specifications of any product.

# %% [markdown]
# ## Part 3. Tensor and pipeline parallelism (30 min)
#
# *Lesson sections: "Tensor (model) parallelism" and "Pipeline parallelism".* Back to the warehouse. Part 2 copied the whole warehouse four times and shared the orders. That fails when a single crate is too big for one warehouse, or when the whole stock does not fit in one. There are two ways out. You can **cut the crate** and let each worker carry a piece, so that the workers must meet and add up their pieces after every step: that is **tensor parallelism**. Or you can build an **assembly line**: the first warehouse does the first layers, the second warehouse the next ones, and the goods flow from one to the next: that is **pipeline parallelism**. In this part you build both by hand, on one computer, with "devices" that are only pieces of a matrix or groups of layers, and you check with numbers that the cut and the line give *exactly* the same answer as the unsplit model. Then you measure what each costs: a sum after every layer for the cut, and an idle **bubble** at the start and end for the line. ==Splitting the work never changes the answer; it changes what has to travel and who waits.== Nothing here needs a GPU: the arithmetic is the same on a real set of GPUs, but the time of the communication is not, and we say clearly where a number is assumed.

# %% [markdown]
# ### ✏️ Imagine first
#
# (1) In one feed-forward block of a Transformer that is cut over several devices the Megatron way (first matrix by columns, second by rows), how many **all-reduces** (sums across devices) does the forward pass need? (2) With the block cut over 4 devices, what share of the weights does one device hold? (3) A pipeline with 4 stages and only 4 micro-batches: what share of the time is the average stage idle? (4) The same pipeline with 16 micro-batches? Write your guesses in the block below (replace each `None`) and compare in Step 3.9.

# %%
# Imagine first: splitting the work
my_guess_allreduce = None   # all-reduces (sums across devices) in the forward pass of ONE Megatron-style feed-forward block, however many devices (a whole number)
my_guess_memory = None      # the weights of one device when a block is split over 4 devices, as a share of the unsplit weights (a number from 0 to 1)
my_guess_bubble4 = None     # idle share of a pipeline with 4 stages and 4 micro-batches (a number from 0 to 1)
my_guess_bubble16 = None    # idle share of a pipeline with 4 stages and 16 micro-batches (a number from 0 to 1)

# %%
# Helpers that this part needs (re-declared here so that Part 3 runs on its own).
if "RESULTS" not in globals(): RESULTS = {}
if "check" not in globals():
    def check(name, ok, hint=""):
        """Print a tick or a cross with a hint, so you know whether your answer is right."""
        print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))

# %% [markdown]
# ### Step 3.1. Cut a matrix product by columns
#
# A layer computes Y = X·W + b, where X holds the activations (one row per token), W is the weight matrix with d_in rows and d_out columns, and b is a bias. **Column split:** cut W into n blocks of columns, give block i to device i (with the matching piece of b), and let every device multiply the *whole* X by its block. Each device then owns a slice of the output columns, and gluing the slices side by side (a **concatenation**, not a sum) gives Y. Because the columns of the output do not depend on each other, nothing has to be added up. Run the cell: it cuts a 64 × 96 matrix over 2 and over 4 simulated devices and compares with the unsplit result.

# %%
# One matrix product Y = X @ W + b, cut by columns across n simulated devices (no GPU needed: a "device" is just a piece of the matrix).
torch.manual_seed(SEED)
p3_batch, p3_din, p3_dout = 5, 64, 96
p3_x = torch.randn(p3_batch, p3_din)
p3_W = torch.randn(p3_din, p3_dout) / 8
p3_b = torch.randn(p3_dout)
p3_full = p3_x @ p3_W + p3_b                      # the unsplit result: the reference

def p3_col_split(x, W, b, n):
    """Column split: each device holds a block of COLUMNS of W (and the matching part of b) and computes its own slice of the output."""
    Ws = torch.chunk(W, n, dim=1); bs = torch.chunk(b, n, dim=0)
    parts = [x @ w + bb for w, bb in zip(Ws, bs)]
    return torch.cat(parts, dim=-1), parts

p3_col_err = {}
for n in (2, 4):
    y, parts = p3_col_split(p3_x, p3_W, p3_b, n)
    p3_col_err[n] = (y - p3_full).abs().max().item()
    shard = torch.chunk(p3_W, n, dim=1)[0]
    print(f"{n} devices: W shard {tuple(shard.shape)} out of {tuple(p3_W.shape)}, each device's output slice {tuple(parts[0].shape)}, "
          f"max |split - full| = {p3_col_err[n]:.1e}")
print("every device needed the WHOLE input x:", tuple(p3_x.shape), "| the pieces are glued together (concatenated), nothing is added up")
assert all(e < 1e-5 for e in p3_col_err.values())

# %% [markdown]
# The split result is the same as the full result: the largest difference is **7e-07** (for 2 devices 0e+00, for 4 devices 7e-07). A column split computes every output number with the same additions in the same order as the unsplit product, so it is often *exactly* equal; a row split (next) adds the pieces in a different order, and in float32 that changes the last digits slightly (Part 4 of this lab shows why). Notice what each device needed: the **whole input** X (so X must be available on every device), and only a **block of the weights**.

# %% [markdown]
# ### ✏️ Your turn: Exercise 3.1. The row split
#
# Your task: write `row_split_forward(x, W, b, n)` for the other way of cutting. **Row split:** cut W into n blocks of **rows** and cut X into n blocks of **columns** to match (the pieces of the input that meet those rows). Device i multiplies its piece of X by its block of W and gets a **partial result** of the *full* output shape. Those partial results are not slices of the answer: they must be **added**. Return `(y, parts)`, where `parts` is the list of partial results and `y` is their sum plus the bias, with the bias added **once** (not on every device). Why it matters: the sum is the communication of tensor parallelism, an all-reduce on real GPUs, and you will now count how often it happens.
#
# Replace the starter, run it, then run the **Check** cell.

# %%
def row_split_forward(x, W, b, n):
    # your code here. Return (y, parts): y is the sum of the partial results plus the bias (added ONCE), parts is the list of the n partial results.
    return None, []

# %%
if row_split_forward(p3_x, p3_W, p3_b, 2)[0] is None:
    print("Write row_split_forward above, then run this cell again.")
else:
    for n in (2, 4, 8):
        y, parts = row_split_forward(p3_x, p3_W, p3_b, n)
        check(f"{n} devices: the sum of the partial results equals the full output", y.shape == p3_full.shape and (y - p3_full).abs().max().item() < 1e-5,
              "Cut W into n blocks of ROWS (torch.chunk(W, n, dim=0)) and x into n blocks of COLUMNS (dim=-1); add the n products; add b once at the end.")
    y, parts = row_split_forward(p3_x, p3_W, p3_b, 4)
    check("one partial result alone is NOT the answer", len(parts) == 4 and (parts[0] - p3_full).abs().max().item() > 0.5, "Each part has the full output shape but only a share of the sum.")
    RESULTS["p3_row_split"] = True

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def row_split_forward(x, W, b, n):
#     Ws = torch.chunk(W, n, dim=0)             # blocks of ROWS of W
#     xs = torch.chunk(x, n, dim=-1)            # the matching blocks of COLUMNS of x
#     parts = [xi @ w for xi, w in zip(xs, Ws)]  # each device: a partial result of the FULL output shape
#     return sum(parts) + b, parts              # the sum (an all-reduce on real GPUs), then the bias once
# ```
#
# The cut is along the shared dimension of the product: X·W is a sum over the d_in index, so cutting that index into blocks and adding the block products gives the same sum. That is why the row split ends in an addition while the column split ends in a concatenation. The mistake to avoid is the bias: if every device added b, the sum would contain it n times (the next step shows the error). On real GPUs the line `sum(parts)` is a single **all-reduce**: every device ends up with the full sum.
#
# </details>

# %% [markdown]
# ### Step 3.2. What the row split gets wrong if you are careless
#
# Here is our version of the row split, and the two classic mistakes: adding the bias on every device, and gluing the partial results together instead of adding them. Seeing the errors makes it clear why the sum is not optional.

# %%
# Row split with the reference function, and the two classic mistakes.
def p3_row_split(x, W, b, n):
    Ws = torch.chunk(W, n, dim=0); xs = torch.chunk(x, n, dim=-1)
    parts = [xi @ w for xi, w in zip(xs, Ws)]
    return sum(parts) + b, parts

y, parts = p3_row_split(p3_x, p3_W, p3_b, 4)
p3_row_err = (y - p3_full).abs().max().item()
p3_part_err = [(p - p3_full).abs().max().item() for p in parts]
p3_bias_err = ((sum(parts) + 4 * p3_b) - p3_full).abs().max().item()     # mistake: every device adds the bias
print(f"row split over 4 devices: max |sum - full| = {p3_row_err:.1e}")
print("a single partial result differs from the answer by up to", [round(e, 2) for e in p3_part_err])
print(f"mistake 1, the bias added on all 4 devices: error {p3_bias_err:.2f} (it is counted 4 times, so 3 times too many)")
print("mistake 2, gluing the partial results together instead of adding them: shape", tuple(torch.cat(parts, dim=-1).shape), "instead of", tuple(p3_full.shape))
assert p3_row_err < 1e-5 and p3_bias_err > 1

# %% [markdown]
# The row split is exact too (1e-06). One partial result alone is far from the answer (off by up to 4.1), because each device saw only a part of the sum. Adding the bias on all 4 devices makes the error **7.94**, and gluing the pieces gives a matrix of the wrong shape. The rule: **column split, then concatenate (or just keep the slices where they are); row split, then sum.**

# %% [markdown]
# ### Step 3.3. The Megatron-style feed-forward block: one sum per block
#
# A Transformer feed-forward block is Y = GELU(X·W1 + b1)·W2 + b2, with W1 of size d × 4d and W2 of size 4d × d. Megatron-LM (Shoeybi et al. 2019) cuts the first matrix by **columns** and the second by **rows**, and that choice is the clever part: the columns of the first layer's output are exactly the rows that the second matrix needs, so **each device runs both layers on its own piece without talking to the others**, and only the final partial results are added. The activation function acts on each element separately, so it can run on every shard. In the cell the class `P3Comm` plays the network: every call of `all_reduce` is counted and its bytes are added up. We cut the block over 1, 2, 4 and 8 devices, and then try the alternative (first matrix by rows) to see why it needs a second sum.

# %%
# A Megatron-style feed-forward block: first matrix by columns, activation on each shard, second matrix by rows, ONE sum at the end.
class P3Comm:
    """Counts the all-reduces (sums across devices) and the bytes of the tensors that are summed."""
    def __init__(self): self.count = 0; self.bytes = 0
    def all_reduce(self, parts):
        self.count += 1; self.bytes += parts[0].numel() * parts[0].element_size()
        return sum(parts)

torch.manual_seed(SEED)
p3_d, p3_h = 64, 256                                     # width d and hidden size h = 4d, as in a real Transformer
p3_W1 = torch.randn(p3_d, p3_h) / 8; p3_b1 = torch.randn(p3_h) * 0.1
p3_W2 = torch.randn(p3_h, p3_d) / 16; p3_b2 = torch.randn(p3_d) * 0.1
p3_xf = torch.randn(4, 10, p3_d)                         # batch 4, 10 tokens
def p3_ffn_full(x): return F.gelu(x @ p3_W1 + p3_b1) @ p3_W2 + p3_b2

def p3_ffn_tp(x, n, comm):
    W1s = torch.chunk(p3_W1, n, dim=1); b1s = torch.chunk(p3_b1, n, dim=0); W2s = torch.chunk(p3_W2, n, dim=0)
    parts = [F.gelu(x @ w1 + bb1) @ w2 for w1, bb1, w2 in zip(W1s, b1s, W2s)]   # every device works alone: no communication in here
    return comm.all_reduce(parts) + p3_b2                                          # the only communication of the block

p3_ref = p3_ffn_full(p3_xf)
p3_ffn_rows = []
for n in (1, 2, 4, 8):
    comm = P3Comm()
    err = (p3_ffn_tp(p3_xf, n, comm) - p3_ref).abs().max().item()
    p3_ffn_rows.append((n, comm.count, comm.bytes, err))
    print(f"{n} devices: all-reduces per block = {comm.count}, bytes summed = {comm.bytes}, max error = {err:.1e}")
assert all(r[1] == 1 and r[3] < 1e-5 for r in p3_ffn_rows)

# What if the first matrix were split by ROWS? The partial results must be added BEFORE the activation, because gelu(a + b) is not gelu(a) + gelu(b).
n = 4
pre = [xi @ w for xi, w in zip(torch.chunk(p3_xf, n, dim=-1), torch.chunk(p3_W1, n, dim=0))]
wrong = sum(F.gelu(p + p3_b1 / n) for p in pre)                        # activation on each partial result, then add: wrong
right = F.gelu(sum(pre) + p3_b1)                                       # add first (an all-reduce), then the activation
p3_gelu_wrong = (wrong - F.gelu(p3_xf @ p3_W1 + p3_b1)).abs().max().item()
p3_gelu_right = (right - F.gelu(p3_xf @ p3_W1 + p3_b1)).abs().max().item()
print(f"first matrix split by rows: activation before the sum: error {p3_gelu_wrong:.2f}; sum first, then activation: error {p3_gelu_right:.1e} (but that costs a second all-reduce)")
assert p3_gelu_wrong > 0.05 and p3_gelu_right < 1e-5

# %% [markdown]
# For every number of devices the block needs **1 all-reduce** and the answer is right to 1e-06. The tensor that is summed has 2560 numbers (10240 bytes): batch 4 × 10 tokens × d = 64 values of 4 bytes. The alternative is worse in two ways. If the first matrix were cut by rows, its partial results would have to be added **before** the GELU, because GELU of a sum is not the sum of the GELUs: doing it the wrong way gives an error of **1.46**, doing it the right way works (1e-06) but costs a **second** all-reduce. Cutting columns first and rows second is the cheapest order: one sum per block.

# %% [markdown]
# ### ✏️ Your turn: Exercise 3.2. How big is one all-reduce?
#
# Your task: write `allreduce_bytes(batch, tokens, d, bytes_per_value)`, the size in bytes of the activation tensor that one all-reduce adds up (batch × tokens × d values). Then compute the size in megabytes for the lesson's example (batch 8, 2,048 tokens, d = 4,096, 16-bit values) and the number of all-reduces in one forward pass of a 32-layer model that needs 2 per layer. Why it matters: this tensor travels at every layer, so the speed of the link decides whether tensor parallelism pays off.
#
# Replace the starter, run it, then run the **Check** cell.

# %%
def allreduce_bytes(batch, tokens, d, bytes_per_value):
    # your code here: the size in bytes of the tensor that one all-reduce of the activations sums
    return None

p3_lesson_mb = None     # the size in megabytes (1 MB = 1,000,000 bytes) for batch 8, 2,048 tokens, d = 4,096, 16-bit values (2 bytes), rounded to a whole number
p3_per_forward = None   # the number of all-reduces in one forward pass of a 32-layer model, with 2 all-reduces per layer (one for attention, one for the feed-forward block)

# %%
if allreduce_bytes(1, 1, 1, 1) is None or p3_lesson_mb is None or p3_per_forward is None:
    print("Write allreduce_bytes and fill in the two variables above, then run this cell again.")
else:
    check("the demo tensor of the FFN above", allreduce_bytes(4, 10, p3_d, 4) == p3_ffn_rows[1][2], "batch x tokens x d x bytes per value: 4 x 10 x 64 x 4.")
    check("the lesson example is 134 MB", p3_lesson_mb == 134, "8 x 2048 x 4096 x 2 bytes, divided by 1,000,000.")
    check("64 all-reduces per forward pass", p3_per_forward == 64, "2 per layer, 32 layers.")
    RESULTS["p3_allreduce_mb"] = p3_lesson_mb

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def allreduce_bytes(batch, tokens, d, bytes_per_value):
#     return batch * tokens * d * bytes_per_value
#
# p3_lesson_mb = round(allreduce_bytes(8, 2048, 4096, 2) / 1e6)      # 134
# p3_per_forward = 2 * 32                                            # 64
# ```
#
# The size is batch × tokens × d × bytes per value: 8 × 2,048 × 4,096 × 2 = 134,217,728 bytes, about **134 MB**, and it does not depend on the number of devices (it is the size of the activations, not of the weights). A 32-layer model has 2 sums per layer (one after attention and one after the feed-forward block), so **64 all-reduces** in every forward pass. Step 3.5 puts a time on them.
#
# </details>

# %% [markdown]
# ### Step 3.4. Attention heads across devices
#
# Attention is already built from independent **heads**: each head has its own slice of the query, key and value matrices and attends on its own. So the natural cut gives each device a few heads: the columns of Wq, Wk and Wv that belong to those heads (a column split), and the matching rows of the output matrix Wo (a row split). Each device computes its heads and its partial result of the output projection; one sum finishes the layer. The cell writes a causal multi-head attention with 8 heads from scratch and runs it on 1, 2, 4 and 8 devices.

# %%
# Attention heads split across devices: each device owns some heads (columns of Wq, Wk, Wv) and the matching rows of Wo; one sum at the end.
torch.manual_seed(SEED)
p3_H, p3_T, p3_B = 8, 12, 2
p3_dh = p3_d // p3_H
p3_Wq, p3_Wk, p3_Wv, p3_Wo = [torch.randn(p3_d, p3_d) / 8 for _ in range(4)]
p3_xa = torch.randn(p3_B, p3_T, p3_d)
p3_mask = torch.triu(torch.ones(p3_T, p3_T, dtype=torch.bool), diagonal=1)      # causal mask: a token never looks ahead

def p3_heads(x, Wq, Wk, Wv, nh):
    """Attention of nh heads (the matrices hold exactly those heads); returns the concatenated head outputs."""
    B, T, _ = x.shape
    q, k, v = [(x @ W).view(B, T, nh, p3_dh).transpose(1, 2) for W in (Wq, Wk, Wv)]
    att = (q @ k.transpose(-1, -2) / math.sqrt(p3_dh)).masked_fill(p3_mask, float("-inf")).softmax(-1)
    return (att @ v).transpose(1, 2).reshape(B, T, nh * p3_dh)

def p3_attn_full(x): return p3_heads(x, p3_Wq, p3_Wk, p3_Wv, p3_H) @ p3_Wo

def p3_attn_tp(x, n, comm):
    cols = [torch.chunk(W, n, dim=1) for W in (p3_Wq, p3_Wk, p3_Wv)]              # each device gets H/n heads
    rows = torch.chunk(p3_Wo, n, dim=0)
    parts = [p3_heads(x, cols[0][i], cols[1][i], cols[2][i], p3_H // n) @ rows[i] for i in range(n)]
    return comm.all_reduce(parts)

p3_aref = p3_attn_full(p3_xa)
p3_attn_rows = []
for n in (1, 2, 4, 8):
    comm = P3Comm()
    err = (p3_attn_tp(p3_xa, n, comm) - p3_aref).abs().max().item()
    p3_attn_rows.append((n, p3_H // n, comm.count, err))
    print(f"{n} devices: {p3_H // n} heads on each, all-reduces = {comm.count}, max error = {err:.1e}")
assert all(r[2] == 1 and r[3] < 1e-5 for r in p3_attn_rows)
print("a whole Transformer layer (attention + feed-forward) therefore needs 2 all-reduces in the forward pass")

# %% [markdown]
# Every split gives the same output as the unsplit attention (largest error 6e-07) with **one all-reduce**, so a whole layer, attention plus feed-forward, costs 2 sums in the forward pass. The number of devices cannot exceed the number of heads (8 here) with this cut: that is a real limit of the method.

# %% [markdown]
# ### Step 3.5. Memory per device, and what travels
#
# The benefit is memory. The weights of a block are cut into n parts, so each device holds about 1/n of them. The cell **measures** it: it takes the shards of device 0 (attention and feed-forward) and adds up the bytes of the tensors. Then it puts numbers on the traffic for a realistic size, with a ring all-reduce that moves 2(N − 1)/N × S bytes per device (the formula of Part 2). The two link speeds are **assumptions** for the example, not specifications of any product.

# %%
# What one device holds, measured from the shards, and what travels between devices.
def p3_device0_bytes(n):
    shards = [torch.chunk(p3_W1, n, dim=1)[0], torch.chunk(p3_b1, n, dim=0)[0], torch.chunk(p3_W2, n, dim=0)[0], p3_b2]   # b2 is added once, after the sum: kept whole
    shards += [torch.chunk(W, n, dim=1)[0] for W in (p3_Wq, p3_Wk, p3_Wv)]
    shards.append(torch.chunk(p3_Wo, n, dim=0)[0])
    return sum(t.numel() * t.element_size() for t in shards)

p3_mem = {n: p3_device0_bytes(n) for n in (1, 2, 4, 8)}
print(f"{'devices':>8} {'weights on device 0':>20} {'share of unsplit':>17} {'1/n':>6}")
for n, b in p3_mem.items():
    print(f"{n:8d} {human_bytes(b):>20} {b / p3_mem[1]:17.3f} {1 / n:6.3f}")
assert all(abs(p3_mem[n] / p3_mem[1] - 1 / n) < 0.01 for n in p3_mem)

# the traffic for a realistic size (the numbers are the lesson's example; the link speeds are ASSUMED)
p3_S = 8 * 2048 * 4096 * 2                                 # bytes of one all-reduce: batch 8, 2,048 tokens, d = 4,096, 2 bytes
p3_N = 8
p3_traffic = 2 * (p3_N - 1) / p3_N * p3_S                  # bytes each device sends and receives in a ring all-reduce
print(f"\none all-reduce: {p3_S / 1e6:.0f} MB of tensor, {p3_traffic / 1e6:.0f} MB of traffic per device (8 devices); 64 of them per forward pass of 32 layers")
p3_link = {}
for name, gbps in (("a fast link inside one machine (assumed 100 GB/s)", 100), ("an ordinary network between machines (assumed 10 GB/s)", 10)):
    p3_link[gbps] = 64 * p3_traffic / (gbps * 1e9) * 1000
    print(f"  {name}: {p3_link[gbps]:.0f} ms of communication per forward pass")

# %% [markdown]
# Device 0 holds **0.501** of the weights with 2 devices, **0.251** with 4 and **0.126** with 8: slightly above 1/n, because the bias b2 is kept whole on every device. The price is the traffic: a batch of 8 × 2,048 tokens at d = 4,096 in 16 bits is one all-reduce of 134 MB, and with 8 devices each device moves 235 MB. For 64 of them, an assumed 100 GB/s link spends about **150 ms** of every forward pass on communication, an assumed 10 GB/s network between machines about **1503 ms**. That is why tensor parallelism is used **inside one machine**, where the links are fast, and why the next method is the choice when the links are slow.

# %% [markdown]
# ### Step 3.6. Pipeline parallelism: cut by layers
#
# Now cut the model the other way: by **layers**. A model of 8 layers (each a 1,024 × 1,024 linear layer and a GELU) is divided into p **stages**, a list of `nn.Sequential`; stage 1 does the first layers, stage 2 the next ones. If a whole batch went through stage by stage, p − 1 of the p devices would wait all the time. So the batch is cut into m **micro-batches** that follow each other: as soon as stage 1 passes micro-batch 1 on, it starts micro-batch 2. The cell runs the stages one after the other on a single computer (the *order* of the arithmetic is what a pipeline does; the *overlap* in time is what the next step simulates) and checks the output against the unsplit model. Then it **measures the time of each stage** for one micro-batch, once for an equal split (2 layers each) and once for an uneven one (1, 1, 2 and 4 layers).

# %%
# Pipeline parallelism: 8 layers, cut into p stages; the batch is cut into m micro-batches that flow through the stages.
torch.manual_seed(SEED)
p3_width = 1024
p3_layers = [nn.Sequential(nn.Linear(p3_width, p3_width), nn.GELU()) for _ in range(8)]
p3_model = nn.Sequential(*p3_layers).eval()
p3_xp = torch.randn(64, p3_width)

def p3_make_stages(sizes):
    """sizes: how many layers each stage gets, for example [2, 2, 2, 2]."""
    assert sum(sizes) == len(p3_layers)
    out, i = [], 0
    for s in sizes:
        out.append(nn.Sequential(*p3_layers[i:i + s]).eval()); i += s
    return out

def p3_run_pipeline(x, stages, m):
    outs = []
    for mb in torch.chunk(x, m, dim=0):          # micro-batch after micro-batch ...
        for st in stages: mb = st(mb)            # ... through stage after stage (on real GPUs the stages work at the same time)
        outs.append(mb)
    return torch.cat(outs)

p3_pipe_err = {}
with torch.no_grad():
    p3_ref_out = p3_model(p3_xp)
    for p, m in ((1, 1), (2, 4), (4, 8), (8, 16)):
        p3_pipe_err[(p, m)] = (p3_run_pipeline(p3_xp, p3_make_stages([8 // p] * p), m) - p3_ref_out).abs().max().item()
        print(f"p = {p} stages, m = {m:2d} micro-batches of {64 // m}: max |pipeline - unsplit model| = {p3_pipe_err[(p, m)]:.1e}")
assert all(e < 1e-5 for e in p3_pipe_err.values())

# measure the time of each stage for one micro-batch of 16 rows
def p3_stage_times(sizes, rows=16):
    mb = torch.randn(rows, p3_width); ts = []
    with torch.no_grad():
        for st in p3_make_stages(sizes):
            ts.append(time_ms(lambda: st(mb), repeat=20 if not QUICK else 5, warmup=3)); mb = st(mb)
    return ts
p3_t_equal = p3_stage_times([2, 2, 2, 2])
p3_t_uneven = p3_stage_times([1, 1, 2, 4])
print("\nmeasured time per micro-batch (ms), 2 layers per stage :", [round(t, 3) for t in p3_t_equal])
print("measured time per micro-batch (ms), layers 1, 1, 2, 4  :", [round(t, 3) for t in p3_t_uneven])

# %% [markdown]
# The pipeline gives the same output as the unsplit model for every p and m (largest difference 1e-08), so cutting a batch into micro-batches loses nothing in the forward pass for a model without batch statistics (layers such as batch normalisation, which look at the whole batch, need extra care). On the machine that ran this lab one micro-batch takes about **0.57 ms** per equal stage, and for the uneven split the slowest stage takes 1.45 ms against 0.22 ms for the fastest: its four layers cost about four times as much as one layer.

# %% [markdown]
# ### Step 3.7. The GPipe schedule, the bubble, and the slowest stage
#
# Now we let the stages work **at the same time** and compute when everything finishes. The rule of the GPipe schedule: stage i can start micro-batch j when two things are true: it has finished micro-batch j − 1, and stage i − 1 has finished micro-batch j. `gpipe_schedule(stage_times, m)` follows this rule for given stage times, returns every (stage, micro-batch, start, end) and the **makespan** (the finishing time of the last micro-batch). The **idle share** is 1 − (busy time) ÷ (p × makespan): the part of the devices' time that is wasted, the **bubble**. The cell draws a Gantt chart for our two measured splits, checks the lesson's formula (p − 1) ÷ (m + p − 1) against the simulation for 18 combinations, and shows what unequal stages do. We simulate the forward pass; a training step runs the backward pass the same way, and the bubble has the same share.

# %%
# The GPipe schedule: stage i can start micro-batch j when it has finished micro-batch j - 1 AND stage i - 1 has finished micro-batch j.
def gpipe_schedule(stage_times, m):
    """Return (events, makespan). events: (stage, micro_batch, start, end)."""
    p = len(stage_times); free = [0.0] * p; done = {}; events = []
    for j in range(m):
        for i in range(p):
            start = max(free[i], done.get((i - 1, j), 0.0))
            end = start + stage_times[i]
            done[(i, j)] = end; free[i] = end; events.append((i, j, start, end))
    return events, max(free)

def p3_idle(stage_times, m):
    events, span = gpipe_schedule(stage_times, m)
    return 1 - sum(e[3] - e[2] for e in events) / (len(stage_times) * span)

def p3_gantt(ax, stage_times, m, title):
    events, span = gpipe_schedule(stage_times, m)
    cmap = plt.get_cmap("tab10")
    for i, j, s, e in events:
        ax.barh(i, e - s, left=s, color=cmap(j % 10), edgecolor="white")
        if m <= 8: ax.text((s + e) / 2, i, str(j + 1), ha="center", va="center", color="white", fontsize=8)
    ax.set_yticks(range(len(stage_times))); ax.set_yticklabels([f"stage {i + 1}" for i in range(len(stage_times))]); ax.invert_yaxis()
    ax.set_xlabel("time (ms)"); ax.set_title(f"{title}\nmakespan {span:.2f} ms, idle {p3_idle(stage_times, m):.0%}", fontsize=9)

p3_m = 4
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
p3_gantt(axes[0], p3_t_equal, p3_m, "measured times, 2 layers per stage")
p3_gantt(axes[1], p3_t_uneven, p3_m, "measured times, layers 1, 1, 2, 4")
plt.tight_layout(); plt.show()

# verify the bubble formula with equal stage times
p3_formula_err = 0
for p in (2, 4, 8):
    for m in (1, 2, 4, 8, 16, 32):
        sim = p3_idle([1.0] * p, m); formula = (p - 1) / (m + p - 1)
        p3_formula_err = max(p3_formula_err, abs(sim - formula))
assert p3_formula_err < 1e-12
print(f"simulated idle share vs (p - 1) / (m + p - 1), 18 combinations of p and m: largest difference {p3_formula_err:.1e}")
print(f"{'p':>3} " + " ".join(f"m={m:<5d}" for m in (1, 2, 4, 8, 16, 32)))
for p in (2, 4, 8):
    print(f"{p:3d} " + " ".join(f"{(p - 1) / (m + p - 1):7.1%}" for m in (1, 2, 4, 8, 16, 32)))

# the slowest stage sets the pace
p3_unequal = [1.0, 1.0, 1.0, 4.0]
_, p3_span_unequal = gpipe_schedule(p3_unequal, 8)
p3_span_formula = sum(p3_unequal) + (8 - 1) * max(p3_unequal)
print(f"\nstage times {p3_unequal}, m = 8: makespan {p3_span_unequal}, formula sum + (m - 1) x slowest = {p3_span_formula}, idle share {p3_idle(p3_unequal, 8):.1%} "
      f"(equal stages with the same total work: {p3_idle([1.75] * 4, 8):.1%})")
assert p3_span_unequal == p3_span_formula

# speed-up of the measured pipeline against one device that runs the micro-batches one after another
p3_seq = lambda ts, m: m * sum(ts)
p3_speed = {m: p3_seq(p3_t_equal, m) / gpipe_schedule(p3_t_equal, m)[1] for m in (1, 4, 16, 64)}
print("speed-up of 4 stages (measured stage times) over one device:", {m: round(v, 2) for m, v in p3_speed.items()}, "(the ideal is 4)")

fig, ax = plt.subplots(figsize=(5.5, 3.2))
for p in (2, 4, 8):
    ms = np.arange(1, 65)
    ax.plot(ms, [(p - 1) / (m + p - 1) for m in ms], label=f"p = {p}")
    ax.plot([1, 2, 4, 8, 16, 32], [p3_idle([1.0] * p, m) for m in (1, 2, 4, 8, 16, 32)], "k.")
ax.set_xlabel("micro-batches m"); ax.set_ylabel("idle share (the bubble)"); ax.set_title("the formula (lines) and the simulation (dots)"); ax.legend(); plt.tight_layout(); plt.show()

# %% [markdown]
# Three things to notice. First, the **formula is exact** for equal stages: the simulation and the formula differ by at most 6e-17 over all 18 combinations; with p = 4 the bubble falls from 75% (one micro-batch: no pipelining at all) to 42.9% (m = 4), 27.3% (m = 8) and 8.6% (m = 32). Second, **the slowest stage sets the pace**: with stage times 1, 1, 1 and 4 the makespan is the sum plus (m − 1) times the slowest stage, 35 time units for m = 8, and the idle share is 60% against 27.3% for four equal stages with the same total work. Balance the stages by their *time*, not by their number of layers. Third, with our measured equal stages the speed-up over one device is 2.20 with 4 micro-batches and 3.52 with 64, against an ideal of 4: more micro-batches keep the devices busy, but each micro-batch is smaller (so each matrix product is less efficient on a real GPU), and every micro-batch in flight keeps its activations in memory.

# %% [markdown]
# ### ✏️ Your turn: Exercise 3.3. The bubble as a function
#
# Your task: write `bubble(p, m)`, the idle share of a pipeline with p stages and m micro-batches of equal stage time. Store in `p3_bubble_4_8` its value for p = 4 and m = 8, and in `min_m` the **smallest number of micro-batches** that brings the idle share of a 4-stage pipeline below 10 percent. Why it matters: you choose m, and this tells you how many you need.
#
# Replace the starter, run it, then run the **Check** cell.

# %%
def bubble(p, m):
    # your code here: the idle share of a pipeline with p stages and m micro-batches, equal stage times
    return None

p3_bubble_4_8 = None    # idle share for p = 4, m = 8, as a number from 0 to 1
min_m = None            # the smallest number of micro-batches that brings the idle share of a 4-stage pipeline below 10 percent (0.10)

# %%
if bubble(2, 2) is None or p3_bubble_4_8 is None or min_m is None:
    print("Write bubble and fill in the two variables above, then run this cell again.")
else:
    check("bubble(4, 8) is 3/11", abs(bubble(4, 8) - 3 / 11) < 1e-9 and abs(p3_bubble_4_8 - 3 / 11) < 1e-9, "(p - 1) / (m + p - 1) with p = 4, m = 8.")
    check("bubble matches the simulation", all(abs(bubble(p, m) - p3_idle([1.0] * p, m)) < 1e-9 for p in (2, 3, 5) for m in (1, 3, 9)), "Compare with p3_idle([1.0] * p, m).")
    check("min_m brings it below 10 percent, and one less does not", bubble(4, min_m) < 0.10 <= bubble(4, min_m - 1), "Try m = 20, 25, 30 ... in your function.")
    RESULTS["p3_bubble"] = (round(p3_bubble_4_8, 4), min_m)

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def bubble(p, m):
#     return (p - 1) / (m + p - 1)
#
# p3_bubble_4_8 = bubble(4, 8)                                   # 3/11 = 0.2727
# min_m = next(m for m in range(1, 1000) if bubble(4, m) < 0.10)   # 28
# ```
#
# The idle share is (p − 1) ÷ (m + p − 1). For p = 4 and m = 8 it is 3 ÷ 11 = **27.3 %**. Below 10 percent we need 3 ÷ (m + 3) < 0.10, so m + 3 > 30, m > 27: the smallest whole number is **m = 28** (3 ÷ 31 = 9.7 %). Remember what the 28 costs: each micro-batch is 1/28 of the batch, and the batch must be big enough to be cut that finely.
#
# </details>

# %% [markdown]
# ### Step 3.8. Which parallelism, and why? The decision table
#
# Three kinds of parallelism, one question each: *does the model fit on one device? How fast are the links? How deep is the model?* The cell prints eight situations with an answer and a reason. Read the table once, then do the exercise below it (the exercise asks you to decide three of the eight *before* you look at the reasons).

# %%
# The decision table. Eight situations; for each, which kind of parallelism, and why.
P3_SITUATIONS = [
    ("1B-parameter model, fits on one GPU; you have 8 GPUs and want to train faster", "data", "the model fits, so copy it; only the gradients travel, once per step"),
    ("70B-parameter model (140 GB at 16 bits) on one machine with 8 GPUs and fast links", "tensor", "each layer is split over the GPUs; the per-layer all-reduces need fast links, and that is what one machine has"),
    ("a 96-layer model cut over 4 machines that are joined by an ordinary network", "pipeline", "only the activations at the stage borders cross the slow network"),
    ("a model with 400 billion parameters on hundreds of GPUs in many machines", "combination", "tensor inside a machine, pipeline across machines, data on top (often called 3D parallelism)"),
    ("one weight matrix is bigger than the memory of one GPU", "tensor", "only splitting the matrix itself helps: stages and copies still hold whole layers"),
    ("serving a 7B model to many users; the model fits on one GPU", "data", "run replicas, each with a full copy, and share the requests"),
    ("2 GPUs joined by a slow link, the model fits on each, you want throughput", "data", "one gradient exchange per step is far cheaper than an all-reduce in every layer"),
    ("a 30-layer model of 48 GB on four GPUs with 16 GB each, joined by a slow link", "pipeline", "cut by layers, 12 GB each; little traffic, which suits a slow link"),
]
for i, (s, a, r) in enumerate(P3_SITUATIONS, 1):
    print(f"{i}. {s}\n   -> {a}: {r}")

# %% [markdown]
# The pattern in the table: **data** parallelism whenever the model fits (it only exchanges gradients once per step, or nothing at all when serving); **tensor** parallelism when a layer or the model is too big for one device and the links are fast (a sum in every layer); **pipeline** parallelism when the model is deep and the links are slow (only the activations at the stage borders travel); and a **combination** when one method is not enough. The largest training runs use all three at once (Narayanan et al. 2021 describe how to combine them).

# %% [markdown]
# ### ✏️ Your turn: Exercise 3.4. Decide three situations
#
# Your task: for situations **2**, **3** and **6** of the table, write in `my_choice` one of `'data'`, `'tensor'`, `'pipeline'` or `'combination'`, and write one sentence of reasoning for each in `my_reason`. The check tells you whether your choice agrees with ours. Why it matters: the table is easy to follow and hard to apply; the skill is to ask the three questions yourself.
#
# Replace the starter, run it, then run the **Check** cell.

# %%
# Before you read our answers above in detail: decide situations 2, 3 and 6 from the table. Write 'data', 'tensor', 'pipeline' or 'combination'.
my_choice = {2: None, 3: None, 6: None}
my_reason = {2: "", 3: "", 6: ""}      # one sentence each, in your own words

# %%
if None in my_choice.values():
    print("Fill in my_choice for situations 2, 3 and 6, then run this cell again.")
else:
    for k in (2, 3, 6):
        check(f"situation {k}", my_choice[k] == P3_SITUATIONS[k - 1][1], f"Think about what has to move between devices, and how fast the link is. Our answer has a reason: {P3_SITUATIONS[k - 1][2]}.")
    RESULTS["p3_decisions"] = dict(my_choice)

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# my_choice = {2: "tensor", 3: "pipeline", 6: "data"}
# my_reason = {2: "the weights do not fit on one GPU and the 8 GPUs of one machine have fast links, so split each layer",
#              3: "the network is slow, and a pipeline only sends the activations between stages",
#              6: "the model fits on one GPU, so replicas give throughput with no communication between them"}
# ```
#
# Situation 2 (70 B parameters, 140 GB at 16 bits, 8 GPUs with fast links in one machine): **tensor**, because the weights do not fit on one GPU and the links inside a machine are fast. Situation 3 (a 96-layer model over 4 machines on an ordinary network): **pipeline**, because only the activations at the stage borders cross the slow network. Situation 6 (serving a 7 B model that fits on one GPU): **data**, run copies of the model and share the requests, with no communication between copies. Your own sentences should name the same three things: does it fit, how fast is the link, how deep is it.
#
# </details>

# %% [markdown]
# ### Step 3.9. Compare with your Imagine guesses
#
# The next cell prints the true values next to your guesses.

# %%
print(f"(1) all-reduces in one Megatron-style FFN block: {p3_ffn_rows[1][1]}   your guess: {my_guess_allreduce}")
print(f"(2) weights per device over 4 devices: {p3_mem[4] / p3_mem[1]:.3f} of the unsplit   your guess: {my_guess_memory}")
print(f"(3) idle share p = 4, m = 4: {(4 - 1) / (4 + 4 - 1):.3f}   your guess: {my_guess_bubble4}")
print(f"(4) idle share p = 4, m = 16: {(4 - 1) / (16 + 4 - 1):.3f}   your guess: {my_guess_bubble16}")
RESULTS["p3_ffn_allreduces"] = p3_ffn_rows[1][1]
RESULTS["p3_mem_share_4"] = round(p3_mem[4] / p3_mem[1], 3)
RESULTS["p3_bubble_4_4"] = round(3 / 7, 3)
RESULTS["p3_bubble_4_16"] = round(3 / 19, 3)
RESULTS["p3_speedup_m16"] = round(p3_speed[16], 2)

# %% [markdown]
# Your numbers against ours: **one** all-reduce per block, **0.25** of the weights on each of 4 devices, a bubble of **42.9%** with 4 micro-batches and **15.8%** with 16. Many people guess that the bubble with 4 micro-batches is small, because 4 stages feels like 4 times the work: the formula says that with p stages you need many more than p micro-batches. Remember the cost of each cut: tensor parallelism adds a sum in every layer and needs fast links; pipeline parallelism adds a bubble and needs balanced stages.
#
# ### What real systems add
#
# You built the idea, not the system. Real training frameworks add much more: **Megatron-LM** implements the tensor cuts you wrote (with the communication done by the GPU communication library, NCCL) and pipeline schedules that interleave forward and backward micro-batches (the **1F1B** schedule and its variants) to shrink the memory that waiting micro-batches hold; **GPipe** (Huang et al. 2019) is the pipeline schedule you simulated; **DeepSpeed ZeRO** (Rajbhandari et al. 2020) and PyTorch **FSDP** keep data parallelism but cut the optimiser state, gradients and weights across the copies; and for mixture-of-experts models there is **expert parallelism**, which sends each token to the device that holds its expert. Next, Part 4 looks at the other way to fit a model: make every number smaller.

# %% [markdown]
# ## Part 4. Mixed precision (30 min)
#
# *Lesson section: "Precision: fewer bits per number".* Back to the warehouse. Every number in a model is a box on a shelf, and the size of the box is the number of bits. A 32-bit box is roomy and precise; a 16-bit box is half the size, so you can store twice as many in the same warehouse and move twice as many on each truck. But a smaller box has two limits: it cannot hold very **big** numbers (it overflows) and it cannot tell apart very **close** numbers (it rounds), and worst of all it flattens very **small** numbers to zero. **Mixed precision** is the manager's compromise: keep one precise master copy of the weights in the big boxes, do the heavy arithmetic in small boxes, and rescale the tiny numbers so that they survive the trip. In this part you open the boxes: you decode the bits of a number, write your own rounding to any format and check it against PyTorch, watch numbers overflow and vanish, train a network with emulated 16-bit arithmetic with and without **loss scaling**, and finally measure what real 16-bit arithmetic does for speed and memory *on this machine*. ==The honest result may surprise you: on a computer without fast 16-bit hardware, 16 bits can be slower, not faster.==

# %% [markdown]
# ### ✏️ Imagine first
#
# (1) What is the largest finite number a 16-bit float (float16) can hold? (2) What does float16 store for 70,000? (3) You add 0.0001 to a float16 running total 100,000 times; the true total is 10. What total do you expect? (4) What does float16 store for the gradient 1e-8? Write your guesses in the block below (replace each `None`) and compare in Step 4.11.

# %%
# Imagine first: the numbers inside the computer
my_guess_largest = None   # the largest finite number that a 16-bit float (float16) can hold
my_guess_70000 = None     # what float16 stores for 70000 (a number, or float("inf") if you think it is not stored)
my_guess_sum = None       # add 0.0001 to a float16 running total 100,000 times (the true answer is 10): what total do you expect?
my_guess_tiny = None      # what float16 stores for the gradient 1e-8 (a number)

# %%
# Helpers that this part needs (re-declared here so that Part 4 runs on its own).
import struct
if "RESULTS" not in globals(): RESULTS = {}
if "check" not in globals():
    def check(name, ok, hint=""):
        """Print a tick or a cross with a hint, so you know whether your answer is right."""
        print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))

# %% [markdown]
# ### Step 4.1. The bits of a number
#
# A floating-point number is stored like scientific notation in binary: a **sign** bit, an **exponent** (which power of two) and a **mantissa** (the digits after the leading 1). The three formats of this part split their bits differently: **float32** has 1 + 8 + 23 bits, **float16** has 1 + 5 + 10, and **bfloat16** (brain float) has 1 + 8 + 7: it keeps the 8 exponent bits of float32 (the same range) but only 7 mantissa bits (less precision). A normal number is (−1)^sign × 2^(exponent field − bias) × (1 + mantissa ÷ 2^m), with bias = 2^(e−1) − 1 (15 for float16, 127 for the other two). The cell prints the bit fields of five numbers in the three formats; `struct` reads the bits of a float32, and NumPy and torch give the 16-bit ones.

# %%
# The bits of a number: sign, exponent, mantissa.
FORMATS = {"float32": (8, 23), "float16": (5, 10), "bfloat16": (8, 7)}      # (exponent bits, mantissa bits)

def p4_bits(x, fmt):
    """The bit pattern of x in a format, as an integer."""
    if fmt == "float32": return struct.unpack(">I", struct.pack(">f", x))[0]
    if fmt == "float16": return int(np.array(x, dtype=np.float16).view(np.uint16))
    return int(torch.tensor(x, dtype=torch.float32).bfloat16().view(torch.int16).item()) & 0xFFFF       # bfloat16

def p4_fields(bits, fmt):
    e, m = FORMATS[fmt]
    return bits >> (e + m), (bits >> m) & ((1 << e) - 1), bits & ((1 << m) - 1)

for x in (0.15625, -2.5, 1 / 3, 70000.0, 1e-8):
    print(f"x = {x!r}")
    for fmt in FORMATS:
        e, m = FORMATS[fmt]
        s, ex, ma = p4_fields(p4_bits(x, fmt), fmt)
        print(f"  {fmt:9s} {s:1b} {ex:0{e}b} {ma:0{m}b}   sign {s}, exponent field {ex:3d}, mantissa field {ma}")

# %% [markdown]
# Read the rows. For 0.15625 = 1.25 × 2^−3 the exponent field of float32 is 124 = 127 − 3 and the mantissa field starts with 01 (the .25 of 1.25). Float16 stores the same number with exponent field 12 = 15 − 3. For **70,000** float16 shows the exponent field **31** (all ones) and mantissa 0: that is the code for **infinity**, because 70,000 is above the largest float16, while bfloat16 stores it with a mantissa of only 7 bits. For **1e-8** float16 shows exponent 0 and mantissa 0: the number was **flattened to zero**, while bfloat16 keeps it with exponent field 100.

# %% [markdown]
# ### ✏️ Your turn: Exercise 4.1. Read the bits back
#
# Your task: write `decode_bits(bits, e_bits, m_bits)`, which turns an integer bit pattern back into the number it stands for, for any format. Use the formula of Step 4.1 for normal numbers; when the exponent field is 0 the number is **subnormal** (no leading 1: the value is 2^(1 − bias) × mantissa ÷ 2^m); when the exponent field is all ones it is infinity (mantissa 0) or not-a-number. Why it matters: a subnormal is how a format stretches its range downwards, at the price of fewer digits, and it is exactly where tiny gradients live.
#
# Replace the starter, run it, then run the **Check** cell.

# %%
def decode_bits(bits, e_bits, m_bits):
    # your code here: the value (a Python float) that the integer `bits` stands for in a format with e_bits exponent bits and m_bits mantissa bits.
    # Normal numbers:    (-1)^sign * 2^(exponent - bias) * (1 + mantissa / 2^m_bits),  with bias = 2^(e_bits - 1) - 1
    # Subnormal numbers (exponent field 0): (-1)^sign * 2^(1 - bias) * (mantissa / 2^m_bits).  All ones in the exponent field: inf (mantissa 0) or nan.
    return None

# %%
if decode_bits(0, 5, 10) is None:
    print("Write decode_bits above, then run this cell again.")
else:
    check("float16 0x3C00 is 1.0", decode_bits(0x3C00, 5, 10) == 1.0, "exponent field 15 = bias, mantissa 0: 2^0 x 1.")
    check("float16 0x7BFF is 65504", decode_bits(0x7BFF, 5, 10) == 65504.0, "exponent field 30, mantissa 1023: 2^15 x (1 + 1023/1024).")
    check("float16 0x0001 is 2^-24, the smallest subnormal", decode_bits(0x0001, 5, 10) == 2.0 ** -24, "exponent field 0: 2^(1 - 15) x (1 / 1024).")
    check("float16 0xC000 is -2", decode_bits(0xC000, 5, 10) == -2.0, "sign bit 1.")
    check("float16 0x7C00 is inf", decode_bits(0x7C00, 5, 10) == float("inf"), "exponent field all ones and mantissa 0.")
    check("bfloat16 0x3F80 is 1.0", decode_bits(0x3F80, 8, 7) == 1.0, "bias 127.")
    check("float32 0x3E800000 is 0.25", decode_bits(0x3E800000, 8, 23) == 0.25, "bias 127, exponent field 125.")
    _ok = all(decode_bits(p4_bits(v, f), *FORMATS[f]) == float(torch.tensor(v).to({"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[f]).item())
              for v in (0.1, -3.75, 1234.5, 6e-5, 3e-7) for f in FORMATS)
    check("decoding the bits of real numbers gives back what torch stores", _ok, "Compare with torch.tensor(v).half().item().")
    RESULTS["p4_decode"] = True

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def decode_bits(bits, e_bits, m_bits):
#     bias = 2 ** (e_bits - 1) - 1
#     sign = -1.0 if (bits >> (e_bits + m_bits)) & 1 else 1.0
#     exp = (bits >> m_bits) & ((1 << e_bits) - 1)
#     man = bits & ((1 << m_bits) - 1)
#     if exp == (1 << e_bits) - 1:
#         return sign * float("inf") if man == 0 else float("nan")
#     if exp == 0:                                   # subnormal: no hidden 1, fixed exponent
#         return sign * 2.0 ** (1 - bias) * (man / 2 ** m_bits)
#     return sign * 2.0 ** (exp - bias) * (1 + man / 2 ** m_bits)
# ```
#
# The three cases are the whole standard. A **normal** number has a hidden leading 1, so its value is 2^(exponent − bias) × (1 + mantissa ÷ 2^m). A **subnormal** has exponent field 0, no hidden 1, and the fixed exponent 1 − bias, so the numbers below the smallest normal one are evenly spaced (for float16 the spacing is 2^−24, about 6 × 10^−8, which is the smallest positive number it can hold). The all-ones exponent is reserved for infinity and not-a-number. Check 0x7BFF in float16: exponent field 30, mantissa 1023, so 2^15 × (1 + 1023 ÷ 1024) = 65,504, the largest finite float16.
#
# </details>

# %% [markdown]
# ### Step 4.2. Write the rounding yourself
#
# Storing a number in a format means **rounding** it to the nearest number the format can hold (when it is exactly halfway, to the neighbour with an even last digit). Around a value with exponent e the neighbours of a format with m mantissa bits are 2^(e − m) apart; below the smallest normal number the gap stops shrinking (the subnormals); above the largest number the result is infinity. That is all `round_to_format(x, e_bits, m_bits)` does, in a few lines of NumPy. The cell compares it with PyTorch's own conversion on 20 hand-picked values (including overflow, subnormals and halfway cases) and on 100,000 random values spread over 18 powers of ten, for float32, float16 and bfloat16.

# %%
# Our own rounding to any format with e_bits exponent bits and m_bits mantissa bits (round to nearest, ties to even).
def round_to_format(x, e_bits, m_bits):
    x = np.asarray(x, dtype=np.float64)
    bias = 2 ** (e_bits - 1) - 1
    emin, emax = 1 - bias, bias
    max_normal = (2 - 2.0 ** -m_bits) * 2.0 ** emax
    ax = np.abs(x)
    e = np.floor(np.log2(np.where(ax > 0, ax, 1.0)))     # the power of two just below |x|
    e = np.maximum(e, emin)                              # below the smallest normal number the spacing stops shrinking: these are the subnormals
    step = 2.0 ** (e - m_bits)                           # the gap between two neighbouring numbers of the format around |x|
    r = np.rint(ax / step) * step                        # np.rint rounds halves to the even neighbour, like the hardware
    r = np.where(r > max_normal, np.inf, r)              # too big for the format: infinity
    r = np.where(np.isfinite(ax), r, ax)                 # inf and nan stay as they are
    return np.copysign(r, x)

p4_torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
def p4_torch_round(x32, fmt):
    return torch.tensor(x32, dtype=torch.float32).to(p4_torch_dtype[fmt]).double().numpy()

p4_special = np.array([0.0, 1.0, -1.0, 0.1, 1 / 3, 3.14159265, 65504.0, 65519.0, 65520.0, 70000.0, -1e5, 6.1035e-5, 6.0e-5,
                       5.96e-8, 3.0e-8, 2.9e-8, 1e-8, -2.5e-5, 1e30, 123456.789], dtype=np.float32)
rng = np.random.default_rng(SEED)
p4_random = (rng.choice([-1, 1], 100_000) * 10.0 ** rng.uniform(-12, 6, 100_000)).astype(np.float32)
print(f"{'x':>12} {'ours float16':>14} {'torch float16':>14} {'ours bfloat16':>14} {'torch bfloat16':>14}")
for v in p4_special[:20]:
    print(f"{float(v):12.6g} {float(round_to_format(v, 5, 10)):14.6g} {float(p4_torch_round(v, 'float16')):14.6g} {float(round_to_format(v, 8, 7)):14.6g} {float(p4_torch_round(v, 'bfloat16')):14.6g}")
p4_same = {}
for fmt, (e, m) in FORMATS.items():
    for name, data in (("20 chosen values", p4_special), ("100,000 random values", p4_random)):
        ours = round_to_format(data.astype(np.float64), e, m); ref = p4_torch_round(data, fmt)
        p4_same[(fmt, name)] = bool(np.array_equal(ours, ref))
    print(f"{fmt:9s} identical to torch on the 20 values: {p4_same[(fmt, '20 chosen values')]}, on the 100,000 random values: {p4_same[(fmt, '100,000 random values')]}")
assert all(p4_same.values())

# %% [markdown]
# Our 12-line function gives **exactly** the same numbers as torch for all three formats, on the 20 chosen values and on all 100,000 random ones: nothing mysterious happens when a tensor is converted. Look at the table: 0.1 becomes 0.0999756 in float16 and 0.100098 in bfloat16 (the error is 0.02 % and 0.1 %); **65,519 rounds down to 65,504 but 65,520 rounds to infinity** (halfway between the largest number and the next power of two, ties go to the even neighbour, which is out of range); 3 × 10^−8 becomes 5.96 × 10^−8 while 2.9 × 10^−8 becomes **0**; and in bfloat16 the same values survive, because its exponent has the range of float32.

# %% [markdown]
# ### ✏️ Your turn: Exercise 4.2. Range and precision of a format
#
# Your task: write `fmt_info(e_bits, m_bits)` returning `(largest, smallest_normal, epsilon)`. With bias = 2^(e − 1) − 1: the largest finite number is (2 − 2^−m) × 2^bias, the smallest normal number is 2^(1 − bias), and the **machine epsilon** (the relative gap between a number and its neighbour) is 2^−m. The check compares your function with `torch.finfo`. Why it matters: these three numbers are everything you need to decide whether a value can overflow, vanish or lose digits in a format.
#
# Replace the starter, run it, then run the **Check** cell.

# %%
def fmt_info(e_bits, m_bits):
    # your code here: return (largest, smallest_normal, epsilon) for a format with e_bits exponent bits and m_bits mantissa bits.
    # bias = 2^(e_bits - 1) - 1;  largest = (2 - 2^-m_bits) * 2^bias;  smallest normal = 2^(1 - bias);  epsilon = 2^-m_bits (the relative gap between neighbours)
    return None

# %%
if fmt_info(5, 10) is None:
    print("Write fmt_info above, then run this cell again.")
else:
    L, S, E = fmt_info(5, 10)
    check("float16: largest 65,504", L == 65504.0, "(2 - 2^-10) x 2^15.")
    check("float16: smallest normal 6.1e-5", abs(S - 6.1035e-5) < 1e-8, "2^(1 - 15) = 2^-14.")
    check("float16: epsilon 9.77e-4", abs(E - 9.765625e-4) < 1e-12, "2^-10.")
    L2, S2, E2 = fmt_info(8, 7)
    check("bfloat16: epsilon 7.8e-3", abs(E2 - 0.0078125) < 1e-12, "2^-7.")
    check("bfloat16: largest is about 3.39e38 and the smallest normal about 1.18e-38", abs(L2 / 3.3895e38 - 1) < 1e-3 and abs(S2 / 1.1755e-38 - 1) < 1e-3, "Same bias 127 as float32.")
    check("agrees with torch.finfo for the three formats", all(abs(fmt_info(*FORMATS[f])[0] / torch.finfo(p4_torch_dtype[f]).max - 1) < 1e-6 and abs(fmt_info(*FORMATS[f])[2] / torch.finfo(p4_torch_dtype[f]).eps - 1) < 1e-6 for f in FORMATS), "torch.finfo(torch.float16).max and .eps.")
    RESULTS["p4_fmt_info"] = (L, round(S, 8), E)

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def fmt_info(e_bits, m_bits):
#     bias = 2 ** (e_bits - 1) - 1
#     return (2 - 2.0 ** -m_bits) * 2.0 ** bias, 2.0 ** (1 - bias), 2.0 ** -m_bits
# ```
#
# For float16 (e = 5, m = 10, bias 15) the largest number is (2 − 2^−10) × 2^15 = **65,504**, the smallest normal one is 2^−14 = **6.1 × 10^−5** and epsilon is 2^−10 = **9.77 × 10^−4**, about 3 decimal digits. For bfloat16 (e = 8, m = 7, bias 127) the largest number is about 3.39 × 10^38 and the smallest normal one about 1.18 × 10^−38 (the range of float32), but epsilon is 2^−7 = **7.8 × 10^−3**, only about 2 decimal digits. So: **float16 has more digits but a small range; bfloat16 has a huge range but fewer digits.**
#
# </details>

# %% [markdown]
# ### Step 4.3. The range and precision table
#
# The cell computes the table for the three formats with the formulas, checks it against `torch.finfo` and against the numbers of the lesson, and measures the **largest relative rounding error** on 100,000 random numbers inside each format's normal range. Rounding to nearest can be wrong by at most half the gap, so the relative error must stay below epsilon ÷ 2.

# %%
# Range and precision of the three formats.
print(f"{'format':9s} {'largest':>11} {'smallest normal':>16} {'smallest subnormal':>19} {'epsilon':>10} {'digits':>7}")
p4_info = {}
for fmt, (e, m) in FORMATS.items():
    bias = 2 ** (e - 1) - 1
    p4_info[fmt] = ((2 - 2.0 ** -m) * 2.0 ** bias, 2.0 ** (1 - bias), 2.0 ** (1 - bias - m), 2.0 ** -m)
    L, S, S2, E = p4_info[fmt]
    print(f"{fmt:9s} {L:11.4g} {S:16.4g} {S2:19.4g} {E:10.3g} {-math.log10(E):7.1f}")
    fi = torch.finfo(p4_torch_dtype[fmt]); assert abs(L / fi.max - 1) < 1e-6 and abs(S / fi.tiny - 1) < 1e-6 and abs(E / fi.eps - 1) < 1e-6
assert p4_info["float16"][0] == 65504 and abs(p4_info["float16"][1] - 6.1e-5) < 1e-6 and abs(p4_info["float16"][3] - 9.77e-4) < 1e-5 and abs(p4_info["bfloat16"][3] - 7.8e-3) < 1e-4
# the largest rounding error, relative to the value, measured on random values inside the normal range
p4_rel = {}
for fmt, (e, m) in FORMATS.items():
    L, S, _, E = p4_info[fmt]
    v = 10.0 ** rng.uniform(math.log10(S) + 0.5, min(math.log10(L) - 0.5, 30), 100_000)
    p4_rel[fmt] = float(np.max(np.abs(round_to_format(v, e, m) - v) / v))
    print(f"{fmt:9s} largest relative rounding error on 100,000 random values: {p4_rel[fmt]:.2e}   (half of epsilon: {E / 2:.2e})")
    assert p4_rel[fmt] <= E / 2 * 1.0001

# %% [markdown]
# The measured worst-case relative errors are **5.9e-08** (float32), **4.9e-04** (float16) and **3.9e-03** (bfloat16): they sit right at epsilon ÷ 2, as the theory says. In words: float16 stores a number to about 0.05 percent, bfloat16 to about 0.4 percent, float32 to 6 × 10^−6 percent. Notice the trade: from float16 to bfloat16 we give up 3 mantissa bits (8 times less precise) and gain 3 exponent bits (the largest number grows from 6.5 × 10^4 to 3.4 × 10^38).

# %% [markdown]
# ### Step 4.4. Overflow, underflow and a sum that stops growing
#
# Three small experiments show how a format fails. (1) **Overflow:** store 70,000. (2) **Underflow:** store gradients from 10^−3 down to 10^−8 in float16 and in bfloat16. (3) **A running total:** add float16(0.0001) to a float16 total 100,000 times; the total stops growing when the gap between neighbouring float16 numbers is more than twice the number you add, because the sum then rounds back to the old total. We also try the same sum in float32 and with `torch.sum` on a half-precision tensor. Finally a small table shows **loss scaling** in numbers: a gradient of 10^−8 multiplied by a scale before it is rounded and divided by the same scale after.

# %%
# Overflow and underflow.
p4_over = {f: float(round_to_format(70000.0, *FORMATS[f])) for f in FORMATS}
print("70000 stored as:", {f: v for f, v in p4_over.items()})
print(f"{'value':>9} {'float16':>12} {'rel. error':>11} {'bfloat16':>12} {'rel. error':>11}")
for v in (1e-3, 1e-5, 1e-7, 6e-8, 3e-8, 1e-8):
    h = float(round_to_format(v, 5, 10)); b = float(round_to_format(v, 8, 7))
    print(f"{v:9.0e} {h:12.4g} {abs(h - v) / v:11.1%} {b:12.4g} {abs(b - v) / v:11.1%}")

# a running total of 100,000 small numbers, in 16 bits and in 32 bits
p4_x = np.float16(1e-4)
p4_run16 = np.cumsum(np.full(100_000, p4_x, dtype=np.float16), dtype=np.float16)[-1]       # the total itself is kept in float16
p4_run32 = np.cumsum(np.full(100_000, p4_x, dtype=np.float32), dtype=np.float32)[-1]
p4_run_t = torch.full((100_000,), float(p4_x), dtype=torch.float16).sum().item()
print(f"\n100,000 additions of float16(0.0001) = {float(p4_x):.6g}: true total {100_000 * float(p4_x):.4f}")
print(f"  running total kept in float16: {float(p4_run16):.4f}   | kept in float32: {float(p4_run32):.4f}   | torch half .sum() (accumulates in 32 bits inside): {p4_run_t:.4f}")
# why it stops: at a total of 0.25 the gap between neighbouring float16 numbers is 2^-12
print(f"  the gap between float16 numbers near {float(p4_run16):.2f} is {2.0 ** (math.floor(math.log2(float(p4_run16))) - 10):.2e}; half of it is more than 0.0001, so adding 0.0001 changes nothing")
assert float(p4_run16) < 1 and abs(float(p4_run32) - 10) < 0.1 and p4_over["float16"] == float("inf") and np.isfinite(p4_over["bfloat16"])

# loss scaling in numbers: a gradient of 1e-8 survives if it is multiplied before it is rounded and divided afterwards
print(f"\n{'loss scale':>11} {'1e-8 x scale':>13} {'stored in float16':>18} {'divided back':>13} {'error':>8}")
for s in (1, 1024, 65536):
    stored = float(round_to_format(1e-8 * s, 5, 10)); back = stored / s
    print(f"{s:11d} {1e-8 * s:13.3e} {stored:18.4e} {back:13.4e} {abs(back - 1e-8) / 1e-8:8.1%}")

# %% [markdown]
# **Overflow:** 70,000 becomes **infinity** in float16 and 70,144 in bfloat16. **Underflow:** the float16 error grows from 0.1 % at 10^−5 to 19% at 10^−7 (the subnormals are coarse), the value 3 × 10^−8 is stored as 5.96 × 10^−8 (an error of almost 100 %) and **1e-8 becomes 0**; bfloat16 keeps 0.1 % of precision all the way down. **The running total** is the most instructive: the true answer is 10.00, a float16 total stops at **0.25**, float32 reaches 10.01, and `torch.sum` on a float16 tensor reaches 10.00 because it *accumulates in 32 bits inside* and only rounds the final result: the libraries already use this trick. The scale table shows loss scaling at work: without scaling the gradient 10^−8 is lost (100 % error); multiplied by 1,024 it survives with an error of 0.1 %, by 65,536 with almost none. Remember the rule that follows: **keep sums and master weights in 32 bits.**

# %% [markdown]
# ### Step 4.5. Emulating 16 bits on any computer
#
# The rounding that matters for accuracy can be reproduced without any special hardware: round every value to 16 bits and go back to float32, `x.half().float()` (or `.bfloat16().float()`). The numbers have exactly the precision and range of 16-bit values, but they still occupy 4 bytes and the arithmetic is the ordinary 32-bit one. We use this **emulation** in the next steps to study the *accuracy* of mixed precision. It cannot say anything about *speed* or *memory*: for that we need the real thing (Step 4.8). The cell checks that the emulation equals your `round_to_format`, counts the bytes and times the extra conversions.

# %%
# The "16-bit storage, fast 32-bit compute" trick: round to 16 bits and go back to float32. The rounding is real; the speed is not.
torch.manual_seed(SEED)
p4_t = torch.randn(1000, 1000) * 100
p4_emu = p4_t.half().float()
print("x.half().float() equals our round_to_format(x, 5, 10):", bool(np.array_equal(p4_emu.double().numpy(), round_to_format(p4_t.double().numpy(), 5, 10))))
print("bytes per value: float32", p4_t.element_size(), "| real half tensor", p4_t.half().element_size(), "| emulated (half().float())", p4_emu.element_size(), "<- no memory saved")
print(f"largest rounding error of the emulation: {(p4_emu - p4_t).abs().max().item():.4f} on values up to {p4_t.abs().max().item():.0f}")
p4_a, p4_b = torch.randn(512, 512), torch.randn(512, 512)
p4_t_plain = time_ms(lambda: p4_a @ p4_b, repeat=5)
p4_t_emu = time_ms(lambda: p4_a.half().float() @ p4_b.half().float(), repeat=5)
print(f"512 x 512 matrix product: plain float32 {p4_t_plain:.2f} ms, with both operands rounded to 16 bits first {p4_t_emu:.2f} ms ({p4_t_emu / p4_t_plain:.1f}x): the emulation costs time, it does not save any")

# %% [markdown]
# The emulation is **identical** to `round_to_format(x, 5, 10)`, but it keeps 4 bytes per value (a real half tensor has 2), and the two extra conversions made the matrix product 0.8 times as slow as plain float32. Emulation is a tool for studying numbers, never a speed-up.

# %% [markdown]
# ### ✏️ Your turn: Exercise 4.3. A dynamic loss scaler
#
# Your task: write the class `LossScaler`. It keeps a number `scale` (starting at 2^16 = 65,536). The method `update(found_inf)` is called after every training step: when the gradients overflowed (`found_inf` is `True`) the step is skipped, the scale is **multiplied by `backoff`** (0.5) and the count of good steps starts again from 0; when they did not, the count of good steps goes up by one, and after `interval` good steps in a row the scale is **multiplied by `growth`** (2.0) and the count restarts. Why it matters: the best scale is the largest one that does not overflow, and it changes during training, so the scaler searches for it by trial: it grows slowly and backs off fast.
#
# Replace the starter, run it, then run the **Check** cell.

# %%
class LossScaler:
    # your code here. Keep self.scale (start 2**16). update(found_inf): after an overflow multiply the scale by backoff (0.5) and restart the count of good steps;
    # after `interval` good steps in a row multiply the scale by growth (2.0) and restart the count.
    def __init__(self, init_scale=2.0 ** 16, growth=2.0, backoff=0.5, interval=50):
        pass
    def update(self, found_inf):
        pass

# %%
try:
    _s = LossScaler(init_scale=1024.0, interval=3)
    _ok = hasattr(_s, "scale")
except Exception:
    _ok = False
if not _ok:
    print("Write the class LossScaler above, then run this cell again.")
else:
    _s = LossScaler(init_scale=1024.0, interval=3)
    _s.update(True);  a = _s.scale
    for _ in range(2): _s.update(False)
    b = _s.scale
    _s.update(False); c = _s.scale
    _s.update(True); _s.update(False); _s.update(False); d = _s.scale
    check("an overflow halves the scale", a == 512.0, "scale *= backoff when found_inf is True.")
    check("two good steps change nothing (interval 3)", b == 512.0, "Count the good steps in a row.")
    check("the third good step doubles the scale", c == 1024.0, "scale *= growth after `interval` good steps, then restart the count.")
    check("an overflow restarts the count of good steps", d == 512.0, "After the overflow only 2 good steps follow: no growth yet.")
    RESULTS["p4_scaler"] = True

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# class LossScaler:
#     def __init__(self, init_scale=2.0 ** 16, growth=2.0, backoff=0.5, interval=50):
#         self.scale, self.growth, self.backoff, self.interval, self.good = init_scale, growth, backoff, interval, 0
#     def update(self, found_inf):
#         if found_inf:
#             self.scale *= self.backoff; self.good = 0
#         else:
#             self.good += 1
#             if self.good == self.interval:
#                 self.scale *= self.growth; self.good = 0
# ```
#
# The scaler is a tiny feedback controller. Overflow is dangerous (an infinity in a gradient would ruin the weights), so it reacts at once: halve the scale and skip that step. Underflow is silent and mild, so it tests upwards only after `interval` quiet steps. Over time the scale settles near the largest power of two that the gradients can stand. This is the same idea as the scaler of PyTorch's automatic mixed precision, written in a dozen lines.
#
# </details>

# %% [markdown]
# ### Step 4.6. Mixed-precision training from scratch: which gradients vanish?
#
# Now a network. It has 6 hidden layers of 64 units and a regression task with 128 outputs, and the loss is the **mean** over 4,096 samples and 128 outputs, which makes every gradient small (each output error is divided by about half a million). The master weights stay in float32. For the forward and backward pass the weights, the inputs, every layer output and **every gradient on its way back** are rounded to the 16-bit format by a custom autograd function (the emulation of Step 4.5). The first experiment takes the starting weights and computes the gradients once in plain float32 and once through float16 or bfloat16 copies, with different **loss scales**: the loss is multiplied by the scale before the backward pass and the gradients are divided by it after. We count the gradient entries that are non-zero in float32 but **zero** after the 16-bit pass (flushed to zero), and the entries that became infinite or not-a-number (overflow).

# %%
# Mixed-precision training from scratch. A deep regression network; the master weights stay float32; forward and backward run on "16-bit" copies (rounded with x.to(dtype).float()).
class P4Round(torch.autograd.Function):
    """Rounds the values on the way forward AND the gradients on the way back to a 16-bit format: this is what 16-bit tensors do to a backward pass."""
    @staticmethod
    def forward(ctx, x, dt): ctx.dt = dt; return x.to(dt).float()
    @staticmethod
    def backward(ctx, g): return g.to(ctx.dt).float(), None

P4_N, P4_IN, P4_W, P4_OUT, P4_DEPTH = 4096, 16, 64, 128, 6
_g = torch.Generator().manual_seed(SEED)
P4_X = torch.randn(P4_N, P4_IN, generator=_g)
P4_TS = 0.3                                   # the size of the targets
P4_Y = torch.tanh(P4_X @ (torch.randn(P4_IN, P4_OUT, generator=_g) / 4)) * P4_TS

def p4_init(seed=SEED):
    torch.manual_seed(seed)
    dims = [P4_IN] + [P4_W] * P4_DEPTH + [P4_OUT]
    ws = [(torch.randn(a, b) * (0.8 * math.sqrt(2 / a))).requires_grad_() for a, b in zip(dims[:-1], dims[1:])]
    outs = dims[1:]
    bs = [torch.zeros(b, requires_grad=True) for b in outs]
    return ws + bs

def p4_forward(params, x, dt=None):
    """dt None: plain float32. Otherwise every weight, input and layer output is rounded to dt (forward) and every gradient too (backward)."""
    ws, bs = params[:len(params) // 2], params[len(params) // 2:]
    r = (lambda t: P4Round.apply(t, dt)) if dt is not None else (lambda t: t)
    h = r(x)
    for i, (w, b) in enumerate(zip(ws, bs)):
        h = r(h @ r(w) + r(b))
        if i < len(ws) - 1: h = torch.relu(h)
    return h

def p4_grads(params, dt, scale):
    """Gradients of the master weights computed through the 16-bit copies with a loss scale, and divided by the scale afterwards."""
    for p in params: p.grad = None
    loss = ((p4_forward(params, P4_X, dt) - P4_Y) ** 2).mean()
    (loss * scale).backward()
    return [p.grad / scale for p in params], loss.item()

# gradients at the starting point: how many that are not zero in float32 become zero in float16 (and how many overflow)?
p4_params0 = p4_init()
p4_g32, p4_loss0 = p4_grads(p4_params0, None, 1.0)
p4_nz = sum(int((g != 0).sum()) for g in p4_g32)
p4_flush = {}
print(f"loss at the start: {p4_loss0:.4f}; gradient entries that are not zero in float32: {p4_nz:,}")
print(f"{'format':9s} {'loss scale':>10} {'flushed to zero':>16} {'overflow (inf/nan)':>19}")
for dt, name in ((torch.float16, "float16"), (torch.bfloat16, "bfloat16")):
    for s in (1.0, 1024.0, 65536.0, 2.0 ** 24):
        g, _ = p4_grads(p4_params0, dt, s)
        bad = sum(int((~torch.isfinite(t)).sum()) for t in g)
        flushed = sum(int(((a != 0) & (b == 0)).sum()) for a, b in zip(p4_g32, g))
        p4_flush[(name, s)] = (flushed / p4_nz, bad)
        print(f"{name:9s} {s:10.0f} {flushed / p4_nz:16.2%} {bad:19,d}")

# %% [markdown]
# At the starting point 29,203 gradient entries are non-zero. With float16 and **no scaling 0.64%** of them are flushed to zero; a scale of 1,024 brings it to 0.00%. A scale that is **too big** causes the opposite failure: with 2^24 there are 5,155 overflowed entries (infinity or not-a-number), which would destroy the weights. So the scale must be neither too small nor too large, which is why it is adjusted during training. **bfloat16 needs none of this:** it flushes 0.00% at scale 1 and never overflows, because its range is that of float32.

# %% [markdown]
# ### Step 4.7. Training runs: scaling or not?
#
# Five settings train the same network for 200 steps with Adam (the Adam update runs on the float32 master weights): plain float32; float16 without scaling; float16 with a fixed scale of 1,024; float16 with the dynamic scaler (starting at 2^16, with `interval = 25`: an overflow skips the step and halves the scale); and bfloat16 without scaling. Every setting is trained from 3 different starting weights; the table gives the mean final loss, its range, the mean number of skipped steps and the mean share of gradient entries flushed to zero (measured every 25 steps against float32 gradients at the same weights). This cell is the slowest of the part: it takes about a minute on a laptop CPU.

# %%
# Training runs: float32, float16 without scaling, float16 with a fixed scale, float16 with the dynamic LossScaler, bfloat16 without scaling.
class P4Scaler:
    """The same dynamic loss scaler as in Exercise 4.3 (copied here so that this cell runs on its own)."""
    def __init__(self, init_scale=2.0 ** 16, growth=2.0, backoff=0.5, interval=50):
        self.scale, self.growth, self.backoff, self.interval, self.good = init_scale, growth, backoff, interval, 0
    def update(self, found_inf):
        if found_inf: self.scale *= self.backoff; self.good = 0
        else:
            self.good += 1
            if self.good == self.interval: self.scale *= self.growth; self.good = 0

def p4_eval(params):
    with torch.no_grad(): return ((p4_forward(params, P4_X) - P4_Y) ** 2).mean().item()

def p4_train(dt, mode, steps, seed=SEED, lr=3e-3, probe_every=25):
    params = p4_init(seed)
    opt = torch.optim.Adam(params, lr=lr)
    scaler = P4Scaler(interval=25) if mode == "dynamic" else None
    scale = {"none": 1.0, "fixed": 1024.0, "dynamic": 2.0 ** 16}[mode] if dt is not None else 1.0
    curve, skipped, scales, flushed = [], 0, [], []
    for s in range(steps):
        if scaler: scale = scaler.scale
        grads, loss = p4_grads(params, dt, scale)
        found_inf = not all(bool(torch.isfinite(g).all()) for g in grads)
        if scaler: scaler.update(found_inf); scales.append(scale)
        if found_inf: skipped += 1; continue
        if dt is not None and s % probe_every == 0:
            g32, _ = p4_grads(params, None, 1.0)
            nz = sum(int((g != 0).sum()) for g in g32)
            flushed.append(sum(int(((a != 0) & (b == 0)).sum()) for a, b in zip(g32, grads)) / nz)
        for p, g in zip(params, grads): p.grad = g
        opt.step()
        if s % 20 == 0: curve.append(p4_eval(params))
    return {"loss": p4_eval(params), "curve": curve, "skipped": skipped, "scales": scales, "flushed": float(np.mean(flushed)) if flushed else 0.0}

P4_STEPS = 60 if QUICK else 200
P4_SEEDS = (SEED,) if QUICK else (SEED, SEED + 1, SEED + 2)        # three different starting weights for every setting
P4_SETTINGS = (("float32", None, "none"), ("float16, no scaling", torch.float16, "none"), ("float16, fixed scale 1024", torch.float16, "fixed"),
               ("float16, dynamic scale", torch.float16, "dynamic"), ("bfloat16, no scaling", torch.bfloat16, "none"))
p4_all = {name: [p4_train(dt, mode, P4_STEPS, seed=sd) for sd in P4_SEEDS] for name, dt, mode in P4_SETTINGS}
p4_runs = {name: rs[0] for name, rs in p4_all.items()}              # the first seed, for the plot
p4_summary = {name: {"loss": float(np.mean([r["loss"] for r in rs])), "lo": min(r["loss"] for r in rs), "hi": max(r["loss"] for r in rs),
                     "skipped": float(np.mean([r["skipped"] for r in rs])), "flushed": float(np.mean([r["flushed"] for r in rs]))} for name, rs in p4_all.items()}
print(f"{'run (mean of ' + str(len(P4_SEEDS)) + ' starts)':28s} {'final loss':>11} {'range':>19} {'skipped steps':>14} {'gradients flushed to 0':>23}")
for name, r in p4_summary.items():
    print(f"{name:28s} {r['loss']:11.5f} {r['lo']:9.5f}-{r['hi']:.5f} {r['skipped']:14.1f} {r['flushed']:23.2%}")
p4_dyn = p4_runs["float16, dynamic scale"]
print("dynamic scale at steps 0, 1, 2, 3, 30, 60, last (first start):", [p4_dyn["scales"][min(i, len(p4_dyn["scales"]) - 1)] for i in (0, 1, 2, 3, 30, 60, len(p4_dyn["scales"]) - 1)])
fig, ax = plt.subplots(figsize=(6, 3.4))
for name, r in p4_runs.items(): ax.plot(np.arange(len(r["curve"])) * 20, r["curve"], label=name)
ax.set_yscale("log"); ax.set_xlabel("optimiser step"); ax.set_ylabel("loss (float32 evaluation)"); ax.set_title("mixed-precision training, emulated 16 bits"); ax.legend(fontsize=7); plt.tight_layout(); plt.show()

# %% [markdown]
# Read this table honestly. **The final loss is almost the same in every setting**: the means lie within 8% of each other, and the range over different starts (for float32: 0.00585 to 0.00916) is as wide as the differences between settings. On this small problem, with Adam, loss scaling did **not** improve the final loss. What it changed is the gradients: without scaling **6.2%** of the non-zero gradient entries were flushed to zero (more than at the start, because the errors, and so the gradients, shrink as the network learns), with a fixed scale of 1,024 only 0.04% and with the dynamic scaler 0.01%. Adam divides each gradient by its own running size, which hides much of the damage here; with plain SGD, with larger networks or with smaller losses the zeros would matter more, and the usual practice is to scale. The dynamic scaler skipped 1 step(s) on average and then raised its scale, from 65,536 to 4,194,304 in the first run, because the gradients kept shrinking. **bfloat16** without any scaling trained as well as the others (mean final loss 0.00770 against 0.00719 for float32), with only 0.10% flushed: that is why modern training prefers it when the hardware has it.

# %% [markdown]
# ### Step 4.8. The real thing on this computer: autocast
#
# Emulation told us about accuracy. Now we measure **speed and memory** with real 16-bit arithmetic. `torch.autocast` runs the matrix products and the linear layers in 16 bits and keeps the sensitive operations (sums, normalisation, softmax, losses) in 32 bits. The model is a small Transformer decoder of the Part 1 type (4 layers, d = 256, 4,000 tokens in the vocabulary, 5.2 M parameters). The cell (1) measures the **activation memory** with forward hooks that add up the bytes of every layer's output, in float32 and under autocast; (2) times one complete training step (forward, backward, AdamW) in float32 and with autocast bfloat16; and (3) times a plain 512 × 512 matrix product in float32, bfloat16 and float16. The Setup cell already measured the bfloat16 ratio of this computer: it was 232 times slower than float32 when the notebook started.

# %%
# Real autocast: the speed and the activation memory of a small Transformer decoder (the Part 1 style model) in float32 and under torch.autocast.
class P4Tiny(nn.Module):
    def __init__(self, vocab=4000, d=256, layers=4, heads=4, T=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, d); self.pos = nn.Parameter(torch.zeros(T, d))
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True, activation="gelu") for _ in range(layers)])
        self.norm = nn.LayerNorm(d); self.head = nn.Linear(d, vocab)
        self.register_buffer("mask", torch.triu(torch.ones(T, T, dtype=torch.bool), 1))
    def forward(self, idx):
        n = idx.shape[1]
        h = self.emb(idx) + self.pos[:n]
        for b in self.blocks: h = b(h, src_mask=self.mask[:n, :n])
        return self.head(self.norm(h))

def p4_activation_bytes(model, idx, ctx=None):
    """Sum of the bytes of the output of every leaf module in one forward pass (a forward hook on each)."""
    total = [0]; hooks = []
    def hook(m, i, o):
        for t in (o if isinstance(o, tuple) else (o,)):
            if torch.is_tensor(t): total[0] += t.numel() * t.element_size()
    for m in model.modules():
        if len(list(m.children())) == 0: hooks.append(m.register_forward_hook(hook))
    try:
        with (ctx() if ctx else contextlib_nullcontext()): model(idx)
    finally:
        for h in hooks: h.remove()
    return total[0]
import contextlib
contextlib_nullcontext = contextlib.nullcontext

torch.manual_seed(SEED)
p4_model = P4Tiny().to(device)
p4_idx = torch.randint(0, 4000, (8, 64), device=device)
p4_tgt = torch.randint(0, 4000, (8, 64), device=device)
p4_opt = torch.optim.AdamW(p4_model.parameters(), lr=1e-4)
print(f"model: {n_params(p4_model) / 1e6:.2f} M parameters, {human_bytes(param_bytes(p4_model))} in float32")

def p4_step(autocast_dtype=None):
    ctx = torch.autocast(device, dtype=autocast_dtype) if autocast_dtype is not None else contextlib.nullcontext()
    with ctx: loss = F.cross_entropy(p4_model(p4_idx).float().view(-1, 4000), p4_tgt.view(-1))
    p4_opt.zero_grad(); loss.backward(); p4_opt.step()

p4_act32 = p4_activation_bytes(p4_model, p4_idx)
p4_act16 = p4_activation_bytes(p4_model, p4_idx, lambda: torch.autocast(device, dtype=torch.bfloat16))
print(f"activation bytes of one forward pass (batch 8, 64 tokens): float32 {human_bytes(p4_act32)}, under autocast bfloat16 {human_bytes(p4_act16)} -> {p4_act16 / p4_act32:.2f} of float32")
p4_reps = 1
p4_ms32 = time_ms(lambda: p4_step(None), repeat=p4_reps)
p4_msbf = time_ms(lambda: p4_step(torch.bfloat16), repeat=p4_reps)
print(f"one training step on {device}: float32 {p4_ms32:.0f} ms, autocast bfloat16 {p4_msbf:.0f} ms -> {p4_msbf / p4_ms32:.2f}x {'slower' if p4_msbf > p4_ms32 else 'faster'}")
p4_mm32 = time_ms(lambda: p4_a @ p4_b, repeat=3)
p4_mmbf = time_ms(lambda: p4_a.bfloat16() @ p4_b.bfloat16(), repeat=2)
p4_mmf16 = time_ms(lambda: p4_a.half() @ p4_b.half(), repeat=2)
print(f"512 x 512 matrix product: float32 {p4_mm32:.2f} ms, bfloat16 {p4_mmbf:.2f} ms ({p4_mmbf / p4_mm32:.1f}x), float16 {p4_mmf16:.2f} ms ({p4_mmf16 / p4_mm32:.1f}x); the Setup measured the bfloat16 ratio {CPU_BF16_RATIO:.1f}x")

# %% [markdown]
# Two results, one good and one bad. **Memory:** under autocast the layer outputs total 20.87 MB instead of 36.50 MB, **0.57** of the float32 size: not exactly one half, because some operations (the layer norms, for example) stay in float32. **Speed:** on the machine that ran this lab the training step took 112 ms in float32 and **4337 ms** with autocast, **39 times slower**, and the plain bfloat16 matrix product was 180 times slower than float32 (float16 216 times). This processor has no fast 16-bit arithmetic: the 16-bit product is carried out by slow fallback code. It is a real result, and exactly what the lesson warned about: **mixed precision is a speed-up only on hardware built for it** (GPUs with Tensor Cores, and CPUs with native bfloat16 instructions). Your own computer may give a different ratio; if it is below 1, it has such hardware. The memory saving, in contrast, does not depend on the machine.

# %% [markdown]
# ### Step 4.9. The same on a GPU, if you have one
#
# If the notebook finds a GPU (`HAS_CUDA`), this cell times a 4,096 × 4,096 matrix product and a training step of the same model in float32, float16 and bfloat16 under `torch.autocast('cuda')`, and reads the **peak memory** with `torch.cuda.max_memory_allocated()`. Without a GPU the cell only prints a note. **The author of this lab could not run this cell on a GPU**, so there are no GPU numbers in this text: read the output of your own run, and expect the 16-bit matrix product to be several times faster than float32 on a GPU with Tensor Cores, and the peak memory of a training step to fall (activations halve; the master weights and the optimiser state stay in 32 bits).

# %%
# The same on a GPU, if there is one (this cell is skipped without a GPU; the author could not test it on a GPU, so read its output carefully).
p4_gpu = {}
if HAS_CUDA:
    torch.cuda.reset_peak_memory_stats()
    a = torch.randn(4096, 4096, device=device); b = torch.randn(4096, 4096, device=device)
    for name, dt in (("float32", torch.float32), ("float16", torch.float16), ("bfloat16", torch.bfloat16)):
        aa, bb = a.to(dt), b.to(dt)
        p4_gpu[name] = {"matmul_ms": time_ms(lambda: aa @ bb, repeat=20, warmup=3)}
    for name, dt in (("float32", None), ("float16", torch.float16), ("bfloat16", torch.bfloat16)):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        p4_gpu[name]["step_ms"] = time_ms(lambda: p4_step(dt), repeat=10, warmup=2)
        p4_gpu[name]["peak_mb"] = torch.cuda.max_memory_allocated() / 1e6
    print(f"{'format':9s} {'4096 matmul (ms)':>17} {'training step (ms)':>19} {'peak memory (MB)':>17}")
    for name, r in p4_gpu.items(): print(f"{name:9s} {r['matmul_ms']:17.2f} {r['step_ms']:19.1f} {r['peak_mb']:17.0f}")
else:
    print("No GPU here: the GPU measurement is skipped. On a GPU with Tensor Cores the 16-bit matrix product is usually several times faster than float32, and the peak memory of a training step falls; run this cell on one to see your own numbers.")

# %% [markdown]
# ### Step 4.10. The honest summary
#
# One table with what 16-bit arithmetic did on this machine: the memory it saved, the speed it cost (or gave), and the accuracy.

# %%
# The honest summary of what 16-bit arithmetic did, on THIS machine.
print(f"{'what we measured':46s} {'result':>14}")
rows = [("activation bytes under autocast / float32", f"{p4_act16 / p4_act32:.2f}"),
        ("training step time, autocast bfloat16 / float32", f"{p4_msbf / p4_ms32:.2f}x"),
        ("512 matrix product time, bfloat16 / float32", f"{p4_mmbf / p4_mm32:.1f}x"),
        ("emulated float16 training: final loss / float32 loss", f"{p4_summary['float16, dynamic scale']['loss'] / p4_summary['float32']['loss']:.3f}"),
        ("float16 without scaling: gradients flushed to zero", f"{p4_summary['float16, no scaling']['flushed']:.2%}"),
        ("bfloat16 without scaling: final loss / float32 loss", f"{p4_summary['bfloat16, no scaling']['loss'] / p4_summary['float32']['loss']:.3f}")]
for k, v in rows: print(f"{k:46s} {v:>14}")
RESULTS["p4_activation_ratio"] = round(p4_act16 / p4_act32, 2)
RESULTS["p4_autocast_time_ratio"] = round(p4_msbf / p4_ms32, 2)
RESULTS["p4_bf16_matmul_ratio"] = round(p4_mmbf / p4_mm32, 1)
RESULTS["p4_flushed_noscale"] = round(p4_summary["float16, no scaling"]["flushed"], 4)
RESULTS["p4_loss_ratio_dynamic"] = round(p4_summary["float16, dynamic scale"]["loss"] / p4_summary["float32"]["loss"], 3)

# %% [markdown]
# The summary: mixed precision **saved memory** (the activations fell to 0.57 of their size), **cost speed on this CPU** (about 39 times slower), and **did not hurt accuracy** in the emulated runs (final loss 1.019 times the float32 loss with dynamic scaling, within the spread between starts). Which of the three matters depends on your hardware, and that is the whole point of measuring before you decide. Part 5 attacks the same memory problem from another side: not 16 bits but 8, 4 or fewer, for the weights only.

# %% [markdown]
# ### Step 4.11. Compare with your Imagine guesses
#
# The next cell prints the true values next to your guesses.

# %%
print(f"(1) largest float16: {p4_info['float16'][0]:.0f}   your guess: {my_guess_largest}")
print(f"(2) float16(70000): {p4_over['float16']}   your guess: {my_guess_70000}")
print(f"(3) float16 running total of 100,000 x 0.0001: {float(p4_run16):.4f} (true 10)   your guess: {my_guess_sum}")
print(f"(4) float16(1e-8): {float(round_to_format(1e-8, 5, 10))}   your guess: {my_guess_tiny}")

# %% [markdown]
# Your numbers against ours: the largest float16 is **65,504**; 70,000 becomes **infinity**; the float16 running total sticks at **0.25** instead of 10 (most people guess "about 10, a little off": the failure is total, not small); and the gradient 1e-8 becomes **zero**. Notice that the first three are about *range and gaps*, and the fourth is the reason why loss scaling exists.

# %% [markdown]
# ## Part 5. Quantization (30 min)
#
# *Lesson section: "Quantization".* A weight in a trained model is a 32-bit number, but the model does not need that many digits: if every weight is stored as a small **integer** (8 bits, 4 bits) and one **scale** number per group of weights, the model is four to eight times smaller and, because decoding is limited by how fast the weights can be read from memory, it can also run faster. The price is a rounding error. ==In this part you write symmetric quantization yourself, measure how the error grows when the bits shrink and when a few outlier weights stretch the scale, quantize a trained classifier, and then use PyTorch's own dynamic quantization to measure the real size and the real speed on this machine.== Everything is measured on small models that run in seconds; the memory arithmetic for a 7-billion and a 70-billion-parameter model is exact and does not depend on the machine.

# %% [markdown]
# ### ✏️ Imagine first
#
# You keep every weight of a small trained classifier in only **8 bits**, then in **4 bits**, then in **2 bits** (one scale for each row of the weight matrix). Before you run anything, guess how the accuracy changes. Then you ask PyTorch to quantize a wide network (1024 → 4096 → 1024) to 8-bit integers, and you guess how much **smaller** the file becomes and how much **faster** the forward pass runs at batch size 1 on this machine.
#
# Write your guesses in the starter cell below (replace each `None`), then run it. Do not look ahead: you compare with the real numbers at the end of this part.

# %%
# Imagine first: what does it cost to throw digits away?
guess_acc_8bit = None        # YOUR GUESS: test accuracy with 8-bit weights, in percent of the full-precision accuracy (100 = no change)
guess_acc_4bit = None        # YOUR GUESS: the same for 4-bit weights, in percent of the full-precision accuracy
guess_acc_2bit = None        # YOUR GUESS: the same for 2-bit weights, in percent of the full-precision accuracy
guess_size_ratio = None      # YOUR GUESS: how many times smaller is the state dict of the 8-bit dynamic-quantized wide network? (a number)
guess_speedup_b1 = None      # YOUR GUESS: how many times faster is its forward pass at batch size 1 on this machine? (a number, 1 = same, 0.5 = twice as slow)

# %%
# Helpers for Part 5 (all names start with p5_ or P5_).
if "check" not in globals():
    def check(name, ok, hint=""):
        """Print a tick or a cross with a hint, so you know whether your answer is right."""
        print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))
if "RESULTS" not in globals():
    RESULTS = {}      # a place to keep your key numbers; the last part of the lab prints them together
import copy
warnings.filterwarnings("ignore", message=".*quantiz.*")      # quantize_dynamic prints a deprecation notice in newer PyTorch (see Step 5.5)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def p5_dequantize(q, scales):
    """Turn integers and scales back into approximate weights: w is about q times scale. `scales` has one column per group in each row."""
    R, C = q.shape; ns = scales.shape[1]; g = C // ns
    return (q.reshape(R, ns, g) * scales[:, :, None]).reshape(R, C)
def p5_mse(a, b):
    """Mean squared difference between two arrays."""
    return float(np.mean((a - b) ** 2))
def p5_rel_err(a, b):
    """Relative error of b with respect to a: the size of the difference divided by the size of a (a tensor)."""
    return float(torch.linalg.norm(a - b) / torch.linalg.norm(a))
print("helpers ready")

# %% [markdown]
# ### Step 5.1. Symmetric quantization from scratch
#
# **Symmetric quantization** with `b` bits uses the integers from −qmax to +qmax, where `qmax = 2^(b−1) − 1` (127 for 8 bits, 7 for 4 bits, 1 for 2 bits). A **scale** says how big one integer step is: `scale = max|w| / qmax`, so that the largest weight in the group lands exactly on ±qmax. Each weight becomes `q = round(w / scale)` (an integer, clipped to the range), and we get an approximation back with `w ≈ q × scale`. The rounding error of every weight is at most half a step, `scale / 2`. A group is the set of weights that share one scale: the whole matrix (`group=None`), one row (`group="row"`) or blocks of `g` neighbours inside each row (`group=g`, for example 64).
#
# A tiny example by hand, 4 bits (qmax = 7), the weights `0.9, −0.31, 0.04, 0.5`: the scale is 0.9 / 7 = 0.1286, the integers are round(7, −2.41, 0.31, 3.89) = `7, −2, 0, 4`, and the approximate weights are `0.9, −0.257, 0, 0.514`. Notice that the small weight 0.04 vanished: it is smaller than half a step (0.064).

# %% [markdown]
# ### ✏️ Your turn: Exercise 5.1. Write the quantizer
#
# *Exercise 5.1:* Write `p5_quantize(w, bits, group=None)` for a 2-D NumPy array `w` (rows by columns). Return two things: `q`, the integer array of the same shape as `w` (use `.astype(np.int32)`), and `scales`, an array with one row per group-row and one column per group: shape `(1, 1)` when `group=None`, `(R, 1)` when `group="row"`, `(R, C // group)` when `group` is an integer. Use `qmax = 2 ** (bits - 1) - 1`, `scale = max|w| / qmax` for each group (use a scale of 1 if the group is all zeros), `q = clip(round(w / scale), -qmax, qmax)`. Why it matters: every quantization method in the lesson, from INT8 to the 4-bit formats, is a variation of these five lines. The dequantizer `p5_dequantize(q, scales)` is given. The check tests the example above, the range of the integers, the error bound `scale / 2`, and the shape of the scales.
#
# Write your answer in the starter cell below, then run the check cell under it.

# %%
def p5_quantize(w, bits, group=None):
    """Symmetric quantization: returns the integer array q and the scales (see the exercise text for the shapes)."""
    return None   # YOUR CODE: split w into groups, scale = max|group| / qmax, q = clip(round(w / scale), -qmax, qmax)

# %%
def p5_check_quantize():
    r = p5_quantize(np.array([ [0.9, -0.31, 0.04, 0.5] ]), 4)
    if r is None:
        print("Fill in your answer to Exercise 5.1 above, then run this cell again."); return
    q, s = r
    check("the example gives the integers 7, -2, 0, 4", q.ravel().tolist() == [7, -2, 0, 4], "scale = 0.9 / 7, then round(w / scale)")
    check("the integers are of integer type and in the range -7 to 7", np.issubdtype(q.dtype, np.integer) and q.min() >= -7 and q.max() <= 7, "clip to -qmax..qmax and use astype(np.int32)")
    rng_ = np.random.default_rng(1); w = rng_.normal(0, 1, (8, 32)); q8, s8 = p5_quantize(w, 8)
    check("the error of every weight is at most scale / 2", np.abs(p5_dequantize(q8, s8) - w).max() <= s8.max() / 2 + 1e-9, "round to the nearest integer")
    shapes = [p5_quantize(w, 4, g)[1].shape for g in (None, "row", 8)]
    check("the scales have shapes (1, 1), (8, 1) and (8, 4)", shapes == [(1, 1), (8, 1), (8, 4)], "one scale per group: whole matrix, row, block of 8")
    check("8 bits are more exact than 4 bits", p5_mse(w, p5_dequantize(*p5_quantize(w, 8))) < p5_mse(w, p5_dequantize(*p5_quantize(w, 4))), "more bits, smaller steps")
p5_check_quantize()

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def p5_quantize(w, bits, group=None):
#     qmax = 2 ** (bits - 1) - 1
#     R, C = w.shape
#     if group is None: blocks = w.reshape(1, 1, R * C)          # one scale for the whole matrix
#     elif group == "row": blocks = w.reshape(R, 1, C)           # one scale per row
#     else: blocks = w.reshape(R, C // group, group)             # one scale per block of `group` weights
#     scales = np.abs(blocks).max(axis=2, keepdims=True) / qmax
#     scales = np.where(scales == 0, 1.0, scales)
#     q = np.clip(np.round(blocks / scales), -qmax, qmax).astype(np.int32)
#     return q.reshape(R, C), scales[:, :, 0]
# q, s = p5_quantize(np.array([ [0.9, -0.31, 0.04, 0.5] ]), 4)
# print("integers", q.ravel().tolist(), "| scale %.4f" % s[0, 0], "| back", np.round(p5_dequantize(q, s), 3).ravel().tolist())
# ```
#
# The only subtle line is the reshape: it turns the matrix into blocks of weights that share a scale (one block for the whole matrix, one per row, or several per row), so that a single `max` over the last axis finds the largest weight of every group at once. `np.round` rounds halves to the nearest even integer, which does not matter here. In a real implementation the integers are packed (two 4-bit values in one byte); we keep them in 32-bit integers because we only measure the error.
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise, and then the check on it.

# %%
def p5_quantize(w, bits, group=None):
    qmax = 2 ** (bits - 1) - 1
    R, C = w.shape
    if group is None: blocks = w.reshape(1, 1, R * C)          # one scale for the whole matrix
    elif group == "row": blocks = w.reshape(R, 1, C)           # one scale per row
    else: blocks = w.reshape(R, C // group, group)             # one scale per block of `group` weights
    scales = np.abs(blocks).max(axis=2, keepdims=True) / qmax
    scales = np.where(scales == 0, 1.0, scales)
    q = np.clip(np.round(blocks / scales), -qmax, qmax).astype(np.int32)
    return q.reshape(R, C), scales[:, :, 0]
q, s = p5_quantize(np.array([ [0.9, -0.31, 0.04, 0.5] ]), 4)
print("integers", q.ravel().tolist(), "| scale %.4f" % s[0, 0], "| back", np.round(p5_dequantize(q, s), 3).ravel().tolist())

# %%
p5_check_quantize()

# %% [markdown]
# ### Step 5.2. Error against bits, with and without outliers
#
# How much does each bit buy? We take a random weight matrix of 256 by 256 numbers from a Gaussian distribution (mean 0, standard deviation 1, like the weights of a freshly initialised layer, only larger) and quantize it with one scale for the whole matrix at 8, 6, 4, 3 and 2 bits. We measure the **mean squared error (MSE)**. A short calculation says that rounding to steps of size `scale` gives an error that is spread evenly between −scale/2 and +scale/2, whose mean square is `scale² / 12`; we print that prediction next to the measurement.
#
# Then the same matrix with **outliers**: we set 16 of the 65,536 weights (0.024 percent) to plus or minus 20 standard deviations. Real language models have a few such huge weights and activations, and they are the main reason why quantization is hard. Watch what one huge number does to the scale of the whole matrix.

# %%
# Step 5.2. Error against bits for a Gaussian matrix and for the same matrix with 16 outliers
p5_rng = np.random.default_rng(SEED)
P5_W = p5_rng.normal(0, 1, (256, 256))
P5_WO = P5_W.copy(); p5_idx = p5_rng.choice(P5_W.size, 16, replace=False)
P5_WO.flat[p5_idx] = p5_rng.choice([-1.0, 1.0], 16) * 20 * P5_W.std()
P5_BITS = [8, 6, 4, 3, 2]
p5_rows = []
for bits in P5_BITS:
    r = []
    for W in (P5_W, P5_WO):
        q, s = p5_quantize(W, bits); r += [p5_mse(W, p5_dequantize(q, s)), float(s[0, 0] ** 2 / 12)]
    p5_rows.append(r)
print("bits | Gaussian: MSE   predicted | with outliers: MSE   predicted | MSE ratio (outliers / Gaussian)")
for bits, r in zip(P5_BITS, p5_rows):
    print("%4d | %17.2e %10.2e | %22.2e %10.2e | %10.1f" % (bits, r[0], r[1], r[2], r[3], r[2] / r[0]))
RESULTS["p5_mse_4bit_gauss"] = float("%.3g" % p5_rows[2][0]); RESULTS["p5_mse_4bit_outlier"] = float("%.3g" % p5_rows[2][2])
plt.figure(figsize=(5.2, 3.6))
plt.semilogy(P5_BITS, [r[0] for r in p5_rows], "o-", label="Gaussian weights")
plt.semilogy(P5_BITS, [r[2] for r in p5_rows], "s-", label="same, with 16 outliers")
plt.gca().invert_xaxis(); plt.xlabel("bits per weight"); plt.ylabel("mean squared error"); plt.title("One scale for the whole matrix"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.show()

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; times can differ between machines):
#
# The table has five rows, from 8 bits down to 2 bits. For the Gaussian matrix the measured MSE is **0.000129 at 8 bits, 0.0426 at 4 bits and 0.23 at 3 bits**, and the prediction `scale² / 12` matches it to within about 1 percent for 8, 6, 4 and 3 bits. Every bit that you remove multiplies the error by about **4** (the step doubles, and the error is the square of the step): from 8 to 6 bits the MSE grows 17 times, which is 4 for each of the two bits. With the 16 outliers the error is about **15 times larger** at 8, 6 and 4 bits (2.1 × 10⁻³, 0.035 and 0.61), although the outliers are only 0.024 percent of the weights; at 3 and 2 bits the ratio shrinks because the error cannot exceed the size of the weights themselves (at 2 bits only the integers −1, 0 and +1 exist, and nearly every ordinary weight becomes 0). The figure shows two parallel lines, one about 15 times above the other.
#
# *What it means.* The formula `scale² / 12` is accurate while the steps are small; at 2 bits (and for the outlier matrix at 3 bits) it overestimates, because the predicted noise would be bigger than the weights. The outlier matrix is the important lesson: **the scale is set by the largest weight, so a handful of big values degrades every other weight in the group**. A 15-fold increase of the error is like losing about two bits.

# %% [markdown]
# ### Step 5.3. One scale for everything, one per row, one per group
#
# One huge weight makes the scale large, and a large scale makes the steps large for **every** weight in the group. The cure is to give each small group its own scale, so that an outlier spoils only its own neighbours. Here is the outlier matrix at 4 bits with the whole matrix as one group, one scale per row (256 weights), and one scale per block of 128, 64, 32 and 16 weights. Smaller groups are more exact but need **more scales**, and every scale costs 16 bits (a half-precision number). So the real storage per weight is **bits + 16 / g** for groups of `g` weights.

# %% [markdown]
# ### ✏️ Your turn: Exercise 5.2. Bits per weight, and the size of a model
#
# *Exercise 5.2:* Write `p5_bits_per_weight(bits, group)`, which returns `bits + 16 / group` (and just `bits` when `group` is `None`), and `p5_size_gb(params_billion, bits, group=None)`, which returns the size in gigabytes (powers of 1,000) of a model with that many billions of parameters: `params_billion × bits_per_weight / 8`. Why it matters: the scales are the hidden cost of 4-bit models, and this is the formula of the lesson. The check tests 4 bits with groups of 64 and 128, 8 bits with no scales counted, and the size of a 7-billion-parameter model.
#
# Write your answer in the starter cell below, then run the check cell under it.

# %%
def p5_bits_per_weight(bits, group):
    """Storage per weight in bits, counting one 16-bit scale per group."""
    return None   # YOUR CODE
def p5_size_gb(params_billion, bits, group=None):
    """Size in GB of a model with params_billion billion weights stored with `bits` bits and groups of `group`."""
    return None   # YOUR CODE

# %%
def p5_check_bits():
    if p5_bits_per_weight(4, 64) is None or p5_size_gb(7, 4, 64) is None:
        print("Fill in your answer to Exercise 5.2 above, then run this cell again."); return
    check("4 bits with groups of 64 cost 4.25 bits per weight", abs(p5_bits_per_weight(4, 64) - 4.25) < 1e-9, "bits + 16 / group")
    check("4 bits with groups of 128 cost 4.125 bits per weight", abs(p5_bits_per_weight(4, 128) - 4.125) < 1e-9, "bits + 16 / group")
    check("without groups the cost is just the bits", p5_bits_per_weight(8, None) == 8, "when group is None, return bits")
    check("7 billion weights at 4 bits, groups of 64: 3.72 GB", abs(p5_size_gb(7, 4, 64) - 3.7187) < 0.01, "params_billion * bits_per_weight / 8")
p5_check_bits()

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def p5_bits_per_weight(bits, group):
#     return bits if group is None else bits + 16 / group
# def p5_size_gb(params_billion, bits, group=None):
#     return params_billion * p5_bits_per_weight(bits, group) / 8
# print("%.4f bits, %.2f GB, %.2f GB" % (p5_bits_per_weight(4, 64), p5_size_gb(7, 4, 64), p5_size_gb(70, 4, 64)))
# ```
#
# A billion weights at 8 bits is one byte each, so 1 GB: that is why `params_billion * bits / 8` gives gigabytes directly. The 16 / group term shrinks as the groups grow: 0.25 bits for g = 64, 0.125 for g = 128, 0.0625 for a whole row of 256. A scale is stored once per group, whatever the number of bits.
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise, and then the check on it.

# %%
def p5_bits_per_weight(bits, group):
    return bits if group is None else bits + 16 / group
def p5_size_gb(params_billion, bits, group=None):
    return params_billion * p5_bits_per_weight(bits, group) / 8
print("%.4f bits, %.2f GB, %.2f GB" % (p5_bits_per_weight(4, 64), p5_size_gb(7, 4, 64), p5_size_gb(70, 4, 64)))

# %%
p5_check_bits()

# %%
# Step 5.3. The outlier matrix at 4 bits with different groups
P5_SCHEMES = [("whole matrix", None, 256 * 256), ("one scale per row", "row", 256), ("blocks of 128", 128, 128), ("blocks of 64", 64, 64), ("blocks of 32", 32, 32), ("blocks of 16", 16, 16)]
p5_sch = []
print("scheme                | scales | bits per weight | MSE at 4 bits | MSE of the Gaussian matrix, same scheme")
for name, grp, gsize in P5_SCHEMES:
    q, s = p5_quantize(P5_WO, 4, grp); mse_o = p5_mse(P5_WO, p5_dequantize(q, s))
    q2, s2 = p5_quantize(P5_W, 4, grp); mse_g = p5_mse(P5_W, p5_dequantize(q2, s2))
    bpw = p5_bits_per_weight(4, gsize); p5_sch.append((name, s.size, bpw, mse_o, mse_g))
    print("%-21s | %6d | %15.4f | %13.3f | %10.3f" % (name, s.size, bpw, mse_o, mse_g))
    assert abs(bpw - (4 + 16 * s.size / P5_WO.size)) < 1e-9          # the formula bits + 16 / g counts the scales exactly
RESULTS["p5_group64_gain"] = round(p5_sch[0][3] / p5_sch[3][3], 1)
print("Groups of 64 cut the error of the outlier matrix by a factor of %.1f compared with one scale, for %.2f extra bits per weight." % (RESULTS["p5_group64_gain"], p5_sch[3][2] - 4))

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; times can differ between machines):
#
# With one scale for the whole matrix the 4-bit error is **0.608** (the first row, 1 scale). One scale per row cuts it to **0.053**, blocks of 64 to **0.021** and blocks of 16 to **0.009**. The last line says that groups of 64 reduce the error of the outlier matrix **29.5 times** for **0.25 extra bits per weight** (4.25 instead of 4.0002). The right-hand column, the same schemes on the Gaussian matrix, shows that groups also help without outliers, but only a little (0.043 to 0.012): their real use is the protection against outliers. The assertion in the cell confirms that the formula `bits + 16 / g` counts the scales exactly. The Check lines of Exercise 5.2 print four ticks.
#
# *What it means.* Scales cost memory, and the cost grows quickly for small groups: blocks of 16 reach 5 bits per weight, so a "4-bit" model is really a 5-bit model. Groups of 64 or 128 are the usual compromise: about 4.1 to 4.25 bits per weight, and most of the benefit.

# %% [markdown]
# ### Step 5.4. Quantize a trained classifier
#
# Error in the weights is only interesting if it changes what the model **does**. We train a small classifier on a made-up task (6 classes, 24 numbers per example, overlapping groups, so that the accuracy is high but not perfect) with two hidden layers (24 → 128 → 64 → 6), for 100 steps of Adam: a second or two. Then we replace every weight matrix by its quantized-and-dequantized version (one scale per row, your `p5_quantize`) at 8, 6, 4, 3 and 2 bits, keep the biases as they are, and measure the **test accuracy** and the **relative output error** (the size of the change in the output numbers divided by the size of the output). This is called **simulated** or *fake* quantization: the weights are rounded, but the arithmetic is still in 32 bits, which is exactly what we need to measure the damage of the rounding.

# %%
# Step 5.4. Train a small classifier, then quantize its weights to fewer and fewer bits
P5_K, P5_D = 6, 24
p5_rng = np.random.default_rng(SEED); p5_means = p5_rng.normal(0, 1.0, (P5_K, P5_D))
def p5_data(n):
    y = p5_rng.integers(0, P5_K, n); X = p5_means[y] + p5_rng.normal(0, 1.6, (n, P5_D))
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y)
P5_Xtr, P5_ytr = p5_data(3000); P5_Xte, P5_yte = p5_data(2000)
torch.manual_seed(SEED)
P5_NET = nn.Sequential(nn.Linear(P5_D, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, P5_K))
p5_opt = torch.optim.Adam(P5_NET.parameters(), lr=1e-2, weight_decay=1e-3)
for _ in range(100):
    p5_opt.zero_grad(); F.cross_entropy(P5_NET(P5_Xtr), P5_ytr).backward(); p5_opt.step()
def p5_acc(model, X=None, y=None):
    """Test accuracy (percent) of a classifier."""
    X = P5_Xte if X is None else X; y = P5_yte if y is None else y
    with torch.no_grad(): return float((model(X).argmax(1) == y).float().mean() * 100)
def p5_fake_quant(model, bits, group="row", only=None):
    """A copy of the model whose Linear weights are replaced by quantized-then-dequantized weights. `only` = list of layer positions to quantize (default all)."""
    m = copy.deepcopy(model)
    for i, layer in enumerate(m):
        if isinstance(layer, nn.Linear) and (only is None or i in only):
            q, s = p5_quantize(layer.weight.detach().numpy().astype(np.float64), bits, group)
            layer.weight.data = torch.tensor(p5_dequantize(q, s), dtype=torch.float32)
    return m
P5_BASE_ACC = p5_acc(P5_NET)
with torch.no_grad(): P5_REF_OUT = P5_NET(P5_Xte)
print("full precision: train accuracy %.1f %%, test accuracy %.1f %%, parameters %d, float32 size %s" % (p5_acc(P5_NET, P5_Xtr, P5_ytr), P5_BASE_ACC, n_params(P5_NET), human_bytes(param_bytes(P5_NET))))
p5_curve = []
print("bits | test accuracy | share of full-precision accuracy | relative output error")
for bits in P5_BITS:
    m = p5_fake_quant(P5_NET, bits)
    with torch.no_grad(): err = p5_rel_err(P5_REF_OUT, m(P5_Xte))
    a = p5_acc(m); p5_curve.append((bits, a, err))
    print("%4d | %12.1f %% | %30.1f %% | %20.3f" % (bits, a, 100 * a / P5_BASE_ACC, err))
RESULTS["p5_acc_full"] = round(P5_BASE_ACC, 1)
for bits, a, err in p5_curve: RESULTS["p5_acc_%dbit" % bits] = round(a, 1)
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.4))
ax[0].plot([c[0] for c in p5_curve], [c[1] for c in p5_curve], "o-"); ax[0].axhline(P5_BASE_ACC, ls="--", c="gray"); ax[0].invert_xaxis(); ax[0].set_xlabel("bits per weight"); ax[0].set_ylabel("test accuracy (%)"); ax[0].set_title("Accuracy against bits")
ax[1].semilogy([c[0] for c in p5_curve], [c[2] for c in p5_curve], "o-", c="C1"); ax[1].invert_xaxis(); ax[1].set_xlabel("bits per weight"); ax[1].set_ylabel("relative output error"); ax[1].set_title("Output error against bits")
for a_ in ax: a_.grid(alpha=.3)
plt.tight_layout(); plt.show()

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; times can differ between machines):
#
# The classifier reaches **82.1 percent** on the test set (100 percent on the training set: with 3,000 examples and 11,846 weights it has memorised the training data, which does not matter for this experiment: we measure how much the **test** accuracy moves). With your quantizer the test accuracy is **82.1 percent at 8 bits, 81.8 at 6, 81.9 at 4, 79.8 at 3 and 63.2 at 2 bits**. The relative output error grows by a factor of 2 to 4 for every bit removed, from 0.004 at 8 bits to 0.068 at 4 bits and 0.629 at 2 bits. The figure shows an accuracy curve that is flat down to 4 bits, bends at 3 bits and **falls off a cliff at 2 bits**.
#
# *What it means.* The accuracy hardly moves while the output error is below a few percent: the decision of a classifier depends on which output is largest, not on its exact value, so it tolerates noise much better than the weights do. 8 and 6 bits are free here, 4 bits costs nothing measurable, and **the cliff is between 3 and 2 bits**. Do not read these numbers as universal: bigger models and harder tasks fall off earlier, and the accuracy of a trained language model must be measured on its own task.

# %% [markdown]
# ### Step 5.5. The real thing: PyTorch dynamic quantization
#
# Now we let PyTorch do it. **Dynamic quantization** stores the weights of every `nn.Linear` as 8-bit integers (with scales) and, when the model runs, quantizes the incoming activations on the fly and multiplies integers with integers. One call does it: `torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)`. We measure what is real: the size of the saved **state dict**, the **latency** of a forward pass at batch size 1, 8 and 64, the output error, and (for the classifier) the accuracy. We try the small classifier and a **wide** network, 1024 → 4096 → 1024 (8.4 million weights), which is large enough for the speed to matter.
#
# **A note on the warning.** In recent PyTorch versions `quantize_dynamic` prints a deprecation notice: the quantization tools are moving to a separate library, `torchao`. The function still works in the version that this lab uses, so we silence the notice; if a later version removes it, look in the documentation of `torchao` for the same idea.

# %%
# Step 5.5. quantize_dynamic on the classifier and on a wide network
try:
    torch.backends.quantized.engine = "fbgemm" if "fbgemm" in torch.backends.quantized.supported_engines else torch.backends.quantized.engine
    p5_dyn_ok = True
except Exception as e:
    p5_dyn_ok = False; print("dynamic quantization is not available here:", e)
def p5_dyn(model):
    """An 8-bit dynamic-quantized copy of the model."""
    return torch.ao.quantization.quantize_dynamic(copy.deepcopy(model), {nn.Linear}, dtype=torch.qint8)
torch.manual_seed(SEED)
P5_WIDE = nn.Sequential(nn.Linear(1024, 4096), nn.ReLU(), nn.Linear(4096, 1024)).eval()
P5_NET.eval()
if p5_dyn_ok:
    P5_NETQ = p5_dyn(P5_NET); P5_WIDEQ = p5_dyn(P5_WIDE)
    print("engine:", torch.backends.quantized.engine)
    print("%-26s | float32 | int8 dynamic | ratio" % "state dict (MB)")
    p5_sz = {}
    for name, a, b in (("classifier (24-128-64-6)", P5_NET, P5_NETQ), ("wide (1024-4096-1024)", P5_WIDE, P5_WIDEQ)):
        p5_sz[name] = (state_mb(a), state_mb(b)); print("%-26s | %7.3f | %12.3f | %4.2fx" % (name, p5_sz[name][0], p5_sz[name][1], p5_sz[name][0] / p5_sz[name][1]))
    with torch.no_grad():
        xw = torch.randn(64, 1024); ow, oq = P5_WIDE(xw), P5_WIDEQ(xw)
        print("output error of the wide network: %.2f %% of the output size" % (100 * p5_rel_err(ow, oq)))
        print("classifier: accuracy %.1f %% -> %.1f %%, output error %.2f %%" % (P5_BASE_ACC, p5_acc(P5_NETQ), 100 * p5_rel_err(P5_REF_OUT, P5_NETQ(P5_Xte))))
    print("batch | wide float32 (ms) | wide int8 (ms) | speed-up | classifier float32 (us) | int8 (us) | speed-up")
    p5_lat = {}
    for bsz in (1, 8, 64):
        x = torch.randn(bsz, 1024); xs = P5_Xte[:bsz]
        with torch.no_grad():
            a = time_ms(lambda: P5_WIDE(x), repeat=20, warmup=3); b = time_ms(lambda: P5_WIDEQ(x), repeat=20, warmup=3)
            c = time_ms(lambda: P5_NET(xs), repeat=200, warmup=10) * 1000; d = time_ms(lambda: P5_NETQ(xs), repeat=200, warmup=10) * 1000
        p5_lat[bsz] = (a, b); print("%5d | %17.2f | %14.2f | %7.2fx | %23.1f | %9.1f | %6.2fx" % (bsz, a, b, a / b, c, d, c / d))
    RESULTS["p5_wide_size_ratio"] = round(p5_sz["wide (1024-4096-1024)"][0] / p5_sz["wide (1024-4096-1024)"][1], 2)
    RESULTS["p5_wide_speedup_b1"] = round(p5_lat[1][0] / p5_lat[1][1], 2); RESULTS["p5_wide_speedup_b64"] = round(p5_lat[64][0] / p5_lat[64][1], 2)
    assert 3.5 < RESULTS["p5_wide_size_ratio"] < 4.1       # four times fewer bytes per weight
    RESULTS["p5_dyn_err_pct"] = round(100 * p5_rel_err(ow, oq), 2)

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; times can differ between machines):
#
# On the machine that ran this lab: the wide network's state dict shrinks from **33.6 MB to 8.4 MB (3.99 times smaller)**, the size of the 8-bit weights plus a few scales; the small classifier shrinks only 2.9 times, because its biases and the file overhead are a larger share. The output of the wide network changes by **2.4 percent** of its size, and the classifier keeps its accuracy (82.1 percent before and after; output error 1.8 percent). The latency of the wide network drops by a factor of about **3 to 5 at batch 1 and 8, and 2 to 3 at batch 64** (for example 1.1 ms to 0.25 ms at batch 1). The surprise is the small classifier: its int8 version is about **3 times slower** (40 microseconds against 125 microseconds at batch 1).
#
# *What it means.* Quantization is not magic: for a tiny network the cost of quantizing the activations on the fly and calling the integer kernels is larger than the saving, while for a wide network that reads 100 MB of weights per forward pass, reading a quarter of the bytes helps a lot. The speed-up falls at batch 64 because with many inputs the arithmetic, not the memory, starts to limit the speed: that is the same story as in Step 6.1. These timings are for this CPU (with its 8-bit integer kernels); on another CPU or on a GPU they can be very different, so measure on the hardware that you will deploy to.
#
# *A word about timings.* The time figures in this paragraph come from a run on a quiet machine (nothing else running); the table printed above comes from your own run and can differ by tens of percent, and much more if other programs are using the processor. The **ratios, the order of the rows and the shape of the curves** are what to compare, not the milliseconds.

# %% [markdown]
# ### Step 5.6. Which layer hurts most?
#
# Not every layer is equally fragile. We quantize the three weight matrices of the classifier **one at a time** to 3 bits (one scale per row) and leave the other two in full precision, and compare with quantizing all three. The layer that loses most accuracy is the one that you would keep at higher precision: real quantization tools do exactly this, for instance by leaving the first and the last layer in 16 bits.

# %%
# Step 5.6. Per-layer sensitivity at 3 bits
P5_LAYERS = [0, 2, 4]       # positions of the three Linear layers in the Sequential
P5_LNAME = {0: "layer 1 (24 -> 128)", 2: "layer 2 (128 -> 64)", 4: "layer 3 (64 -> 6)"}
p5_sens = {}
print("quantized layer      | weights | test accuracy | change | relative output error")
for pos in P5_LAYERS + [None]:
    m = p5_fake_quant(P5_NET, 3, "row", only=None if pos is None else [pos])
    with torch.no_grad(): err = p5_rel_err(P5_REF_OUT, m(P5_Xte))
    a = p5_acc(m); name = "all three layers" if pos is None else P5_LNAME[pos]
    nw = sum(P5_NET[i].weight.numel() for i in P5_LAYERS) if pos is None else P5_NET[pos].weight.numel()
    p5_sens[name] = (a, err); print("%-20s | %7d | %12.1f %% | %+5.1f | %.3f" % (name, nw, a, a - P5_BASE_ACC, err))
p5_worst = min((k for k in p5_sens if k != "all three layers"), key=lambda k: p5_sens[k][0])
RESULTS["p5_worst_layer"] = p5_worst
print("Most sensitive single layer at 3 bits:", p5_worst)

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; times can differ between machines):
#
# Quantizing one layer at a time to 3 bits changes the test accuracy by **−0.4 points for layer 1, −0.4 for layer 2 and −1.1 for layer 3**; all three together lose **2.3 points** (79.8 percent), a little more than the sum of the three (1.9): the errors add up. The **most sensitive single layer is the last one** (64 → 6, with only 384 weights), although it is by far the smallest: its output error is 0.113, as large as the first layer's (0.109), and more than twice that of the middle layer (0.046), which holds most of the weights.
#
# *What it means.* Sensitivity has nothing to do with size. The last layer produces the scores that are compared directly, so its errors are not averaged out by later layers; the biggest layer is the most robust here. Real tools exploit this: the small, sensitive layers (the first and the last, the embeddings, the normalisation layers) are kept in 16 bits, and the big, tolerant middle layers are quantized hard, which saves most of the memory for little loss.

# %% [markdown]
# ### Step 5.7. Is decoding memory-bound on this machine?
#
# The lesson says that generating text one token at a time is limited by **memory bandwidth**, not by arithmetic: to produce one token the computer must read **every weight once**, and does only about two operations per weight it reads. Then the speed limit is `tokens per second ≤ bandwidth ÷ model bytes`. Quantizing to 8 bits makes the model four times smaller, so the bound is four times higher. Let us test the claim on one big matrix, 4096 by 4096 (16.8 million weights, 67 MB in float32), the shape of a layer in a 7-billion-parameter model: a matrix-vector product (batch 1) in float32 and in 8-bit dynamic quantization, with the time per byte and the bandwidth that the timing implies. First we measure this machine's bandwidth by reading a big array (128 MB), which is larger than any cache.

# %% [markdown]
# ### ✏️ Your turn: Exercise 5.3. The speed limit of decoding
#
# *Exercise 5.3:* Write `p5_decode_bound(bandwidth_gb_s, model_gb)`: the largest possible number of tokens per second, `bandwidth_gb_s / model_gb`. Why it matters: it is the quickest way to know whether a model can be fast enough on given hardware, before you test anything. The check uses the lesson's example numbers (an *assumed* bandwidth of 200 GB/s): a model of 14 GB gives 14.3 tokens per second at most, and the same model in 4 bits (3.5 GB) gives 57.1.
#
# Write your answer in the starter cell below, then run the check cell under it.

# %%
def p5_decode_bound(bandwidth_gb_s, model_gb):
    """Upper bound on tokens per second when every token needs one read of the whole model."""
    return None   # YOUR CODE

# %%
def p5_check_bound():
    if p5_decode_bound(200, 14) is None:
        print("Fill in your answer to Exercise 5.3 above, then run this cell again."); return
    check("14 GB at 200 GB/s: 14.3 tokens per second", abs(p5_decode_bound(200, 14) - 14.2857) < 0.01, "bandwidth divided by model size")
    check("3.5 GB at 200 GB/s: 57.1 tokens per second", abs(p5_decode_bound(200, 3.5) - 57.1429) < 0.01, "a smaller model reads faster")
    check("twice the bandwidth gives twice the bound", abs(p5_decode_bound(400, 14) / p5_decode_bound(200, 14) - 2) < 1e-9, "the bound is proportional to bandwidth")
p5_check_bound()

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def p5_decode_bound(bandwidth_gb_s, model_gb):
#     return bandwidth_gb_s / model_gb
# print("%.1f and %.1f tokens per second" % (p5_decode_bound(200, 14), p5_decode_bound(200, 3.5)))
# ```
#
# Units do the work: gigabytes per second divided by gigabytes per token gives tokens per second. The bound ignores the arithmetic, the cache and the overhead, so real systems reach a fraction of it; but it tells you which resource limits you, and why halving the bytes (a smaller number of bits) nearly doubles the speed of decoding.
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise, and then the check on it.

# %%
def p5_decode_bound(bandwidth_gb_s, model_gb):
    return bandwidth_gb_s / model_gb
print("%.1f and %.1f tokens per second" % (p5_decode_bound(200, 14), p5_decode_bound(200, 3.5)))

# %%
p5_check_bound()

# %%
# Step 5.7. Bandwidth of this machine, then a matrix-vector product in float32 and in int8
p5_big = torch.ones(32_000_000)                                  # 128 MB of float32: larger than any cache
p5_t_read = time_ms(lambda: p5_big.sum(), repeat=10, warmup=2) / 1000
P5_BW = p5_big.numel() * 4 / p5_t_read / 1e9                     # GB/s read by a sum over the array
del p5_big
torch.manual_seed(SEED)
p5_lin = nn.Sequential(nn.Linear(4096, 4096, bias=False)).eval()
p5_linq = p5_dyn(p5_lin) if p5_dyn_ok else None
p5_bytes32 = param_bytes(p5_lin); p5_bytes8 = p5_lin[0].weight.numel()          # int8 = one byte per weight (the scales are tiny)
print("Bandwidth measured by reading a 128 MB array: %.1f GB/s" % P5_BW)
print("float32 weights: %.1f MB | int8 weights: %.1f MB" % (p5_bytes32 / 1e6, p5_bytes8 / 1e6))
print("batch | float32 (ms) | float32 GB/s read | int8 (ms) | int8 GB/s read | speed-up | bound from bandwidth: float32, int8 (matrix products per second)")
p5_mv = {}
for bsz in (1, 64):
    x = torch.randn(bsz, 4096)
    with torch.no_grad():
        a = time_ms(lambda: p5_lin(x), repeat=30, warmup=5); b = time_ms(lambda: p5_linq(x), repeat=30, warmup=5) if p5_linq is not None else float("nan")
    p5_mv[bsz] = (a, b)
    print("%5d | %12.2f | %17.1f | %9.2f | %14.1f | %7.2fx | %.0f, %.0f" % (bsz, a, p5_bytes32 / (a / 1000) / 1e9, b, p5_bytes8 / (b / 1000) / 1e9, a / b, p5_decode_bound(P5_BW, p5_bytes32 / 1e9), p5_decode_bound(P5_BW, p5_bytes8 / 1e9)))
RESULTS["p5_bandwidth_gb_s"] = round(P5_BW, 1)
RESULTS["p5_mv_speedup_b1"] = round(p5_mv[1][0] / p5_mv[1][1], 2); RESULTS["p5_mv_speedup_b64"] = round(p5_mv[64][0] / p5_mv[64][1], 2)
print("Time per megabyte read at batch 1: float32 %.3f ms, int8 %.3f ms; at batch 64: float32 %.3f ms, int8 %.3f ms." % (p5_mv[1][0] / (p5_bytes32 / 1e6), p5_mv[1][1] / (p5_bytes8 / 1e6), p5_mv[64][0] / (p5_bytes32 / 1e6), p5_mv[64][1] / (p5_bytes8 / 1e6)))

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; times can differ between machines):
#
# The bandwidth probe reads about **30 GB/s** on this machine (31 to 33 in our runs). At batch 1 the float32 matrix–vector product takes about **2.3 ms**, which means that it reads the 67 MB of weights at about **29 GB/s**: **the same as the bandwidth of the machine, so this product is memory-bound** (the bound from bandwidth, about 470 products per second, is close to the measured 430). The int8 version reads a quarter of the bytes and is about **5 times faster** (0.47 ms): the speed-up is a little above the 4 that the byte count predicts, because the int8 kernel is also more efficient and a part of its 16.8 MB sits in the processor's cache, which is faster than main memory. At batch 64 the picture changes: the time per product is only 5 to 6 times larger although the work is 64 times larger, and the bytes read per second drop to about 5 GB/s: the weights are read once and **used 64 times**, so the arithmetic, not the memory, sets the time, and the speed-up of int8 over float32 is smaller (about 3 times).
#
# *What it means.* The claim of the lesson holds on this laptop: decoding one token at a time is limited by how fast the weights can be read, so fewer bytes means more tokens per second, and quantization helps. It also holds that **batching changes the regime**: with many sequences in the batch, every byte read is used many times and the machine becomes compute-bound. A GPU has far more memory bandwidth (and far more arithmetic) than this CPU, but the same two regimes.
#
# *A word about timings.* The time figures in this paragraph come from a run on a quiet machine (nothing else running); the table printed above comes from your own run and can differ by tens of percent, and much more if other programs are using the processor. The **ratios, the order of the rows and the shape of the curves** are what to compare, not the milliseconds.

# %% [markdown]
# ### Step 5.8. Weight-only 4-bit: the arithmetic for 7 B and 70 B
#
# Large language models are usually quantized **weight-only**: the weights are stored in 4 or 8 bits (with group scales), and are converted back to 16-bit numbers inside the matrix product, while the activations stay in 16 bits. The size arithmetic does not depend on the machine, so we can do it exactly with your function from Exercise 5.2, with the lesson's numbers as assertions, and add the decoding speed limit of Exercise 5.3 at an **assumed** bandwidth of 200 GB/s (an example, not a real device: replace it by the figure for your own hardware).

# %%
# Step 5.8. Sizes of 7 B and 70 B models at 16, 8 and 4 bits, and the decoding bound
P5_ASSUMED_BW = 200.0       # GB/s, an assumption for the table: use the figure of your hardware
print("model | format                       | bits per weight | size (GB) | decoding bound at an assumed %d GB/s (tokens/s)" % P5_ASSUMED_BW)
for pb in (7, 70):
    for name, bits, g in (("16-bit", 16, None), ("8-bit", 8, None), ("4-bit, groups of 128", 4, 128), ("4-bit, groups of 64", 4, 64)):
        gb = p5_size_gb(pb, bits, g)
        print("%3d B | %-28s | %15.3f | %9.2f | %10.1f" % (pb, name, p5_bits_per_weight(bits, g), gb, p5_decode_bound(P5_ASSUMED_BW, gb)))
assert abs(p5_size_gb(7, 16) - 14) < 1e-9 and abs(p5_size_gb(7, 8) - 7) < 1e-9 and abs(p5_size_gb(7, 4) - 3.5) < 1e-9      # the lesson: 14, 7, 3.5 GB
assert abs(p5_size_gb(70, 16) - 140) < 1e-9 and abs(p5_size_gb(70, 8) - 70) < 1e-9 and abs(p5_size_gb(70, 4) - 35) < 1e-9
assert abs(p5_size_gb(7, 4, 64) - 3.71875) < 1e-9 and abs(p5_size_gb(70, 4, 64) - 37.1875) < 1e-9
RESULTS["p5_7b_int4_g64_gb"] = round(p5_size_gb(7, 4, 64), 2); RESULTS["p5_70b_int4_g64_gb"] = round(p5_size_gb(70, 4, 64), 2)
print("An 80 GB GPU holds a 70 B model at 4 bits (%.1f GB of weights) with room for the cache; at 16 bits (140 GB) it needs two GPUs for the weights alone." % p5_size_gb(70, 4, 64))

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; times can differ between machines):
#
# The sizes are exact. A 7-billion-parameter model needs **14 GB at 16 bits, 7 GB at 8 bits, 3.5 GB at 4 bits** and **3.72 GB at 4 bits with groups of 64** (4.25 bits per weight); a 70-billion-parameter model needs **140, 70, 35 and 37.19 GB**. The assertions confirm the lesson's numbers. At an assumed 200 GB/s the decoding bound rises from 14.3 tokens per second for the 7 B model in 16 bits to 53.8 in 4 bits with groups of 64 (3.8 times: a little less than 4 because of the scales); for the 70 B model it rises from 1.4 to 5.4.
#
# *What it means.* A 70 B model in 16 bits needs two 80 GB GPUs just for the weights, and in 4 bits it fits on one with room for the cache. The decoding bound is the most optimistic number: it ignores the cache, the arithmetic, the overhead and the time to talk between GPUs.

# %% [markdown]
# ### Step 5.9. Compare with your guesses, and the limits of what we did
#
# Run the cell below to put your guesses next to the measurements (if you left a guess as `None` it shows "(not written)").

# %%
# Step 5.9. Imagine first against the measurements
def p5_g(x, unit=""):
    return "(not written)" if x is None else "%s%s" % (x, unit)
full = RESULTS["p5_acc_full"]
print("what you guessed -> what was measured")
print("8-bit accuracy:   %s -> %.1f %% of full-precision accuracy" % (p5_g(guess_acc_8bit, " %"), 100 * RESULTS["p5_acc_8bit"] / full))
print("4-bit accuracy:   %s -> %.1f %%" % (p5_g(guess_acc_4bit, " %"), 100 * RESULTS["p5_acc_4bit"] / full))
print("2-bit accuracy:   %s -> %.1f %%" % (p5_g(guess_acc_2bit, " %"), 100 * RESULTS["p5_acc_2bit"] / full))
print("state dict size:  %s times smaller -> %.2f times smaller" % (p5_g(guess_size_ratio), RESULTS.get("p5_wide_size_ratio", float("nan"))))
print("speed at batch 1: %s times faster -> %.2f times faster" % (p5_g(guess_speedup_b1), RESULTS.get("p5_wide_speedup_b1", float("nan"))))

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; times can differ between machines):
#
# The cell compares your guesses with the numbers. For the classifier, **8-bit weights keep 100 percent of the accuracy, 4-bit weights 99.8 percent and 2-bit weights only 77 percent**. The wide network's state dict becomes **3.99 times smaller** and its forward pass at batch 1 about **3 to 5 times faster** on this machine. If you guessed that 4 bits would hurt much more and that quantization would only save memory, now you know that on a wide network it can also save time, when the model is limited by memory reads, and that the cliff is lower than most people expect.
#
# **The limits of this part, written down.** (1) We quantized small models; large language models have **outliers** in their weights and especially in their **activations** (a few channels with values a hundred times larger than the rest), and a plain 8-bit scheme for the activations can fail on them: the method LLM.int8() (Dettmers et al., 2022) was designed around this problem, and 4-bit methods such as GPTQ (Frantar et al., 2022) choose the rounding of each weight to minimise the error of the layer's *output* instead of the error of the weight. (2) **Weight-only** quantization saves memory and memory bandwidth, but the arithmetic is still in 16 bits; **weight-and-activation** quantization (INT8 multiplied by INT8) also speeds up the arithmetic where the hardware supports it, and needs **calibration data** (a few hundred typical inputs) to choose the ranges of the activations. (3) **Post-training** quantization, which we did, rounds a trained model; **quantization-aware training** trains with the rounding inside the loop and usually loses less at 4 bits and below. (4) We measured accuracy on a toy task: **always re-measure the quality of your own model on your own task** after quantizing it, ideally including the rare inputs that matter most.


# %% [markdown]
# ## Part 6. Serving: latency, throughput and the KV cache (30 min)
#
# *Lesson section: "Serving a model".* A trained model is only useful when people can use it, and a server has to balance two things that pull in opposite directions: **latency** (how long one person waits) and **throughput** (how many requests per second the machine finishes, and therefore what each costs). ==In this part you measure why batching raises throughput, simulate a server with random arrivals to see the price in waiting time, write the KV cache of a Transformer decoder and check that it gives exactly the same text faster, and do the memory and cost arithmetic of serving a 7-billion-parameter model.== The measurements are on this machine; the memory sizes are counted exactly from tensors and formulas. Prices and the sizes of the "example GPU" are **assumptions** that are flagged where they appear.

# %% [markdown]
# ### ✏️ Imagine first
#
# A restaurant kitchen can cook one dish in 10 minutes, but the oven holds 8 dishes, and a batch of 8 also takes about 12 minutes. A model server is like that: running **one** request through the network costs nearly as much as running **several** together, because the weights are read from memory once either way. Before you run anything, guess:
#
# 1. A batch of 64 inputs against a batch of 1: how many times **longer** does the forward pass take (a number: 1 would mean no extra time, 64 would mean no gain from batching)?
# 2. So how many times more **requests per second** does the batch of 64 finish than the batch of 1?
# 3. Generating 160 tokens one by one **without** a KV cache (re-computing the whole sequence at every step), how many times **slower** is the last token than with a cache?
#
# Write your guesses in the starter cell below (replace each `None`), then run it. Do not look ahead: you compare with the real numbers at the end of this part.

# %%
# Imagine first: what does batching buy, and what does a cache buy?
guess_batch64_time_ratio = None     # YOUR GUESS: time of a forward pass at batch 64 divided by the time at batch 1 (a number)
guess_throughput_gain = None        # YOUR GUESS: requests per second at batch 64 divided by requests per second at batch 1 (a number)
guess_nocache_slowdown = None       # YOUR GUESS: time of the 160th token without a cache divided by its time with a cache (a number)

# %%
# Helpers for Part 6 (all names start with p6_ or P6_).
if "check" not in globals():
    def check(name, ok, hint=""):
        """Print a tick or a cross with a hint, so you know whether your answer is right."""
        print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))
if "RESULTS" not in globals():
    RESULTS = {}      # a place to keep your key numbers; the last part of the lab prints them together

def p6_weights_gb(params_billion, bits, group=None):
    """Size in GB (powers of 1000) of the weights: bits per weight plus one 16-bit scale per group, if there are groups."""
    return params_billion * (bits if group is None else bits + 16 / group) / 8
def p6_median_ms(fn, repeat=25, warmup=3):
    """Median time of fn() in milliseconds (the median ignores the odd slow run). On a GPU it waits for the GPU."""
    for _ in range(warmup): fn()
    ts = []
    for _ in range(repeat):
        if HAS_CUDA: torch.cuda.synchronize()
        t = time.perf_counter(); fn()
        if HAS_CUDA: torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))
print("helpers ready")

# %% [markdown]
# ### Step 6.1. Measure batching
#
# We time the forward pass of a wide network (1024 → 4096 → 4096 → 1024, 25 million weights, a stand-in for the feed-forward part of a Transformer) for batch sizes 1, 2, 4, 8, 16, 32 and 64. With a small batch the computer spends most of its time **reading the weights** (about 100 MB of them), and that costs the same for one input or for several; with a large batch the arithmetic starts to dominate and the time grows with the batch. A straight line describes the simplest version of the story: **time = t₀ + c × b**, where `t₀` is the fixed cost of a batch (reading the weights, starting the work) and `c` the extra cost of every additional request. One warning about this machine: for a **single** input PyTorch uses a different routine (a matrix-vector product), which on many CPUs is faster than the general matrix product used for two or more inputs. So we time all seven batch sizes but **fit the line on batch sizes 2 to 64**, and show batch 1 next to the line.

# %% [markdown]
# ### ✏️ Your turn: Exercise 6.1. Fit the line
#
# *Exercise 6.1:* Write `p6_fit(batch_sizes, times_ms)`, which finds the least-squares line `time = t0 + c * b` and returns the pair `(t0, c)`. Use `np.polyfit(batch_sizes, times_ms, 1)`, which returns the slope first and the intercept second. Why it matters: with `t0` and `c` you can predict the time of any batch, the capacity of a server and the batch size that you need. The check tests the function on exact data (`t = 10 + 2 b`) and on noisy data.
#
# Write your answer in the starter cell below, then run the check cell under it.

# %%
def p6_fit(batch_sizes, times_ms):
    """Least-squares line time = t0 + c * b. Returns (t0, c)."""
    return None   # YOUR CODE: slope, intercept = np.polyfit(...); return the intercept first

# %%
def p6_check_fit():
    if p6_fit([1, 2, 3], [12, 14, 16]) is None:
        print("Fill in your answer to Exercise 6.1 above, then run this cell again."); return
    t0, c = p6_fit([1, 2, 4, 8, 16], [12, 14, 18, 26, 42])
    check("exact data t = 10 + 2 b gives t0 = 10 and c = 2", abs(t0 - 10) < 1e-6 and abs(c - 2) < 1e-6, "polyfit returns (slope, intercept): return the intercept first")
    r = np.random.default_rng(3); b = np.arange(1, 65); t0n, cn = p6_fit(b, 5 + 0.3 * b + r.normal(0, 0.2, 64))
    check("noisy data: t0 is close to 5 and c is close to 0.3", abs(t0n - 5) < 0.2 and abs(cn - 0.3) < 0.02, "a degree-1 least-squares fit")
p6_check_fit()

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def p6_fit(batch_sizes, times_ms):
#     c, t0 = np.polyfit(batch_sizes, times_ms, 1)
#     return float(t0), float(c)
# print("t0 = %.1f, c = %.2f" % p6_fit([1, 2, 4, 8, 16], [12, 14, 18, 26, 42]))
# ```
#
# `np.polyfit(x, y, 1)` returns the coefficients from the highest power to the lowest, so the slope `c` comes first and the intercept `t0` second. The slope is the extra time for one more request in the batch; the intercept is what a batch costs even when it is empty, which is the price that batching shares among the requests.
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise, and then the check on it.

# %%
def p6_fit(batch_sizes, times_ms):
    c, t0 = np.polyfit(batch_sizes, times_ms, 1)
    return float(t0), float(c)
print("t0 = %.1f, c = %.2f" % p6_fit([1, 2, 4, 8, 16], [12, 14, 18, 26, 42]))

# %%
p6_check_fit()

# %%
# Step 6.1. Time a forward pass at batch sizes 1 ... 64, fit t0 + c b, and plot
torch.manual_seed(SEED)
P6_MLP = nn.Sequential(nn.Linear(1024, 4096), nn.ReLU(), nn.Linear(4096, 4096), nn.ReLU(), nn.Linear(4096, 1024)).eval()
P6_BS = [1, 2, 4, 8, 16, 32, 64]
p6_ts = []
with torch.no_grad():
    for bsz in P6_BS:
        x = torch.randn(bsz, 1024); p6_ts.append(p6_median_ms(lambda: P6_MLP(x), repeat=(5 if QUICK else 25), warmup=3))
P6_T0, P6_C = p6_fit(P6_BS[1:], p6_ts[1:])          # fitted on batch sizes 2 to 64: batch 1 uses a special, faster routine (see the text)
p6_pred = np.array([P6_T0 + P6_C * b for b in P6_BS]); p6_r2 = 1 - np.sum((np.array(p6_ts[1:]) - p6_pred[1:]) ** 2) / np.sum((np.array(p6_ts[1:]) - np.mean(p6_ts[1:])) ** 2)
print("model: %d weights (%s as float32)" % (n_params(P6_MLP), human_bytes(param_bytes(P6_MLP))))
print("batch | time (ms) | requests per second | time per request (ms) | line t0 + c b (ms)")
for bsz, t, pr in zip(P6_BS, p6_ts, p6_pred):
    print("%5d | %9.2f | %19.0f | %21.3f | %7.2f" % (bsz, t, 1000 * bsz / t, t / bsz, pr))
print("fit on batch sizes 2 to 64: t0 = %.2f ms, c = %.3f ms per request, R squared = %.3f" % (P6_T0, P6_C, p6_r2))
print("batch 1 against the line: measured %.2f ms, the line says %.2f ms" % (p6_ts[0], p6_pred[0]))
print("batch 64 takes %.1f times as long as batch 1 and finishes %.1f times as many requests per second." % (p6_ts[-1] / p6_ts[0], (64 / p6_ts[-1]) / (1 / p6_ts[0])))
RESULTS["p6_t0_ms"] = round(P6_T0, 2); RESULTS["p6_c_ms"] = round(P6_C, 3)
RESULTS["p6_b64_time_ratio"] = round(p6_ts[-1] / p6_ts[0], 1); RESULTS["p6_throughput_gain_b64"] = round((64 / p6_ts[-1]) / (1 / p6_ts[0]), 1)
fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.5))
ax[0].plot(P6_BS, p6_ts, "o", label="measured"); ax[0].plot(P6_BS, p6_pred, "-", label="t0 + c b"); ax[0].set_xlabel("batch size b"); ax[0].set_ylabel("time of a batch (ms)"); ax[0].set_title("Latency of a batch"); ax[0].legend()
ax[1].plot(P6_BS, [1000 * b / t for b, t in zip(P6_BS, p6_ts)], "o-", c="C2"); ax[1].set_xlabel("batch size b"); ax[1].set_ylabel("requests per second"); ax[1].set_title("Throughput")
for a_ in ax: a_.grid(alpha=.3)
plt.tight_layout(); plt.show()

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; times can differ between machines):
#
# Batching is cheap. On the machine that ran this lab the 100 MB network takes about **3.6 ms at batch 1** (the special single-row routine) and then **9.5 ms at batch 2, 4 and 8** (flat: the time is the cost of reading the weights), 12.6 at batch 16, 15 at batch 32 and **17 ms at batch 64**. The line fitted on batch sizes 2 to 64 has **t₀ of about 9.5 ms** and **c of about 0.13 ms per request**, with R² around 0.9 (a straight line is only an approximation: the real curve is flat first and then rises). Batch 64 takes about **5 times as long as batch 1** and finishes about **12 times as many requests per second** (about 3,500 against 280). The time per request falls from 3.6 ms to 0.27 ms. The two plots show a latency that rises slowly and a throughput that rises almost in proportion to the batch.
#
# *What it means.* A fixed cost shared among many requests is why servers batch. The price is that every request in the batch has to wait for the slowest part of the batch and, in a real server, for the batch to fill (Step 6.2). On a GPU the shape is the same with much larger `t₀`-to-`c` ratios.
#
# *A word about timings.* The time figures in this paragraph come from a run on a quiet machine (nothing else running); the table printed above comes from your own run and can differ by tens of percent, and much more if other programs are using the processor. The **ratios, the order of the rows and the shape of the curves** are what to compare, not the milliseconds.

# %% [markdown]
# ### Step 6.2. Simulate a server
#
# Requests do not arrive in a neat row. We model them as a **Poisson process**: the gaps between arrivals are random (exponential), with an average rate of λ requests per second. The server collects requests into a **batch**: it takes the first waiting request, then waits until it has `b` requests **or** a **timeout** has passed since that first request arrived (or until the server is free again, if it is busy), and serves the batch in `t₀ + c × (size of the batch)` milliseconds, using the `t₀` and `c` that you just measured. We record the latency of every request (from its arrival to the end of its batch) and report the median (**p50**) and the tail (**p95**, **p99**: the time that 95 or 99 percent of the requests beat). The lesson's formula for the average latency without a timeout is **wait + service = (b − 1) / (2λ) + t₀ + c b**: a request waits on average for half of the other `b − 1` arrivals to complete its batch.

# %% [markdown]
# ### ✏️ Your turn: Exercise 6.2. The latency formula
#
# *Exercise 6.2:* Write `p6_formula_latency(lam, b, t0, c)`, which returns the average latency in milliseconds predicted by the lesson's formula: `(b - 1) / (2 * lam) * 1000 + t0 + c * b`, where `lam` is the arrival rate in requests per **second** and `t0`, `c` are in milliseconds (the factor 1000 turns seconds into milliseconds). Why it matters: it shows in one line why a bigger batch is cheaper per request but slower for the person who waits. The check uses the lesson's worked example: `t0 = 40` ms, `c = 5` ms, `b = 8`, `lam = 50` per second gives a wait of 70 ms plus a service of 80 ms, so 150 ms.
#
# Write your answer in the starter cell below, then run the check cell under it.

# %%
def p6_formula_latency(lam, b, t0, c):
    """Predicted average latency in ms: waiting for the batch to fill plus the service time."""
    return None   # YOUR CODE

# %%
def p6_check_formula():
    if p6_formula_latency(50, 8, 40, 5) is None:
        print("Fill in your answer to Exercise 6.2 above, then run this cell again."); return
    check("the lesson's example (lam 50, b 8, t0 40, c 5) gives 150 ms", abs(p6_formula_latency(50, 8, 40, 5) - 150) < 1e-9, "wait (b - 1) / (2 lam) * 1000 plus service t0 + c b")
    check("with b = 1 there is no waiting: the latency is t0 + c", abs(p6_formula_latency(10, 1, 40, 5) - 45) < 1e-9, "(b - 1) is zero")
    check("a higher arrival rate shortens the wait", p6_formula_latency(100, 8, 40, 5) < p6_formula_latency(50, 8, 40, 5), "lam is in the denominator")
p6_check_formula()

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def p6_formula_latency(lam, b, t0, c):
#     return (b - 1) / (2 * lam) * 1000 + t0 + c * b
# print("%.0f ms" % p6_formula_latency(50, 8, 40, 5))
# ```
#
# The first term is the average wait: if requests arrive every 1 / lam seconds, the first request of a batch waits for `b - 1` arrivals and the last waits for none, so on average half of `b - 1` gaps. The second term is the service time of the batch. Batching trades the first term (which grows with `b`) for a cheaper second term per request (`t0 / b` shrinks).
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise, and then the check on it.

# %%
def p6_formula_latency(lam, b, t0, c):
    return (b - 1) / (2 * lam) * 1000 + t0 + c * b
print("%.0f ms" % p6_formula_latency(50, 8, 40, 5))

# %%
p6_check_formula()

# %%
# Step 6.2. The simulator, and a table over batch sizes and arrival rates
def p6_simulate(lam, b, timeout_ms, t0, c, n=20000, seed=SEED):
    """Poisson arrivals (lam per second), a batcher (up to b requests, or the timeout after the first one), service t0 + c * size.
    Returns the latencies in ms (one per request), the sizes of the batches and the throughput (requests per second)."""
    rng = np.random.default_rng(seed)
    arr = np.cumsum(rng.exponential(1000.0 / lam, n))          # arrival times in ms
    lat = np.empty(n); sizes = []; i = 0; free = 0.0
    while i < n:
        last = min(i + b - 1, n - 1)
        close = max(free, min(arr[last], arr[i] + timeout_ms))  # the batch closes when it is full or on timeout, but not before the server is free
        j = max(i + 1, min(int(np.searchsorted(arr, close, side="right")), i + b, n))
        free = close + t0 + c * (j - i)
        lat[i:j] = free - arr[i:j]; sizes.append(j - i); i = j
    return lat, sizes, n / free * 1000
P6_N = 4000 if QUICK else 20000
p6_cap1 = 1000 / (P6_T0 + P6_C)
P6_LAMS = [round(0.6 * p6_cap1, -1), round(1.5 * p6_cap1, -1), round(4 * p6_cap1, -1)]
P6_BSIZES = [1, 2, 4, 8, 16, 32]
p6_sim = {}
print("fit: t0 = %.2f ms, c = %.3f ms; capacity at b = 1: %.0f requests per second; no timeout (the batch waits until it is full)" % (P6_T0, P6_C, p6_cap1))
for lam in P6_LAMS:
    print("\narrival rate %d per second" % lam)
    print("  b | capacity (req/s) | stable | p50 (ms) | p95 (ms) | p99 (ms) | throughput | mean (ms) | formula mean (ms)")
    for b in P6_BSIZES:
        lat, sizes, thr = p6_simulate(lam, b, 1e12, P6_T0, P6_C, n=P6_N)
        cap = 1000 * b / (P6_T0 + P6_C * b); p6_sim[(lam, b)] = (np.percentile(lat, 50), np.percentile(lat, 95), np.percentile(lat, 99), thr, lat.mean(), cap)
        print("%3d | %16.0f | %6s | %8.1f | %8.1f | %8.1f | %10.0f | %9.1f | %9.1f" % (b, cap, "yes" if cap > lam else "NO", *p6_sim[(lam, b)][:3], thr, lat.mean(), p6_formula_latency(lam, b, P6_T0, P6_C)))

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; times can differ between machines):
#
# Look at the three blocks, one per arrival rate. The capacity column is `1000 b / (t₀ + c b)`; "stable" means that the capacity exceeds the arrival rate. **At the lowest rate (60 per second)** every batch size is stable, but latency **grows with b**: the mean is about 16 ms for b = 1 and about 270 ms for b = 32, because a request waits for the batch to fill (the p95 is 35 ms against 550 ms). **At 160 per second, above the capacity of b = 1 (104 per second)**, the batch size 1 is not stable: its median latency is about 35 *seconds*; b = 2 to 32 are fine, and b = 4 has the lowest p95 (39 ms). **At 410 per second** b = 1, 2 and 4 are all unstable (capacities 104, 204 and 398), and b = 8 is the smallest stable batch (p95 about 32 ms). In the stable cases the simulated **mean latency agrees with the lesson's formula** to within about 1 to 2 percent (for example 32.3 against 32.4 ms for b = 8 at 160 per second); the formula is **wrong at b = 1 and b = 2 at the lowest rate** (16.3 against 9.7 ms) and wherever the load is close to the capacity, because it counts the wait for the batch to fill and the service time but **not the queue**, which appears when requests arrive while the server is busy. The throughput column equals the arrival rate when the server is stable, and equals the capacity when it is not.
#
# *What it means.* Batching is a trade: it buys capacity and costs waiting time, and the best batch size depends on the traffic. The formula is a good guide when the server has spare capacity and the batch is large enough, and it is silent about the queue. The p95 and p99 are 2 to 3 times the median: a service-level objective is written on the tail, not on the mean.
#
# *A word about timings.* The time figures in this paragraph come from a run on a quiet machine (nothing else running); the table printed above comes from your own run and can differ by tens of percent, and much more if other programs are using the processor. The **ratios, the order of the rows and the shape of the curves** are what to compare, not the milliseconds.

# %% [markdown]
# ### Step 6.3. Too much load, and a latency target
#
# Two questions are left. What happens when the arrival rate is **above** the capacity of the server? The queue grows without limit: every request waits a little longer than the one before. We watch the latency of requests 0 to 20,000 for `b = 1` at the middle arrival rate, where the capacity is too small. And how do you choose the batch size? A typical rule: the **largest batch whose p95 latency stays under a target** (the service-level objective, SLO), because a bigger batch means fewer machines for the same traffic.

# %% [markdown]
# ### ✏️ Your turn: Exercise 6.3. Choose the batch size
#
# *Exercise 6.3:* Write `p6_pick(rows, target_ms)`. `rows` is a list of `(b, p95_ms)` pairs. Return the **largest** `b` whose `p95_ms` is at most `target_ms`, or `None` if no batch size meets the target. Why it matters: this is the decision that a serving team makes for every service. The check tests three small tables, one of which has no answer.
#
# Write your answer in the starter cell below, then run the check cell under it.

# %%
def p6_pick(rows, target_ms):
    """The largest batch size whose p95 latency is at most the target, or None."""
    return None   # YOUR CODE

# %%
def p6_check_pick():
    if p6_pick([(1, 10)], 100) is None:
        print("Fill in your answer to Exercise 6.3 above, then run this cell again."); return
    check("the largest b under the target is chosen", p6_pick([(1, 500), (2, 60), (4, 45), (8, 90), (16, 300)], 100) == 8, "keep the rows with p95 <= target, then take the maximum b")
    check("the target is inclusive", p6_pick([(1, 50), (2, 100), (4, 101)], 100) == 2, "at most the target means p95 <= target")
    check("no answer gives None", p6_pick([(1, 500), (2, 400)], 100) is None, "return None when no row qualifies")
p6_check_pick()

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def p6_pick(rows, target_ms):
#     ok = [b for b, p95 in rows if p95 <= target_ms]
#     return max(ok) if ok else None
# print(p6_pick([(1, 500), (2, 60), (4, 45), (8, 90), (16, 300)], 100))
# ```
#
# Notice that the p95 is **not** monotonic in `b`: at low load a small batch is unstable (the queue grows), a medium batch is best, and a large one makes people wait for the batch to fill. That is why the rule picks the largest batch that still meets the target and not simply the smallest p95.
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise, and then the check on it.

# %%
def p6_pick(rows, target_ms):
    ok = [b for b, p95 in rows if p95 <= target_ms]
    return max(ok) if ok else None
print(p6_pick([(1, 500), (2, 60), (4, 45), (8, 90), (16, 300)], 100))

# %%
p6_check_pick()

# %%
# Step 6.3. The queue that never empties, and the choice of b under a target
lam_mid = P6_LAMS[1]
lat1, _, thr1 = p6_simulate(lam_mid, 1, 1e12, P6_T0, P6_C, n=P6_N)
print("arrival rate %d per second against a capacity of %.0f at b = 1:" % (lam_mid, p6_cap1))
for k in (0.0, 0.25, 0.5, 0.75, 1.0):
    i = min(int(k * (P6_N - 1)), P6_N - 1); print("  request %5d: latency %9.1f ms" % (i, lat1[i]))
print("throughput %.0f requests per second: it cannot exceed the capacity, so the rest waits." % thr1)
P6_TARGET = round(6 * (P6_T0 + P6_C), 0)
print("\nLatency target: p95 at most %.0f ms (six times the time of a single request)" % P6_TARGET)
for lam in P6_LAMS:
    rows = [(b, p6_sim[(lam, b)][1]) for b in P6_BSIZES]
    best_p95 = min(rows, key=lambda r: r[1])
    print("  arrival rate %4d: b with the lowest p95 = %d (%.1f ms), largest b under the target = %s" % (lam, best_p95[0], best_p95[1], p6_pick(rows, P6_TARGET)))
RESULTS["p6_queue_latency_end_ms"] = round(float(lat1[-1]), 0)
plt.figure(figsize=(5.4, 3.5)); plt.plot(lat1[::20]); plt.xlabel("request number (every 20th)"); plt.ylabel("latency (ms)"); plt.title("Arrival rate above capacity (b = 1)"); plt.grid(alpha=.3); plt.tight_layout(); plt.show()

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; times can differ between machines):
#
# Above the capacity the queue never empties: for b = 1 at 160 requests per second against a capacity of 104, the latency of request 0 is 9.7 ms, of request 5,000 about **17.5 seconds**, and of request 20,000 about **68 seconds**; the throughput stays at the capacity (104 per second) and the figure shows a straight, rising line. With the target "p95 at most six times the time of one request" (58 ms) the rule of Exercise 6.3 picks **b = 2 at 60 requests per second, b = 4 at 160 and b = 16 at 410**. The Check lines print three ticks.
#
# *What it means.* The right batch size grows with the load: at low traffic a small batch is best (nobody waits for others), at high traffic you need a larger batch to keep up, and the SLO caps it from above. In practice servers add a **timeout** so that a lonely request is not held back for a batch that never fills, and measure the p95 under real traffic.
#
# *A word about timings.* The time figures in this paragraph come from a run on a quiet machine (nothing else running); the table printed above comes from your own run and can differ by tens of percent, and much more if other programs are using the processor. The **ratios, the order of the rows and the shape of the curves** are what to compare, not the milliseconds.

# %% [markdown]
# ### Step 6.4. The KV cache, written by hand
#
# Generating text means running the decoder again for every new token. Without a cache, each step **recomputes the keys and values of all the earlier tokens**, which are exactly the same as in the previous step: the cost of step `n` grows with `n`. With a **KV cache** you keep the keys and values of the past in memory and compute only the new token's row: the projections cost the same at every step, and only the attention itself (the new query against `n` stored keys) still reads `n` entries. The price is **memory**: for every layer, every position and every sequence in the batch you store one key vector and one value vector of size `d`.
#
# Below is a tiny decoder written from scratch (2 layers, 8 heads, `d` = 256, a vocabulary of 200, random weights: it produces gibberish but it is a real, causal, multi-head Transformer decoder), in two versions: `p6_generate_nocache` recomputes everything at each step, and `p6_generate_cache` keeps the keys and values. We generate 160 tokens with both, assert that **the tokens are identical and the scores agree**, and time every step.

# %% [markdown]
# ### ✏️ Your turn: Exercise 6.4. How many bytes does the cache need?
#
# *Exercise 6.4:* Write `p6_kv_bytes(layers, n, d, bytes_per_value, batch)`: the bytes of the KV cache, `2 * layers * n * d * bytes_per_value * batch` (the 2 is for keys and values). Why it matters: the cache grows with the length of the context and with the number of simultaneous users, and for long contexts it can be larger than the model. The check uses the lesson's example: 32 layers, `d` = 4,096, 4,096 tokens, a batch of 8 and 2 bytes per value (16-bit) need 17.2 GB.
#
# Write your answer in the starter cell below, then run the check cell under it.

# %%
def p6_kv_bytes(layers, n, d, bytes_per_value, batch):
    """Bytes of the KV cache: keys and values, for every layer, position and sequence."""
    return None   # YOUR CODE

# %%
def p6_check_kv():
    if p6_kv_bytes(1, 1, 1, 1, 1) is None:
        print("Fill in your answer to Exercise 6.4 above, then run this cell again."); return
    check("the lesson's example needs 17.2 GB", abs(p6_kv_bytes(32, 4096, 4096, 2, 8) / 1e9 - 17.18) < 0.01, "2 * layers * n * d * bytes * batch")
    check("the cache of 1 sequence of 1 token in 1 layer, d = 10, 4 bytes: 80 bytes", p6_kv_bytes(1, 1, 10, 4, 1) == 80, "keys and values: the factor 2")
    check("doubling the context doubles the cache", p6_kv_bytes(32, 8192, 4096, 2, 1) == 2 * p6_kv_bytes(32, 4096, 4096, 2, 1), "the cache is proportional to n")
p6_check_kv()

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def p6_kv_bytes(layers, n, d, bytes_per_value, batch):
#     return 2 * layers * n * d * bytes_per_value * batch
# print("%.2f GB" % (p6_kv_bytes(32, 4096, 4096, 2, 8) / 1e9))
# ```
#
# Per token, per layer, you keep a key and a value of `d` numbers each; multiplied by the layers, the tokens and the sequences, with the bytes of each number, that is the whole cache. At 4,096 tokens and a batch of 8, the cache of this model (17.2 GB) is already larger than the 14 GB of its 16-bit weights. Models with grouped-query attention store fewer key and value heads, which divides the formula by a constant; the formula here is the plain multi-head case.
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise, and then the check on it.

# %%
def p6_kv_bytes(layers, n, d, bytes_per_value, batch):
    return 2 * layers * n * d * bytes_per_value * batch
print("%.2f GB" % (p6_kv_bytes(32, 4096, 4096, 2, 8) / 1e9))

# %%
p6_check_kv()

# %%
# Step 6.4. A tiny decoder with and without a KV cache: same tokens, different cost
P6_V, P6_D, P6_H, P6_L = 200, 256, 8, 2
_g = torch.Generator().manual_seed(SEED); _s = P6_D ** -0.5
P6_EMB = torch.randn(P6_V, P6_D, generator=_g); P6_OUT = torch.randn(P6_V, P6_D, generator=_g)      # separate input and output tables
P6_W = [dict(q=torch.randn(P6_D, P6_D, generator=_g) * _s, k=torch.randn(P6_D, P6_D, generator=_g) * _s, v=torch.randn(P6_D, P6_D, generator=_g) * _s,
             o=torch.randn(P6_D, P6_D, generator=_g) * _s, f1=torch.randn(P6_D, 4 * P6_D, generator=_g) * _s, f2=torch.randn(4 * P6_D, P6_D, generator=_g) * (4 * P6_D) ** -0.5) for _ in range(P6_L)]
_pos = torch.arange(1024)[:, None] / (10000 ** (torch.arange(0, P6_D, 2)[None, :] / P6_D))
P6_POS = torch.zeros(1024, P6_D); P6_POS[:, 0::2] = torch.sin(_pos); P6_POS[:, 1::2] = torch.cos(_pos)
def p6_norm(x): return x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6)
def p6_heads(x): return x.reshape(x.shape[0], x.shape[1], P6_H, P6_D // P6_H).transpose(1, 2)        # (batch, heads, tokens, d / heads)
def p6_attend(q, K, V, causal):
    s = q @ K.transpose(-1, -2) / math.sqrt(P6_D // P6_H)
    if causal: s = s.masked_fill(torch.triu(torch.ones(s.shape[-2], s.shape[-1], dtype=torch.bool), 1), float("-inf"))
    return (torch.softmax(s, -1) @ V).transpose(1, 2).reshape(q.shape[0], q.shape[2], P6_D)
def p6_logits_full(tokens):
    """No cache: run all the tokens through the decoder and return the scores for the last position."""
    n_tok = tokens.shape[1]; x = P6_EMB[tokens] + P6_POS[:n_tok]
    for w in P6_W:
        h = p6_norm(x); x = x + p6_attend(p6_heads(h @ w["q"]), p6_heads(h @ w["k"]), p6_heads(h @ w["v"]), True) @ w["o"]
        x = x + torch.relu(p6_norm(x) @ w["f1"]) @ w["f2"]
    return p6_norm(x)[:, -1] @ P6_OUT.T
def p6_logits_step(tok, t, cache):
    """With a cache: process one new token at position t, append its key and value to the cache, return its scores."""
    x = (P6_EMB[tok] + P6_POS[t])[:, None, :]
    for l, w in enumerate(P6_W):
        h = p6_norm(x); k, v = p6_heads(h @ w["k"]), p6_heads(h @ w["v"])
        cache[l] = (k, v) if cache[l] is None else (torch.cat([cache[l][0], k], 2), torch.cat([cache[l][1], v], 2))
        x = x + p6_attend(p6_heads(h @ w["q"]), cache[l][0], cache[l][1], False) @ w["o"]
        x = x + torch.relu(p6_norm(x) @ w["f1"]) @ w["f2"]
    return p6_norm(x)[:, 0] @ P6_OUT.T
def p6_generate_nocache(prompt, n_new):
    toks = prompt.clone(); times = []; lg = []
    for _ in range(n_new):
        t = time.perf_counter(); s = p6_logits_full(toks); times.append((time.perf_counter() - t) * 1000)
        lg.append(s); toks = torch.cat([toks, s.argmax(-1, keepdim=True)], 1)
    return toks, times, lg
def p6_generate_cache(prompt, n_new):
    cache = [None] * P6_L; toks = prompt.clone(); times = []; lg = []
    for t in range(prompt.shape[1] - 1): p6_logits_step(prompt[:, t], t, cache)          # fill the cache with the prompt
    nxt = prompt[:, -1]
    for _ in range(n_new):
        t0_ = time.perf_counter(); s = p6_logits_step(nxt, toks.shape[1] - 1, cache); times.append((time.perf_counter() - t0_) * 1000)
        lg.append(s); nxt = s.argmax(-1); toks = torch.cat((toks, nxt.unsqueeze(1)), 1)
    return toks, times, lg, cache
P6_NNEW = 40 if QUICK else 160
P6_PROMPT = torch.tensor([ [5, 17, 42, 8] ])
with torch.no_grad():
    p6_ta, p6_tn, p6_ln = p6_generate_nocache(P6_PROMPT, P6_NNEW)
    p6_tb, p6_tc, p6_lc, p6_cache = p6_generate_cache(P6_PROMPT, P6_NNEW)
p6_maxdiff = max(float((a - b).abs().max()) for a, b in zip(p6_ln, p6_lc))
assert torch.equal(p6_ta, p6_tb), "the cached decoder must produce exactly the same tokens"
assert p6_maxdiff < 1e-3
print("generated %d tokens from a prompt of %d: identical tokens: %s | largest difference between the scores of the two versions: %.1e" % (P6_NNEW, P6_PROMPT.shape[1], bool(torch.equal(p6_ta, p6_tb)), p6_maxdiff))
print("first generated tokens:", p6_ta[0, 4:14].tolist())
print("position | no cache (ms/token) | with cache (ms/token) | no cache / cache   (average over 10 steps)")
p6_rows4 = []
for end in range(10, P6_NNEW + 1, 30 if not QUICK else 10):
    a = float(np.mean(p6_tn[end - 10:end])); b = float(np.mean(p6_tc[end - 10:end])); p6_rows4.append((P6_PROMPT.shape[1] + end, a, b))
    print("%8d | %19.3f | %21.3f | %10.1f" % (P6_PROMPT.shape[1] + end, a, b, a / b))
RESULTS["p6_nocache_slowdown"] = round(p6_rows4[-1][1] / p6_rows4[-1][2], 1)
RESULTS["p6_nocache_growth"] = round(float(np.mean(p6_tn[-10:]) / np.mean(p6_tn[:10])), 1); RESULTS["p6_cache_growth"] = round(float(np.mean(p6_tc[-10:]) / np.mean(p6_tc[:10])), 1)
print("From the first to the last 10 tokens, the time per token grows %.1f times without a cache and %.1f times with one." % (RESULTS["p6_nocache_growth"], RESULTS["p6_cache_growth"]))
plt.figure(figsize=(5.6, 3.6)); plt.plot(range(P6_PROMPT.shape[1], P6_PROMPT.shape[1] + P6_NNEW), p6_tn, label="no cache"); plt.plot(range(P6_PROMPT.shape[1], P6_PROMPT.shape[1] + P6_NNEW), p6_tc, label="with KV cache")
plt.xlabel("position of the new token"); plt.ylabel("time for one token (ms)"); plt.title("Time per generated token"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.show()

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; times can differ between machines):
#
# The two versions generate the **same 160 tokens** and their scores agree to about **3 × 10⁻⁵** (the difference of the float32 rounding of two ways to compute the same numbers). Without a cache, the time to generate a token **grows with the position**: from about 0.8 ms near the start to about **3.7 ms near the end** (4 to 5 times more); with the cache it stays between **0.55 and 0.65 ms** (it grows only 1.1 times, because the attention still reads all the stored keys). At position 164 the cache makes a token about **5 to 6 times faster**. The figure shows a rising curve and a nearly flat one.
#
# *What it means.* Without a cache the total work to generate `n` tokens grows with the **square** of `n`; with the cache it grows (almost) linearly. The speed-up is modest here because this toy is small and each step is dominated by the fixed cost of calling many small operations: in a real model, at thousands of tokens, the difference between recomputing and remembering is the difference between unusable and usable. We did not change a single weight: the cache is only a trade of memory for time.
#
# *A word about timings.* The time figures in this paragraph come from a run on a quiet machine (nothing else running); the table printed above comes from your own run and can differ by tens of percent, and much more if other programs are using the processor. The **ratios, the order of the rows and the shape of the curves** are what to compare, not the milliseconds.

# %% [markdown]
# ### Step 6.5. Count the cache from the real tensors
#
# The formula of Exercise 6.4 should match the bytes of the tensors that the cache really holds. We run the cached decoder for a **batch of 4 sequences** and 40 new tokens, add up the `nbytes` of every key and value tensor, and compare with your function (4 bytes per value, because this toy runs in float32). Then we look at what the same formula says for a 7-billion-parameter-like model.

# %%
# Step 6.5. Cache bytes: measured from the tensors against the formula
P6_PROMPT4 = torch.tensor([ [5, 17, 42, 8], [1, 2, 3, 4], [9, 99, 19, 91], [7, 7, 7, 7] ])
with torch.no_grad():
    toks4, _, _, cache4 = p6_generate_cache(P6_PROMPT4, 40)
p6_measured = sum(k.nbytes + v.nbytes for k, v in cache4)
n_cached = cache4[0][0].shape[2]
p6_formula = p6_kv_bytes(P6_L, n_cached, P6_D, 4, 4)
print("the cache holds %d positions for %d sequences and %d layers: measured %d bytes (%s), formula %d bytes" % (n_cached, 4, P6_L, p6_measured, human_bytes(p6_measured), p6_formula))
assert p6_measured == p6_formula == 2 * P6_L * n_cached * P6_D * 4 * 4
print("in 16 bits the same cache would need %s; in 8 bits %s" % (human_bytes(p6_kv_bytes(P6_L, n_cached, P6_D, 2, 4)), human_bytes(p6_kv_bytes(P6_L, n_cached, P6_D, 1, 4))))
print("model weights of this toy: %s (float32); the cache is %.0f %% of them" % (human_bytes(4 * (2 * P6_V * P6_D + P6_L * 12 * P6_D * P6_D)), 100 * p6_measured / (4 * (2 * P6_V * P6_D + P6_L * 12 * P6_D * P6_D))))
RESULTS["p6_cache_bytes_toy"] = int(p6_measured)

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; times can differ between machines):
#
# The cache holds 43 positions (3 prompt tokens stored in advance and one more for each of the 40 steps) for 4 sequences and 2 layers: **704,512 bytes measured, 704,512 bytes by the formula**, and the assertion passes. In 16 bits the same cache would need 352 KB and in 8 bits 176 KB. Even for this toy the cache is already **11 percent of the model's weights** after only 43 tokens, and it grows with every token.
#
# *What it means.* The formula `2 × L × n × d × bytes × batch` is not an approximation but an exact count of the tensors: keys and values (the 2), for each layer, each position, each of the `d` numbers, each sequence in the batch. A 16-bit cache halves it, an 8-bit cache quarters it (at some cost in quality that you must test), and fewer key and value heads (grouped-query attention) divide it again.

# %% [markdown]
# ### Step 6.6. The memory budget of a 7 B-like server
#
# Does a model **fit** on a GPU? The memory must hold the weights **and** the KV cache of every sequence that is being served (plus activations and overhead, which we ignore here, so our maximum batch is an optimistic upper bound). We use a 7-billion-parameter-like configuration (32 layers, `d` = 4,096) and an **assumed example GPU with 80 GB of memory** (not a real device: put the memory of your own hardware in `P6_MEM_GB`). The first table shows the cache in GB for several batch sizes and context lengths. The second shows the **largest batch** that fits at a context of 4,096 tokens with 16-bit or 4-bit weights (groups of 64) and with a 16-bit or 8-bit cache.

# %%
# Step 6.6. Weights plus cache against the memory of an (assumed) 80 GB GPU
P6_MEM_GB = 80.0                    # an assumption for the example: use the memory of your hardware
P6_LAYERS, P6_DMODEL, P6_PARAMS_B = 32, 4096, 7
def p6_budget(params_b, bits, group, layers, d, n, batch, kv_bytes, mem_gb):
    """Return (weights GB, cache GB, total GB, fits?) for serving `batch` sequences of `n` tokens."""
    w = p6_weights_gb(params_b, bits, group); kv = p6_kv_bytes(layers, n, d, kv_bytes, batch) / 1e9
    return w, kv, w + kv, w + kv <= mem_gb
def p6_max_batch(params_b, bits, group, layers, d, n, kv_bytes, mem_gb):
    """The largest number of sequences of n tokens that fit next to the weights (0 if even the weights do not fit)."""
    w = p6_weights_gb(params_b, bits, group); per_seq = p6_kv_bytes(layers, n, d, kv_bytes, 1) / 1e9
    return max(0, int((mem_gb - w) // per_seq))
print("KV cache in GB (16-bit values), %d layers, d = %d" % (P6_LAYERS, P6_DMODEL))
print("context | " + " | ".join("batch %2d" % b for b in (1, 8, 32)))
for n in (2048, 4096, 8192):
    print("%7d | " % n + " | ".join("%8.2f" % p6_budget(P6_PARAMS_B, 16, None, P6_LAYERS, P6_DMODEL, n, b, 2, P6_MEM_GB)[1] for b in (1, 8, 32)))
assert abs(p6_budget(7, 16, None, 32, 4096, 4096, 8, 2, 80)[1] - 17.18) < 0.01              # the lesson's example
print("\nLargest batch at 4096 tokens on %.0f GB (optimistic: activations and overhead ignored)" % P6_MEM_GB)
print("weights              | cache values | weights GB | cache per sequence GB | largest batch")
p6_mb = {}
for wname, bits, grp in (("16-bit", 16, None), ("4-bit, groups of 64", 4, 64)):
    for kvb, kname in ((2, "16-bit"), (1, "8-bit")):
        m = p6_max_batch(P6_PARAMS_B, bits, grp, P6_LAYERS, P6_DMODEL, 4096, kvb, P6_MEM_GB); p6_mb[(wname, kname)] = m
        print("%-20s | %12s | %10.2f | %21.3f | %13d" % (wname, kname, p6_weights_gb(P6_PARAMS_B, bits, grp), p6_kv_bytes(P6_LAYERS, 4096, P6_DMODEL, kvb, 1) / 1e9, m))
        assert p6_budget(P6_PARAMS_B, bits, grp, P6_LAYERS, P6_DMODEL, 4096, m, kvb, P6_MEM_GB)[3] and not p6_budget(P6_PARAMS_B, bits, grp, P6_LAYERS, P6_DMODEL, 4096, m + 1, kvb, P6_MEM_GB)[3]
assert p6_mb[("16-bit", "16-bit")] == 30 and p6_mb[("4-bit, groups of 64", "16-bit")] == 35
RESULTS["p6_maxbatch_fp16"] = p6_mb[("16-bit", "16-bit")]; RESULTS["p6_maxbatch_int4"] = p6_mb[("4-bit, groups of 64", "16-bit")]; RESULTS["p6_maxbatch_int4_kv8"] = p6_mb[("4-bit, groups of 64", "8-bit")]

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; times can differ between machines):
#
# For the 7 B-like configuration the cache of **one** sequence of 4,096 tokens is **2.15 GB** in 16 bits; the first table shows 1.07 GB at 2,048 tokens, 17.2 GB for a batch of 8 at 4,096 (the lesson's number, asserted) and **137 GB for a batch of 32 at 8,192 tokens**, far beyond one 80 GB device. On the assumed 80 GB GPU, the largest batch at 4,096 tokens is **30 sequences with 16-bit weights and 16-bit cache**, **35 with 4-bit weights** (the weights shrink from 14 to 3.72 GB, which frees room for five more sequences), **61 with 16-bit weights and an 8-bit cache** and **71 with both**. The assertions check each result and that one more sequence does not fit.
#
# *What it means.* At long contexts the cache, not the weights, is what limits the number of users per GPU: quantizing the weights gained 5 sequences, quantizing the cache gained 31. Halving the context also doubles the batch. Remember that these are upper bounds: activations, the framework and fragmentation take a share of the memory.

# %% [markdown]
# ### Step 6.7. What does a million tokens cost?
#
# Cost per million tokens is the price of the machine for the time it takes: `price per hour ÷ (tokens per second × 3,600) × 1,000,000`. The tokens per second of decoding come from the memory-bound rule of Part 5: each step reads the weights **once** plus the cache of **every** sequence in the batch, so a step takes `(weights + b × cache per sequence) ÷ bandwidth` and produces `b` tokens. The inputs below are **assumptions**, not real prices or devices: a bandwidth of 200 GB/s, a price of 2.00 per hour (any currency), a context of 4,096 tokens. They are only here to show the **shape** of the answer: batching and quantization both lower the cost, and the cache sets the limit.

# %%
# Step 6.7. Cost per million generated tokens (ALL numbers are assumptions)
P6_BW_GB_S, P6_PRICE_HOUR = 200.0, 2.00
def p6_cost_per_m(tokens_per_s, price_per_hour):
    """Cost of one million tokens when the machine makes `tokens_per_s` tokens per second and costs `price_per_hour`."""
    return price_per_hour / (tokens_per_s * 3600) * 1e6
def p6_decode_tps(weights_gb, kv_seq_gb, batch, bw_gb_s):
    """Memory-bound tokens per second: a step reads the weights once and the cache of each of the `batch` sequences."""
    return batch / ((weights_gb + batch * kv_seq_gb) / bw_gb_s)
assert abs(p6_cost_per_m(1000, 3.6) - 1.0) < 1e-9                    # 1000 tokens/s at 3.60 an hour: 1.00 per million tokens
print("assumed: bandwidth %.0f GB/s, price %.2f per hour, context 4096 tokens, 16-bit cache" % (P6_BW_GB_S, P6_PRICE_HOUR))
print("weights              | batch | tokens/s (bound) | step (ms) | cost per million tokens")
p6_cost = {}
for wname, bits, grp in (("16-bit", 16, None), ("4-bit, groups of 64", 4, 64)):
    w = p6_weights_gb(P6_PARAMS_B, bits, grp); kvs = p6_kv_bytes(P6_LAYERS, 4096, P6_DMODEL, 2, 1) / 1e9
    for b in (1, 8, 30):
        tps = p6_decode_tps(w, kvs, b, P6_BW_GB_S); p6_cost[(wname, b)] = p6_cost_per_m(tps, P6_PRICE_HOUR)
        print("%-20s | %5d | %16.1f | %9.1f | %10.2f" % (wname, b, tps, 1000 * b / tps, p6_cost[(wname, b)]))
RESULTS["p6_cost_fp16_b1"] = round(p6_cost[("16-bit", 1)], 1); RESULTS["p6_cost_fp16_b30"] = round(p6_cost[("16-bit", 30)], 2); RESULTS["p6_cost_int4_b30"] = round(p6_cost[("4-bit, groups of 64", 30)], 2)
print("Batch 30 against batch 1 at 16 bits: %.1f times cheaper per token (the same machine, shared by more requests)." % (p6_cost[("16-bit", 1)] / p6_cost[("16-bit", 30)]))

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; times can differ between machines):
#
# With the assumed numbers (200 GB/s, 2.00 per hour, a context of 4,096 tokens), a 16-bit 7 B-like model at batch 1 can decode at most **12.4 tokens per second**, which costs **44.85 per million tokens**; at batch 8 the bound is 51 tokens per second (**10.83** per million), and at the largest batch that fits, 30, it is 76.5 tokens per second (**7.26**): batching makes a token **6.2 times cheaper**. With 4-bit weights the cost at batch 1 falls to 16.30 and at batch 30 to **6.31**. The step time grows with the batch (81 ms at batch 1, 392 ms at batch 30): every sequence in the batch must wait for the cache of all the others to be read.
#
# *What it means.* The shape matters more than the numbers, which are assumptions: cost per token falls steeply with the batch at first and then flattens, because at a large batch the **cache**, not the weights, is what each step reads; quantizing the weights helps most at small batches (where the weights dominate). A real service chooses the batch that meets its latency target (Step 6.3) and then counts the cost.
#
# **Continuous batching, named but not built.** The server of Step 6.2 collects a batch, serves it, and only then starts the next one: a long answer holds up the short ones in its batch. Modern serving systems use **continuous batching** (also called in-flight batching): after every decoding step, finished sequences leave the batch and waiting requests join it, so the GPU never idles waiting for the slowest sequence. Together with a cache that is stored in small blocks instead of one long strip per sequence (the idea of PagedAttention in the vLLM system, Kwon et al., 2023), it is what lets real servers keep the batch large and the memory full. We do not implement it here; the capacity planner in Part 7 uses the same memory and bandwidth formulas.

# %% [markdown]
# ### Step 6.8. Compare with your guesses
#
# Run the cell below to put your guesses next to the measurements (if you left a guess as `None` it shows "(not written)").

# %%
# Step 6.8. Imagine first against the measurements
def p6_g(x):
    return "(not written)" if x is None else "%s" % x
print("what you guessed -> what was measured")
print("time at batch 64 divided by time at batch 1:  %s -> %.1f" % (p6_g(guess_batch64_time_ratio), RESULTS["p6_b64_time_ratio"]))
print("throughput gain of batch 64:                  %s -> %.1f" % (p6_g(guess_throughput_gain), RESULTS["p6_throughput_gain_b64"]))
print("slowdown of the last token without a cache:   %s -> %.1f" % (p6_g(guess_nocache_slowdown), RESULTS["p6_nocache_slowdown"]))

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; times can differ between machines):
#
# The three comparisons are the heart of the part. **Batch 64 took about 5 times as long as batch 1 but finished about 12 times as many requests per second**: batching is nearly free until the arithmetic starts to count. **Without a cache the last of 160 tokens was about 5 to 6 times slower** than with one, and the gap widens as the context grows. If your guesses were far from these numbers, look at which assumption was wrong: the idea that "twice the work takes twice the time" is true only when the machine is busy with arithmetic; for a model that waits for memory it is not.
#
# *A word about timings.* The time figures in this paragraph come from a run on a quiet machine (nothing else running); the table printed above comes from your own run and can differ by tens of percent, and much more if other programs are using the processor. The **ratios, the order of the rows and the shape of the curves** are what to compare, not the milliseconds.
#
# ### ✏️ Reflect: Part 6
#
# 1. Your server has a fixed `t0` and `c`. At which arrival rates does a batch of 1 stop being stable, and what does the batch size have to be to stay stable at twice that rate? Use the capacity formula `1000 b / (t0 + c b)`.
# 2. Why does the p95 latency of the larger batches **rise** at low arrival rates, although they are more efficient? What would a **timeout** change?
# 3. The cache for a 4,096-token context was larger than the weights of the model at a batch of 8. Name two ways to make it smaller, and say what each costs.

# %% [markdown]
# ## Part 7. Plan the deployment and reflect (20 min)
#
# *Lesson sections: "Serving a model" and "Key takeaways".* In Parts 1 to 6 you measured one thing at a time: memory, a gradient average, a split matrix, a pipeline bubble, a number format, a quantizer and a batching server. A real deployment needs all of them at once, and it needs one person to say: *this model, on this hardware, for this many users, at this cost, and this is what we do when it breaks.* ==In this part you write a small capacity planner, run it for three deployments, turn the result into a checklist and a design card, and collect the honest findings of the whole week.== Every price, bandwidth and memory size in this part is **ASSUMED**: it is a parameter that you replace with the numbers of your own hardware report from the Setup, not a specification of any real product.
#
# **Imagine first.** You run the school helpdesk model. Before you run anything, guess three things. (1) Can a 70-billion-parameter model in 16-bit numbers fit on **one** 80 GB GPU? (2) For that 70 B model, which is cheaper per million tokens: four 80 GB GPUs with 16-bit weights, or one 80 GB GPU with 4-bit weights? (3) A 7 B model with 4-bit weights, on a GPU that reads 300 GB per second: about how many tokens per second can **one stream** of text get, at most? Write your guesses, then see how close you were.

# %%
# Imagine first: memory, cost and speed of a deployment
guess_70b_fits_one_gpu = None    # YOUR GUESS: "yes" or "no" (70 B in 16-bit numbers on one 80 GB GPU?)
guess_cheaper = None             # YOUR GUESS: "four GPUs fp16" or "one GPU int4" (cheaper per million tokens?)
guess_tokens_s = None            # YOUR GUESS: tokens per second for one stream of a 7 B model, 4-bit weights, 300 GB/s (upper bound)

# %% [markdown]
# ### Step 7.1. The memory of a deployment
#
# A deployment has to hold two things in GPU memory all the time: the **weights** (parameters times bytes per parameter, Part 1 and Part 5) and the **KV cache** (2 · layers · tokens · d · bytes per value, for every request in the batch, Part 6). The formulas of the lesson were `weights = parameters × bits / 8` and `KV = 2 × L × n × d × bytes × batch`. Large models today usually share each key and value among several query heads (**grouped-query attention**), which makes the cache smaller by a factor `kv_share` (for example 8 key-value heads for 64 query heads is `kv_share = 0.125`); for a plain multi-head model `kv_share = 1`.
#
# *Exercise 7.1:* Write `p7_memory(params_b, w_bits, kv_bits, ctx, batch, layers, d_model, kv_share=1.0)`. It returns the pair `(weights_gb, kv_gb)` in gigabytes (1 GB is 10⁹ bytes; `params_b` is in billions). Why it matters: every decision in this part starts from these two numbers, and you have measured both in earlier parts. The check compares your function with the lesson's figures (7 B: 28, 14, 7 and 3.5 GB; 70 B: 140 GB) and with a hand computation of a cache.

# %%
def p7_memory(params_b, w_bits, kv_bits, ctx, batch, layers, d_model, kv_share=1.0):
    """Return (weights_gb, kv_gb)."""
    return None   # YOUR CODE: weights from the parameters and bits; KV cache from layers, tokens, d, bytes per value, batch and kv_share

# %%
if p7_memory(7, 16, 16, 4096, 8, 32, 4096) is None:
    print("Fill in p7_memory above, then run this cell again.")
else:
    w_ = [p7_memory(7, b_, 16, 1, 1, 32, 4096)[0] for b_ in (32, 16, 8, 4)]
    check("7 B at 32, 16, 8 and 4 bits is 28, 14, 7 and 3.5 GB", np.allclose(w_, [28, 14, 7, 3.5]), "weights = parameters (billions) times bits divided by 8, in GB")
    check("70 B at 16 bits is 140 GB", abs(p7_memory(70, 16, 16, 1, 1, 80, 8192)[0] - 140) < 1e-9, "70 times 16 divided by 8")
    check("KV cache of 7 B, 4,096 tokens, batch 8, 16-bit is 17.18 GB", abs(p7_memory(7, 16, 16, 4096, 8, 32, 4096)[1] - 17.179869184) < 1e-6, "2 * 32 * 4096 * 4096 * 2 bytes * 8, divided by 1e9")
    check("kv_share 0.125 makes the cache 8 times smaller", abs(p7_memory(7, 16, 16, 4096, 8, 32, 4096, kv_share=0.125)[1] * 8 - 17.179869184) < 1e-6, "multiply the cache by kv_share")
    check("the cache grows with the batch, the weights do not", p7_memory(7, 16, 16, 4096, 16, 32, 4096)[1] > p7_memory(7, 16, 16, 4096, 8, 32, 4096)[1] and p7_memory(7, 16, 16, 4096, 16, 32, 4096)[0] == p7_memory(7, 16, 16, 4096, 8, 32, 4096)[0], "only the cache depends on the batch and the context")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def p7_memory(params_b, w_bits, kv_bits, ctx, batch, layers, d_model, kv_share=1.0):
#     weights_gb = params_b * 1e9 * w_bits / 8 / 1e9
#     kv_gb = 2 * layers * ctx * d_model * (kv_bits / 8) * batch * kv_share / 1e9
#     return weights_gb, kv_gb
# ```
#
# The factor 2 is for keys and values, `layers * ctx * d_model` is the number of stored values per request and sequence position, `kv_bits / 8` converts to bytes, and the batch multiplies because every request keeps its own cache.
#
# </details>

# %% [markdown]
# <!-- nb-only -->
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise, and then the check on it.

# %%
def p7_memory(params_b, w_bits, kv_bits, ctx, batch, layers, d_model, kv_share=1.0):
    weights_gb = params_b * 1e9 * w_bits / 8 / 1e9
    kv_gb = 2 * layers * ctx * d_model * (kv_bits / 8) * batch * kv_share / 1e9
    return weights_gb, kv_gb
w_ = [p7_memory(7, b_, 16, 1, 1, 32, 4096)[0] for b_ in (32, 16, 8, 4)]
check("7 B at 32, 16, 8 and 4 bits is 28, 14, 7 and 3.5 GB", np.allclose(w_, [28, 14, 7, 3.5]), "weights = parameters (billions) times bits divided by 8, in GB")
check("70 B at 16 bits is 140 GB", abs(p7_memory(70, 16, 16, 1, 1, 80, 8192)[0] - 140) < 1e-9, "70 times 16 divided by 8")
check("KV cache of 7 B, 4,096 tokens, batch 8, 16-bit is 17.18 GB", abs(p7_memory(7, 16, 16, 4096, 8, 32, 4096)[1] - 17.179869184) < 1e-6, "2 * 32 * 4096 * 4096 * 2 bytes * 8, divided by 1e9")
check("kv_share 0.125 makes the cache 8 times smaller", abs(p7_memory(7, 16, 16, 4096, 8, 32, 4096, kv_share=0.125)[1] * 8 - 17.179869184) < 1e-6, "multiply the cache by kv_share")
check("the cache grows with the batch, the weights do not", p7_memory(7, 16, 16, 4096, 16, 32, 4096)[1] > p7_memory(7, 16, 16, 4096, 8, 32, 4096)[1] and p7_memory(7, 16, 16, 4096, 16, 32, 4096)[0] == p7_memory(7, 16, 16, 4096, 8, 32, 4096)[0], "only the cache depends on the batch and the context")

# %% [markdown]
# ### Step 7.2. The capacity planner
#
# Now we add the other two limits of the lesson. **Speed:** decoding reads all the weights and the cache for every token, so with total memory bandwidth `bw × N` the step takes at least `(weights + KV) / (bw × N)` seconds, and one stream gets at most the inverse of that in tokens per second. When the model is split over N GPUs with tensor parallelism (Part 3) every layer needs two all-reduces of `batch × d × 2` bytes, with traffic `2(N − 1)/N × S` (Part 2); we add that time using the link speed. **Cost:** a real server reaches only a fraction of the bound, which we call `eff` and assume to be 0.5; the price per million tokens is `GPUs × price per hour / (tokens per second × 3600) × 10⁶`.
#
# The function below does all this and gives **advice**. Read it before you run it: every rule is one you can argue with. The 10 percent that `reserve` keeps free is an assumed allowance for activations and the framework (Part 1), and the threshold of 100 GB/s for a link that is fast enough for tensor parallelism is also assumed. The function uses the `p7_memory` of Step 7.1.

# %%
import math

def p7_plan(params_b, w_bits, kv_bits, ctx, batch, gpu_gb, bw_gb_s, n_gpus, link_gb_s,
            layers=32, d_model=4096, kv_share=1.0, price_gpu_hr=2.0, eff=0.5, reserve=0.10):
    """Capacity plan for serving one model. Every hardware number is an ASSUMPTION you pass in."""
    weights_gb, kv_gb = p7_memory(params_b, w_bits, kv_bits, ctx, batch, layers, d_model, kv_share)
    total_gb = weights_gb + kv_gb
    usable_gb = n_gpus * gpu_gb * (1 - reserve)
    fits = total_gb <= usable_gb
    step_s = total_gb / (bw_gb_s * n_gpus)                                  # time to read weights and cache once
    comm_s = 0.0
    if n_gpus > 1 and link_gb_s > 0 and layers > 0:                         # two all-reduces per layer when tensor-parallel
        size_gb = batch * d_model * 2 / 1e9
        comm_s = 2 * layers * (2 * (n_gpus - 1) / n_gpus) * size_gb / link_gb_s
    stream_bound = 1 / (step_s + comm_s)                                    # tokens/s of one stream, upper bound
    total_tps = batch * stream_bound * eff                                  # expected tokens/s for the whole batch
    cost = n_gpus * price_gpu_hr / (total_tps * 3600) * 1e6
    if not fits:
        need = math.ceil(total_gb / (gpu_gb * (1 - reserve)))
        advice = "does not fit: needs at least %d GPUs of this size, or fewer bits, a shorter context or a smaller batch" % need
    elif total_gb <= gpu_gb * (1 - reserve):
        advice = "fits on one GPU: keep the model whole and add replicas (data parallel) when more users arrive"
    elif link_gb_s >= 100:
        advice = "split over %d GPUs with tensor parallelism (the link is fast enough for two all-reduces per layer)" % n_gpus
    else:
        advice = "link too slow for tensor parallelism: use pipeline stages and micro-batches between the boxes"
    return {"weights_gb": weights_gb, "kv_gb": kv_gb, "total_gb": total_gb, "usable_gb": usable_gb, "fits": fits,
            "comm_ms": comm_s * 1000, "stream_tps": stream_bound, "total_tps": total_tps, "cost_per_m": cost, "advice": advice}

# Sanity checks against numbers you can do by hand
a_ = p7_plan(7, 16, 16, 4096, 8, gpu_gb=80, bw_gb_s=2000, n_gpus=1, link_gb_s=0)
check("one stream of 7 B 16-bit, 1 GPU of 2,000 GB/s: bound is 2000 / (14 + 17.18) tokens/s", abs(a_["stream_tps"] - 2000 / (14 + 17.179869184)) < 1e-6, "no communication on one GPU")
check("a 70 B model in 16 bits does not fit on one 80 GB GPU", not p7_plan(70, 16, 16, 1, 1, 80, 2000, 1, 0, layers=80, d_model=8192)["fits"], "140 GB is more than 72 GB usable")
check("more GPUs add all-reduce time", p7_plan(7, 16, 16, 4096, 8, 80, 2000, 4, 300)["comm_ms"] > 0 and p7_plan(7, 16, 16, 4096, 8, 80, 2000, 1, 300)["comm_ms"] == 0, "communication only with N above 1")
print("one stream bound for the check case: %.1f tokens/s; expected for the batch of 8 at eff 0.5: %.1f tokens/s; cost %.2f per million tokens (assumed price)" % (a_["stream_tps"], a_["total_tps"], a_["cost_per_m"]))

# %% [markdown]
# ### Step 7.3. Three deployments
#
# Now we plan three deployments. **All hardware numbers are assumed**: `MY_BOX` is where you put the numbers of *your* machine from the hardware report of the Setup (the memory of your GPU, or of your RAM if you have no GPU, and a bandwidth you measured in Part 5 or looked up in the documentation of your hardware). We use deliberately round values that are not those of any product.
#
# 1. **The school helpdesk model** (7 B, 4-bit weights with scales, so 4.25 bits per weight, 2,048 tokens of context, 4 requests at once) on **one box**.
# 2. **A 70 B model for a district** (16-bit weights, 4,096 tokens, 16 requests at once, grouped-query attention with `kv_share = 0.125`, 80 GB GPUs): on 4 GPUs with 16-bit weights, and on 1 GPU with 4-bit weights.
# 3. **The Structural Inspection Assistant on the school's server**: a small image classifier (assumed 25 million parameters, 32-bit) **plus** the same 7 B language model at 4.25 bits with 4,096 tokens, one request at a time. Both must fit in the same memory.
#
# *Exercise 7.3:* Run the three deployments through `p7_plan` and print one line each. Fill in `answers_73` with the strings `"fits"` or `"does not fit"` for the helpdesk box (key `"helpdesk"`), for the 70 B model on 4 GPUs in 16 bits (`"district_fp16"`) and for the 70 B model on one GPU in 4 bits (`"district_int4"`). Why it matters: the planner is only useful if you read its verdict before you buy anything. The check compares your three words with the planner.

# %%
MY_BOX = {"gpu_gb": 24, "bw_gb_s": 300, "price_gpu_hr": 1.0}      # ASSUMED: replace with the numbers of your own hardware report

P7_SCEN = {
    "helpdesk":      dict(params_b=7,  w_bits=4.25, kv_bits=16, ctx=2048, batch=4,  n_gpus=1, link_gb_s=0,   layers=32, d_model=4096, **MY_BOX),
    "district_fp16": dict(params_b=70, w_bits=16,   kv_bits=16, ctx=4096, batch=16, n_gpus=4, link_gb_s=300, layers=80, d_model=8192, kv_share=0.125, gpu_gb=80, bw_gb_s=2000, price_gpu_hr=2.0),
    "district_int4": dict(params_b=70, w_bits=4.25, kv_bits=16, ctx=4096, batch=16, n_gpus=1, link_gb_s=0,   layers=80, d_model=8192, kv_share=0.125, gpu_gb=80, bw_gb_s=2000, price_gpu_hr=2.0),
}
p7_res = {k_: p7_plan(**v_) for k_, v_ in P7_SCEN.items()}
print("%-14s %7s %6s %7s %5s %8s %9s %9s" % ("deployment", "weights", "KV", "total", "fits", "stream", "batch", "$ per M"))
for k_, r_ in p7_res.items():
    print("%-14s %5.1f G %4.1f G %5.1f G %5s %6.1f t/s %7.1f t/s %9.2f" % (k_, r_["weights_gb"], r_["kv_gb"], r_["total_gb"], "yes" if r_["fits"] else "NO", r_["stream_tps"], r_["total_tps"], r_["cost_per_m"]))
for k_, r_ in p7_res.items(): print("  %-14s %s" % (k_, r_["advice"]))
answers_73 = {"helpdesk": None, "district_fp16": None, "district_int4": None}    # YOUR ANSWERS: "fits" or "does not fit"

# %%
if None in answers_73.values():
    print("Fill in answers_73 above, then run this cell again.")
else:
    for k_ in ("helpdesk", "district_fp16", "district_int4"):
        check("%s: %s" % (k_, answers_73[k_]), answers_73[k_] == ("fits" if p7_res[k_]["fits"] else "does not fit"), "compare total with usable memory (90 percent of the GPUs)")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answers_73 = {"helpdesk": "fits", "district_fp16": "fits", "district_int4": "fits"}
# ```
#
# All three fit, but for different reasons: the helpdesk model is small, the 16-bit district model needs all four GPUs, and the 4-bit one fits on a single GPU because the weights are four times smaller. If you change `MY_BOX` to a smaller memory, the first answer changes: that is the experiment to try.
#
# </details>

# %% [markdown]
# <!-- nb-only -->
# Below the exercise we run our version of the code above, then the check on it.

# %%
answers_73 = {"helpdesk": "fits", "district_fp16": "fits", "district_int4": "fits"}
for k_ in ("helpdesk", "district_fp16", "district_int4"):
    check("%s: %s" % (k_, answers_73[k_]), answers_73[k_] == ("fits" if p7_res[k_]["fits"] else "does not fit"), "compare total with usable memory (90 percent of the GPUs)")
RESULTS["p7_cost_fp16_4gpu"] = round(p7_res["district_fp16"]["cost_per_m"], 2)
RESULTS["p7_cost_int4_1gpu"] = round(p7_res["district_int4"]["cost_per_m"], 2)

# %%
# Step 7.3 (continued). The Structural Inspection Assistant: classifier plus language model in the same memory
cls_ = p7_plan(0.025, 32, 16, 1, 1, layers=0, d_model=1, gpu_gb=16, bw_gb_s=200, n_gpus=1, link_gb_s=0)
llm_ = p7_plan(7, 4.25, 16, 4096, 1, gpu_gb=16, bw_gb_s=200, n_gpus=1, link_gb_s=0, layers=32, d_model=4096, price_gpu_hr=1.0)
tot_ = cls_["weights_gb"] + llm_["total_gb"]
print("classifier weights %.3f GB | language model weights %.2f GB + KV cache %.2f GB | together %.2f GB of %.1f GB usable -> %s" % (cls_["weights_gb"], llm_["weights_gb"], llm_["kv_gb"], tot_, 16 * 0.9, "fits" if tot_ <= 16 * 0.9 else "does not fit"))
print("language model alone: stream bound %.1f tokens/s, expected %.1f tokens/s, about %.2f per million tokens (assumed price)" % (llm_["stream_tps"], llm_["total_tps"], llm_["cost_per_m"]))
check("the classifier is tiny next to the language model", cls_["weights_gb"] < 0.05 * llm_["weights_gb"], "100 MB against 3.7 GB")
check("the two models fit together in the assumed 16 GB", tot_ <= 16 * 0.9, "lower the context or the bits")
RESULTS["p7_inspection_gb"] = round(tot_, 2)
w_only_ = MY_BOX["bw_gb_s"] / p7_memory(7, 4.25, 16, 1, 1, 32, 4096)[0]
print("guess 3 of the Imagine step: weights alone (3.72 GB) at %d GB/s give at most %.1f tokens/s for one stream; with the KV cache of the helpdesk plan %.1f" % (MY_BOX["bw_gb_s"], w_only_, p7_res["helpdesk"]["stream_tps"]))

# %% [markdown]
# **What you should see:**
#
# # The planner table, one line per deployment, then the advice lines, then the three ticks of the check (or the `Fill in answers_73` message until you fill it in):
#
# ```text
# deployment     weights     KV   total  fits   stream     batch   $ per M
# helpdesk         3.7 G  4.3 G   8.0 G   yes   37.4 t/s    74.9 t/s      3.71
# district_fp16  140.0 G 21.5 G 161.5 G   yes   49.0 t/s   392.3 t/s      5.67
# district_int4   37.2 G 21.5 G  58.7 G   yes   34.1 t/s   272.7 t/s      2.04
# ```
#
# and for the inspection assistant: `classifier weights 0.100 GB | language model weights 3.72 GB + KV cache 2.15 GB | together 5.97 GB of 14.4 GB usable -> fits`, then `language model alone: stream bound 34.1 tokens/s, expected 17.0 tokens/s, about 16.30 per million tokens (assumed price)`, two ticks, and `weights alone (3.72 GB) at 300 GB/s give at most 80.7 tokens/s for one stream; with the KV cache of the helpdesk plan 37.4`.
#
# *What it means.* The three guesses of the Imagine step: (1) **no**, 70 B in 16 bits needs 140 GB and one 80 GB GPU offers 72 GB after the 10 percent reserve; (2) **one GPU with 4-bit weights** was cheaper in this model (2.04 against 5.67 per million tokens at the assumed prices), because it needs a quarter of the GPUs and has no all-reduce, even though each stream is a little slower (34 against 49 tokens per second); (3) about 80 tokens per second if you count only the weights, and about 37 once the cache of four requests is read as well: the cache is not free. The prices are only as good as the assumptions, and the accuracy of the 4-bit model is **not** in this table: Part 5 showed that you must measure it. The helpdesk and the inspection assistant fit on a small box; the 70 B model needs either four GPUs and a fast link, or quantization.

# %% [markdown]
# ### Step 7.4. What moves the cost? Batch size and bits
#
# The planner says that cost per million tokens falls when more requests share one pass over the weights (Part 6), until memory runs out because every request needs its own KV cache. Let us find that edge for the helpdesk model. We sweep the batch size for 4-bit and for 16-bit weights on the assumed box, and find the largest batch that fits. **Caution:** the planner's speed bound assumes that decoding stays memory-bound; at large batches the arithmetic starts to limit too (Part 6 measured `t₀ + c·b`), so the real cost curve flattens earlier than this one.

# %%
def p7_max_batch(**kw):
    kw = dict(kw); b_ = 1
    while b_ < 4096 and p7_plan(**dict(kw, batch=b_ + 1))["fits"]: b_ += 1
    return b_ if p7_plan(**dict(kw, batch=b_))["fits"] else 0

base_ = dict(P7_SCEN["helpdesk"])
sweep_ = {}
for bits_ in (4.25, 16):
    kw_ = dict(base_, w_bits=bits_)
    sweep_[bits_] = {"max_batch": p7_max_batch(**kw_), "rows": [(b_, p7_plan(**dict(kw_, batch=b_))) for b_ in (1, 2, 4, 8, 16, 32, 64)]}
print("%-6s %-5s %8s %6s %10s %8s" % ("bits", "batch", "total", "fits", "tokens/s", "$ per M"))
for bits_, d_ in sweep_.items():
    for b_, r_ in d_["rows"]:
        print("%-6s %-5d %6.1f G %6s %10.1f %8.2f" % (bits_, b_, r_["total_gb"], "yes" if r_["fits"] else "NO", r_["total_tps"], r_["cost_per_m"]))
    print("largest batch that fits with %s-bit weights: %d" % (bits_, d_["max_batch"]))
fig_, ax_ = plt.subplots(figsize=(6, 3.2))
for bits_, d_ in sweep_.items():
    ok_ = [(b_, r_["cost_per_m"]) for b_, r_ in d_["rows"] if r_["fits"] ]
    ax_.plot([x for x, _ in ok_], [y for _, y in ok_], "o-", label="%s-bit weights (stops at the memory limit)" % bits_)
ax_.set_xscale("log", base=2); ax_.set_yscale("log"); ax_.set_xlabel("batch size"); ax_.set_ylabel("assumed price per million tokens"); ax_.legend(fontsize=8); ax_.grid(alpha=.3)
plt.tight_layout(); plt.show()
RESULTS["p7_max_batch_int4"] = sweep_[4.25]["max_batch"]; RESULTS["p7_max_batch_fp16"] = sweep_[16]["max_batch"]
check("4-bit weights leave room for a larger batch than 16-bit weights", sweep_[4.25]["max_batch"] > sweep_[16]["max_batch"], "smaller weights leave more memory for the cache")
falls_ = []
for bits_, d_ in sweep_.items():
    fit_costs_ = [r_["cost_per_m"] for _, r_ in d_["rows"] if r_["fits"] ]
    falls_.append(fit_costs_[0] > fit_costs_[-1])
check("cost per million tokens falls as the batch grows (while it fits)", all(falls_), "one pass over the weights serves more requests")

# %% [markdown]
# **What you should see:**
#
# # The table of batch sizes for 4.25-bit and 16-bit weights, `largest batch that fits with 4.25-bit weights: 16` and `largest batch that fits with 16-bit weights: 7`, a plot of the price against the batch, and two ticks. Excerpt:
#
# ```text
# 4.25   1        4.8 G    yes       31.3     8.87
# 4.25   16      20.9 G    yes      114.8     2.42
# 4.25   32      38.1 G     NO      126.1     2.20
# 16     1       15.1 G    yes       10.0    27.91
# 16     4       18.3 G    yes       32.8     8.47
# 16     8       22.6 G     NO       53.1     5.23
# ```
#
# *What it means.* On the assumed 24 GB box, going from batch 1 to batch 16 cuts the price by a factor of 3.7 (8.87 to 2.42) with 4-bit weights, and the 4-bit model fits a batch of 16 while the 16-bit one stops at 7: smaller weights leave room for more caches, so quantization buys both a smaller model **and** a cheaper token. The rows marked `NO` are only a calculation: they show where memory would run out. The curve flattens because the cache read starts to dominate, and in a real server it would flatten even earlier, because at a large batch the arithmetic limits the speed as well (the line `t₀ + c·b` of Part 6).

# %% [markdown]
# ### Step 7.5. The deployment checklist and your design card
#
# A number from a planner is not a deployment. A deployment also has a **precision** and a **parallelism** decision, a **batching** policy, a **service level objective** (SLO: for example "95 percent of answers in under 3 seconds", the p95 of Part 6), **monitoring**, **versioning**, a **rollback** plan, **privacy and guardrails** (Week 17) and a **cost**. The function below fills in the checklist for the helpdesk from the planner result and prints it as a markdown table, so you can paste it into a report.

# %%
CHECK_FIELDS = ("precision", "parallelism", "batching", "slo", "monitoring", "versioning", "rollback", "privacy_guardrails", "cost")

def p7_checklist(name, plan, slo_p95_s):
    return {
        "precision": "4.25-bit weights (4 bits plus a scale for every group of 64, Part 5) and a 16-bit KV cache; accuracy rechecked on our own questions after quantizing.",
        "parallelism": "none: the whole model uses %.1f GB of %.1f GB, so we keep one copy per box and add replicas when users grow (data parallel)." % (plan["total_gb"], MY_BOX["gpu_gb"]),
        "batching": "the server waits at most 50 ms to form a batch of up to %d requests (Part 6), because a larger batch lowers the cost but raises the wait." % RESULTS.get("p7_max_batch_int4", 8),
        "slo": "95 percent of answers (p95) in under %.0f seconds, measured over each week, and fewer than 1 request in 100 dropped." % slo_p95_s,
        "monitoring": "log the p50, p95 and p99 latency, the queue length, the GPU memory in use and the share of answers that the Week 17 rails blocked, and alert when p95 is above the SLO.",
        "versioning": "every model file carries a name, the quantization settings and a checksum, and the evaluation results of that file are stored next to it.",
        "rollback": "keep the previous model file loaded on a second box; if p95 or the blocked share doubles for 10 minutes, send the traffic back to it by hand.",
        "privacy_guardrails": "the input, tool and output rails of Week 17 stay on, a human reads a sample of 20 conversations every Friday, and logs hide names and are deleted after 30 days.",
        "cost": "about %.2f per million tokens at the assumed price of %.2f per GPU hour and half of the speed bound (assumed)." % (plan["cost_per_m"], MY_BOX["price_gpu_hr"]),
    }

ref_card = p7_checklist("helpdesk", p7_res["helpdesk"], 3)
print("| item | decision |\n|---|---|")
for f_ in CHECK_FIELDS: print("| %s | %s |" % (f_, ref_card[f_]))

# %% [markdown]
# *Exercise 7.5:* Fill in `my_card` for **your own deployment** (for example the helpdesk of your school, a study helper or the Structural Inspection Assistant) with the nine fields of `CHECK_FIELDS`, at least 8 words in each. Use the planner for your memory and cost numbers. Why it matters: the card makes you decide the monitoring, the rollback and the human step before the first user arrives. The check tests the structure: the `slo` must contain a number and the word p95, the `cost` a number, the `precision` a number format, and the `privacy_guardrails` one of the protections of Week 17. It never judges the content.

# %%
my_card = {f_: "" for f_ in CHECK_FIELDS}      # YOUR TEXT: nine fields, each at least 8 words

# %%
if not any(my_card.values()):
    print("Fill in my_card above, then run this cell again.")
else:
    check("every field has at least 8 words", all(len(my_card[f_].split()) >= 8 for f_ in CHECK_FIELDS), "write a full sentence in each field")
    check("slo has a number and p95", bool(re.search(r"\d", my_card["slo"])) and "p95" in my_card["slo"].lower(), "which share of requests, how fast?")
    check("cost has a number", bool(re.search(r"\d", my_card["cost"])), "price per million tokens or per month")
    check("precision names a number format", bool(re.search(r"int\d|fp\d|bf16|bit|float", my_card["precision"].lower())), "which bits for the weights?")
    check("privacy_guardrails names a protection of Week 17", bool(re.search(r"rail|human|confirm|log|permission|allow", my_card["privacy_guardrails"].lower())), "which rail, which human step?")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# The reference card printed above is one possible answer for the helpdesk. A shorter one, for the inspection assistant:
#
# ```python
# my_card = {
#     "precision": "The language model uses 4-bit weights with groups of 64 and the classifier stays in 32-bit floats, rechecked on 200 test photos.",
#     "parallelism": "None: classifier and language model together need about 6 GB of the 16 GB server, so one copy on one box is enough.",
#     "batching": "One request at a time, because an engineer waits for each draft, so we choose a low wait over a low cost.",
#     "slo": "95 percent of drafts (p95) in under 20 seconds and 0 verdicts released without an engineer's approval.",
#     "monitoring": "Log the confidence of every photo, the latency p95, the share of drafts with a verdict removed by the rail, and the engineer's rejections.",
#     "versioning": "The classifier and the language model files are versioned together, with the test results of that pair stored next to them.",
#     "rollback": "Keep the previous pair on disk and switch back with one command if the rejection rate of the engineer doubles in a week.",
#     "privacy_guardrails": "The Week 17 verdict rail stays on, the licensed engineer approves every record, and photos are deleted after 90 days.",
#     "cost": "About 16 per million tokens at the assumed price of 1.00 per GPU hour, which is small because a draft is only a few hundred tokens.",
# }
# ```
#
# </details>

# %% [markdown]
# <!-- nb-only -->
# Below the exercise we run our version of the code above, then the check on it.

# %%
my_card = {
    "precision": "The language model uses 4-bit weights with groups of 64 and the classifier stays in 32-bit floats, rechecked on 200 test photos.",
    "parallelism": "None: classifier and language model together need about 6 GB of the 16 GB server, so one copy on one box is enough.",
    "batching": "One request at a time, because an engineer waits for each draft, so we choose a low wait over a low cost.",
    "slo": "95 percent of drafts (p95) in under 20 seconds and 0 verdicts released without an engineer's approval.",
    "monitoring": "Log the confidence of every photo, the latency p95, the share of drafts with a verdict removed by the rail, and the engineer's rejections.",
    "versioning": "The classifier and the language model files are versioned together, with the test results of that pair stored next to them.",
    "rollback": "Keep the previous pair on disk and switch back with one command if the rejection rate of the engineer doubles in a week.",
    "privacy_guardrails": "The Week 17 verdict rail stays on, the licensed engineer approves every record, and photos are deleted after 90 days.",
    "cost": "About 16 per million tokens at the assumed price of 1.00 per GPU hour, which is small because a draft is only a few hundred tokens.",
}
check("every field has at least 8 words", all(len(my_card[f_].split()) >= 8 for f_ in CHECK_FIELDS), "write a full sentence in each field")
check("slo has a number and p95", bool(re.search(r"\d", my_card["slo"])) and "p95" in my_card["slo"].lower(), "which share of requests, how fast?")
check("cost has a number", bool(re.search(r"\d", my_card["cost"])), "price per million tokens or per month")
check("precision names a number format", bool(re.search(r"int\d|fp\d|bf16|bit|float", my_card["precision"].lower())), "which bits for the weights?")
check("privacy_guardrails names a protection of Week 17", bool(re.search(r"rail|human|confirm|log|permission|allow", my_card["privacy_guardrails"].lower())), "which rail, which human step?")

# %% [markdown]
# ### Step 7.6. Your honest findings table, and what comes next
#
# The most useful page of a report is the one with the experiments that did not work. In the first cell below, copy the numbers you measured in Parts 1 to 6 into `my_numbers` (the earlier parts print them; replace each `None`). The cell then prints them next to the numbers of this part.
#
# *Exercise 7.6:* Fill in `findings` with at least 6 dictionaries with the keys `tried`, `number` and `verdict`, drawing rows from Parts 1 to 7. Include at least one result that did **not** work or surprised you (on a CPU-only machine, half precision is the obvious candidate), at least one row about the communication or the bubble (Part 2 or 3), and one row from Part 7. Why it matters: a row with a number and an honest verdict is what makes a deployment decision checkable. The check tests the structure only.

# %%
my_numbers = {   # copy these from your own output of Parts 1 to 6
    "training memory per parameter in bytes, measured (Part 1)": None,
    "gradient average equals the full-batch gradient: largest difference (Part 2)": None,
    "all-reduce bandwidth on this machine in GB/s, and DDP speed-up (Part 2)": None,
    "tensor-parallel split equals the full block: largest difference (Part 3)": None,
    "pipeline bubble: formula and simulation for your p and m (Part 3)": None,
    "bfloat16 against float32 time on this CPU, and on a GPU if you have one (Part 4)": None,
    "int8 dynamic quantization: size, speed-up and output error (Part 5)": None,
    "batch size where throughput stops improving, and the KV cache bytes (Part 6)": None,
}
print("%-82s %s" % ("what", "your number"))
for k_, v_ in my_numbers.items(): print("%-82s %s" % (k_, v_))
print("%-82s %s" % ("70 B: price per million tokens, 4 GPUs 16-bit / 1 GPU 4-bit (Part 7, assumed prices)", "%s / %s" % (RESULTS.get("p7_cost_fp16_4gpu"), RESULTS.get("p7_cost_int4_1gpu"))))
print("%-82s %s" % ("helpdesk: largest batch that fits, 4-bit / 16-bit weights (Part 7)", "%s / %s" % (RESULTS.get("p7_max_batch_int4"), RESULTS.get("p7_max_batch_fp16"))))
findings = []      # YOUR ROWS: {"tried": "...", "number": "...", "verdict": "..."}

# %%
if len(findings) == 0:
    print("Fill in findings above, then run this cell again.")
else:
    ok_keys = all(isinstance(x_, dict) and {"tried", "number", "verdict"} <= set(x_) for x_ in findings)
    check("at least 6 rows", len(findings) >= 6, "draw rows from Parts 1 to 7")
    check("every row has tried, number and verdict", ok_keys, "three keys in each dictionary")
    if ok_keys:
        check("every number cell contains a digit", all(re.search(r"\d", str(x_["number"])) for x_ in findings), "write the number you measured")
        check("at least one verdict says it did not work or surprised you", any(re.search(r"did not|not work|surpris|slower|worse|no real|no speed|only|failed", str(x_["verdict"]).lower()) for x_ in findings), "include one honest negative result")
        check("at least one row is about communication or the bubble", any(re.search(r"all-reduce|bubble|communic|bandwidth", (str(x_["tried"]) + str(x_["verdict"])).lower()) for x_ in findings), "one row from Part 2 or 3")
        check("at least one row is about Part 7", any(re.search(r"planner|cost|checklist|card|per million|deployment", (str(x_["tried"]) + str(x_["verdict"])).lower()) for x_ in findings), "one row from this part")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# findings = [
#     {"tried": "mixed precision on this CPU (Part 4)", "number": "your Part 4 ratio of bfloat16 time to float32 time", "verdict": "it did not speed anything up here, because this CPU has no fast 16-bit arithmetic; a GPU with Tensor Cores behaves differently"},
#     {"tried": "int8 dynamic quantization of the wide MLP (Part 5)", "number": "your Part 5 size ratio and speed-up", "verdict": "4 times smaller and faster on this machine, with a small output error: quantization helped where mixed precision did not"},
#     {"tried": "all-reduce between processes (Part 2)", "number": "your Part 2 bandwidth in GB/s and DDP speed-up", "verdict": "no real speed-up with gloo on one CPU, because the processes share the same cores and memory; the formula still explains the traffic"},
#     {"tried": "the pipeline bubble (Part 3)", "number": "your Part 3 formula and simulated idle share", "verdict": "they matched for equal stages, and unequal stages made it worse, because the slowest stage sets the pace"},
#     {"tried": "batching a server (Part 6)", "number": "your Part 6 batch size and KV cache bytes", "verdict": "throughput grew with the batch until memory or arithmetic became the limit, at the price of a longer wait for each request"},
#     {"tried": "the capacity planner for a 70 B model (Part 7)", "number": "%s per million tokens with 4 GPUs in 16 bits, %s with 1 GPU in 4 bits (assumed prices)" % (RESULTS.get("p7_cost_fp16_4gpu"), RESULTS.get("p7_cost_int4_1gpu")), "verdict": "4-bit weights were cheaper because they need fewer GPUs, but I must recheck the accuracy on my own questions before I believe it"},
# ]
# print("%-48s | %-44s | %s" % ("tried", "number", "verdict"))
# for x in findings: print("%-48s | %-44s | %s" % (x["tried"][:48], x["number"][:44], x["verdict"][:90]))
# ```
#
# Rows that say "your Part N" must be replaced by what *you* measured. Keep the negative rows: they are the ones a reader trusts.
#
# </details>

# %% [markdown]
# <!-- nb-only -->
# Below the exercise we run our version of the code above (with the numbers of this session where this part measured them), then the check on it.

# %%
findings = [
    {"tried": "mixed precision on this CPU (Part 4)", "number": "your Part 4 ratio of bfloat16 time to float32 time", "verdict": "it did not speed anything up here, because this CPU has no fast 16-bit arithmetic; a GPU with Tensor Cores behaves differently"},
    {"tried": "int8 dynamic quantization of the wide MLP (Part 5)", "number": "your Part 5 size ratio and speed-up", "verdict": "4 times smaller and faster on this machine, with a small output error: quantization helped where mixed precision did not"},
    {"tried": "all-reduce between processes (Part 2)", "number": "your Part 2 bandwidth in GB/s and DDP speed-up", "verdict": "no real speed-up with gloo on one CPU, because the processes share the same cores and memory; the formula still explains the traffic"},
    {"tried": "the pipeline bubble (Part 3)", "number": "your Part 3 formula and simulated idle share", "verdict": "they matched for equal stages, and unequal stages made it worse, because the slowest stage sets the pace"},
    {"tried": "batching a server (Part 6)", "number": "your Part 6 batch size and KV cache bytes", "verdict": "throughput grew with the batch until memory or arithmetic became the limit, at the price of a longer wait for each request"},
    {"tried": "the capacity planner for a 70 B model (Part 7)", "number": "%s per million tokens with 4 GPUs in 16 bits, %s with 1 GPU in 4 bits (assumed prices)" % (RESULTS.get("p7_cost_fp16_4gpu"), RESULTS.get("p7_cost_int4_1gpu")), "verdict": "4-bit weights were cheaper because they need fewer GPUs, but I must recheck the accuracy on my own questions before I believe it"},
]
print("%-48s | %-44s | %s" % ("tried", "number", "verdict"))
for x_ in findings: print("%-48s | %-44s | %s" % (x_["tried"][:48], x_["number"][:44], x_["verdict"][:90]))
ok_keys = all(isinstance(x_, dict) and {"tried", "number", "verdict"} <= set(x_) for x_ in findings)
check("at least 6 rows", len(findings) >= 6, "draw rows from Parts 1 to 7")
check("every row has tried, number and verdict", ok_keys, "three keys in each dictionary")
check("every number cell contains a digit", all(re.search(r"\d", str(x_["number"])) for x_ in findings), "write the number you measured")
check("at least one verdict says it did not work or surprised you", any(re.search(r"did not|not work|surpris|slower|worse|no real|no speed|only|failed", str(x_["verdict"]).lower()) for x_ in findings), "include one honest negative result")
check("at least one row is about communication or the bubble", any(re.search(r"all-reduce|bubble|communic|bandwidth", (str(x_["tried"]) + str(x_["verdict"])).lower()) for x_ in findings), "one row from Part 2 or 3")
check("at least one row is about Part 7", any(re.search(r"planner|cost|checklist|card|per million|deployment", (str(x_["tried"]) + str(x_["verdict"])).lower()) for x_ in findings), "one row from this part")

# %% [markdown]
# ### Reflection questions
#
# Write two or three sentences for each, in a markdown cell, using your own numbers.
#
# 1. Which of the two limits, memory or time, decided your helpdesk plan, and which number in the planner showed it?
# 2. Mixed precision, quantization, batching, tensor parallelism, pipelines and data parallelism each solve a different problem. Pick the two that mattered most for **your** hardware and say why the others did not.
# 3. Which result of this week did you not expect, and what did you do to check that it was real and not a mistake in your code?
# 4. What would you measure on the real hardware before you promise a school a response time (the p95 of Part 6)?
# 5. Which of your assumed numbers (a price, a bandwidth, the 50 percent efficiency) would change the decision most if it were wrong by a factor of two? Test it by changing it in `MY_BOX` and running Step 7.3 again.
#
# ### What comes next
#
# You now have the whole chain of the week: the warehouse (memory, Part 1), the trucks that share a load (data, tensor and pipeline parallelism, Parts 2 and 3), the cheaper boxes (mixed precision and quantization, Parts 4 and 5), the delivery desk (batching and the KV cache, Part 6) and the plan that puts them together (Part 7). The lesson of the week is that **a deployment is a set of trade-offs that you can measure**: memory against bits, speed against communication, throughput against waiting, cost against accuracy. Bring your design card and your findings table to **Week 19**, where you build the capstone, and to the **Capstone** itself: the Structural Inspection Assistant needs exactly this plan (what runs where, at what precision, with what response time and with which guardrails from Week 17) and a licensed engineer who still makes every pass or fail decision.
