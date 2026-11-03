# accelerator: GPU

# %% [markdown]
# # Week 18 — Scaling & Deploying AI: lab notebook
#
# Built for **Google Colab**. Before you start, choose **File → Save a copy in Drive** so your work is saved in your own copy.
#
# **Runtime:** GPU recommended: Runtime → Change runtime type → T4 GPU. Every part also runs on CPU. The two GPU-only cells print a message and skip themselves when no GPU is found.
# **Time:** about 3 hours (compute is under 15 minutes).
#
# **What you will do**
# - Work out the weight memory of a model at each precision by hand.
# - Time a matrix multiply and measure peak GPU memory in FP32, FP16 and BF16 (GPU only).
# - Use `torch.autocast` and see which numbers change precision.
# - Quantize a small network to INT8 on the CPU and compare size, speed and output.
# - Simulate data parallelism and tensor parallelism with plain tensors and check that the split result equals the single-device result.
# - Compute the idle share of a simple pipeline schedule.
#
# **How this notebook works:** run the cells from top to bottom with the ▶ button. **Your turn** cells are for you to fill in. A **Check** cell tells you if you are right. Open the **Show a possible answer** block only after you tried.

# %%
import os, io, time, random
import numpy as np
import torch
import torch.nn as nn

SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "|", torch.cuda.get_device_name(0) if device == "cuda" else "no GPU found: GPU-only cells will be skipped")
QUICK = bool(os.environ.get("NB_TEST"))

def check(name, ok, hint=""):
    print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))

assert torch.tensor([1.0, 2.0]).sum().item() == 3.0   # sanity check that PyTorch works

# %% [markdown]
# ## Step 1 — Memory by hand
#
# Weight memory is the number of parameters times the bytes per parameter (see "Memory and precision" in the lesson). FP32 uses 4 bytes, FP16 and BF16 use 2, INT8 uses 1 and INT4 uses 0.5. The lesson's rough estimate for training with Adam is 16 bytes per parameter, before activations.

# %%
def weight_gb(params, bytes_per_param):
    return params * bytes_per_param / 1e9

params = 7e9
for name, b in [("FP32", 4), ("FP16", 2), ("INT8", 1), ("INT4", 0.5)]:
    print(f"{name}: {weight_gb(params, b):.1f} GB")
print("Adam training estimate:", weight_gb(params, 16), "GB")
assert weight_gb(7e9, 2) == 14.0 and weight_gb(7e9, 16) == 112.0

# %% [markdown]
# ### ✏️ Your turn: a 70 B model
#
# Compute the weight memory of a 70 billion parameter model in FP16, in GB, and put it in `answer`. Then think about whether it fits on one 80 GB GPU.

# %%
answer = None  # YOUR CODE HERE: weight memory in GB of a 70B model in FP16

# %%
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("70B in FP16", abs(answer - weight_gb(70e9, 2)) < 1e-6, "parameters times 2 bytes, divided by 1e9")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer = weight_gb(70e9, 2)   # 140.0
# ```
#
# 140 GB is more than a single 80 GB GPU holds, so the weights alone must be split across at least two GPUs, or quantized (INT8 would be 70 GB).
# </details>

# %% [markdown]
# ## Step 2 — Matrix multiply: FP32 versus FP16 and BF16 (GPU)
#
# First the expected numbers. Three n × n matrices at n = 4096 hold 3 × 16,777,216 numbers. At 4 bytes each that is about 201 MB, and at 2 bytes about 101 MB. The next cell runs only on a GPU and measures the real peak with `torch.cuda.max_memory_allocated`.

# %%
n = 4096
numbers = 3 * n * n
print("numbers:", numbers)
print(f"expected FP32: {numbers * 4 / 1e6:.1f} MB | expected 16-bit: {numbers * 2 / 1e6:.1f} MB")
assert numbers == 3 * 16_777_216

# %%
# nb-test: skip
# GPU only: this cell times matmuls and records peak memory.
if device != "cuda":
    print("Skipped: no GPU in this runtime. Change the runtime type to T4 GPU to run this cell.")
else:
    def bench(dtype, n=4096, reps=20):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        a = torch.randn(n, n, device="cuda", dtype=dtype)
        b = torch.randn(n, n, device="cuda", dtype=dtype)
        c = torch.empty(n, n, device="cuda", dtype=dtype)
        for _ in range(3): torch.matmul(a, b, out=c)          # warm-up
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(reps): torch.matmul(a, b, out=c)
        torch.cuda.synchronize(); sec = (time.perf_counter() - t0) / reps   # synchronize: the GPU is asynchronous
        mem = torch.cuda.max_memory_allocated() / 1e6
        del a, b, c; torch.cuda.empty_cache()
        return sec * 1000, 2 * n**3 / sec / 1e12, mem
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        try:
            ms, tf, mem = bench(dt)
            print(f"{str(dt):16s} {ms:7.2f} ms  {tf:6.1f} TFLOP/s  peak {mem:7.1f} MB")
        except Exception as e:
            print(f"{str(dt):16s} not available on this GPU: {e}")

# %% [markdown]
# ### ✏️ Your turn: expected memory
#
# Put in `answer` the expected MB for the three matrices at 2 bytes per number (as a number). If you ran the GPU cell, compare it with the measured peak for FP16 and think about why they can differ a little (a library workspace is possible).

# %%
answer = None  # YOUR CODE HERE: expected MB at 2 bytes per number

# %%
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("expected MB at 2 bytes", abs(answer - numbers * 2 / 1e6) < 0.1, "3 * 4096 * 4096 numbers times 2 bytes, divided by 1e6")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer = 3 * 4096 * 4096 * 2 / 1e6   # about 100.7
# ```
#
# On a GPU, FP16 and BF16 multiplies are usually clearly faster than FP32 (the exact times depend on the GPU), and the peak memory is about half. Measured peaks can be slightly higher than the formula because of workspace memory.
# </details>

# %% [markdown]
# ## Step 3 — Mixed precision with `torch.autocast`
#
# Autocast runs selected operations in a lower precision while the weights stay in FP32 (see "Mixed precision"). Look at which dtypes appear. This part runs on both CPU and GPU: on CPU, autocast uses BF16.

# %%
dim, batch = (2048, 256) if device == "cuda" else (512, 64)
model = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)).to(device)
x = torch.randn(batch, dim, device=device)
print("weights:", model[0].weight.dtype)
with torch.autocast(device_type=device, dtype=torch.bfloat16):
    y = model(x)
    loss = y.float().pow(2).mean()     # compute the loss in FP32
print("output:", y.dtype, "| loss:", loss.dtype)
loss.backward()
print("grad dtype:", model[0].weight.grad.dtype)
assert model[0].weight.dtype == torch.float32 and y.dtype == torch.bfloat16
assert model[0].weight.grad.dtype == torch.float32

# %% [markdown]
# ### ✏️ Your turn: which dtype?
#
# Put in `out_dtype` the dtype of the layer output under autocast (`y.dtype`) and in `grad_dtype` the dtype of the weight gradient. Then explain to yourself why the weights and gradients stay FP32 while the output is BF16.

# %%
out_dtype = None    # YOUR CODE HERE: e.g. torch.float32
grad_dtype = None   # YOUR CODE HERE

# %%
if out_dtype is None or grad_dtype is None:
    print("Fill in your answers above, then run this cell again.")
else:
    check("output dtype", out_dtype == y.dtype, "print y.dtype inside the autocast result")
    check("gradient dtype", grad_dtype == model[0].weight.grad.dtype, "gradients match the dtype of the parameters")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# out_dtype, grad_dtype = y.dtype, model[0].weight.grad.dtype   # torch.bfloat16, torch.float32
# ```
#
# Autocast casts the inputs of the matrix multiply to BF16 for speed, but the parameters are stored in FP32, so their gradients are FP32 too. The FP32 weights are where small updates are accumulated without being lost.
# </details>

# %%
# nb-test: skip
# GPU only: peak memory with and without autocast.
if device != "cuda":
    print("Skipped: no GPU in this runtime, so peak GPU memory cannot be measured.")
else:
    def peak(use_autocast):
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
            out = model(x)
        out.float().pow(2).mean().backward()
        return torch.cuda.max_memory_allocated() / 1e6
    print(f"peak MB fp32: {peak(False):.1f} | autocast: {peak(True):.1f}")
    print("With a small batch the weights dominate; autocast saves more as the batch (activations) grows.")

# %% [markdown]
# ## Step 4 — Dynamic INT8 quantization on the CPU
#
# Quantization stores weights in fewer bits. `quantize_dynamic` converts the weights of `nn.Linear` layers to INT8 after training, for **CPU inference** (see "Quantization"). Always call `.eval()` first and keep the model on the CPU.

# %%
from torch.ao.quantization import quantize_dynamic
torch.manual_seed(0)
fp = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 1024),
                   nn.ReLU(), nn.Linear(1024, 10)).eval()
q = quantize_dynamic(fp, {nn.Linear}, dtype=torch.qint8)

def size_mb(m):
    buf = io.BytesIO(); torch.save(m.state_dict(), buf); return buf.getbuffer().nbytes / 1e6

xq = torch.randn(64, 1024)
def ms(m, reps=10 if QUICK else 50):
    with torch.no_grad():
        m(xq); t0 = time.perf_counter()                      # one warm-up call first
        for _ in range(reps): m(xq)
    return (time.perf_counter() - t0) / reps * 1000

n_params = sum(p.numel() for p in fp.parameters())
s_fp, s_q = size_mb(fp), size_mb(q)
t_fp, t_q = ms(fp), ms(q)
with torch.no_grad():
    diff = (fp(xq) - q(xq)).abs().max().item()
print("parameters:", n_params)
print(f"size MB: {s_fp:.2f} -> {s_q:.2f}  (ratio {s_fp / s_q:.2f}x)")
print(f"time ms: {t_fp:.2f} -> {t_q:.2f}  (speed ratio {t_fp / t_q:.2f}x)")
print(f"max |difference| of the outputs: {diff:.4f}  (output values are around {fp(xq).abs().mean().item():.3f})")
assert n_params == 2_109_450
assert s_q < s_fp / 3

# %% [markdown]
# ### ✏️ Your turn: size by hand
#
# The network has 2,109,450 parameters. Put in `answer` its FP32 size in MB (4 bytes each, MB = 1e6 bytes) and compare it with the printed size above. Then compare the size ratio with the speed ratio: is the speed-up as large as the size saving? (Speed depends on your CPU and may vary between runs.)

# %%
answer = None  # YOUR CODE HERE: FP32 size in MB from the parameter count

# %%
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("FP32 size in MB", abs(answer - 2_109_450 * 4 / 1e6) < 0.01, "parameters times 4 bytes, divided by 1e6")
    print(f"printed file size was {s_fp:.2f} MB (a little larger than the weights alone because of file overhead)")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer = 2_109_450 * 4 / 1e6   # 8.4378
# ```
#
# The saved FP32 model is about 8.44 MB and the INT8 one is roughly a quarter of that. The speed-up is usually smaller than the 4x size saving, and can even be absent on some CPUs: the layers are small, the weights are converted back and forth, and speed depends on the CPU and on other work (for example the ReLU and the activations stay in floating point). Memory saved and time saved are different things.
# </details>

# %% [markdown]
# ## Step 5 — Check data and tensor parallelism on one machine
#
# We simulate two "GPUs" with two slices of a tensor. **Data parallelism**: each worker has a full copy of the weights and half the batch; the gradients are averaged. **Tensor parallelism**: one weight matrix is cut into column halves, each half computes part of the output, and the parts are joined.

# %%
torch.manual_seed(0)
w = torch.randn(5, requires_grad=True)
X, y_t = torch.randn(8, 5), torch.randn(8)

def grad(Xb, yb):
    w.grad = None
    ((Xb @ w - yb) ** 2).mean().backward()      # mean loss
    return w.grad.clone()

full = grad(X, y_t)
avg = (grad(X[:4], y_t[:4]) + grad(X[4:], y_t[4:])) / 2     # two equal halves, then average
dp_ok = torch.allclose(full, avg, atol=1e-6)
print("data parallel matches:", dp_ok)

W, inp = torch.randn(6, 8), torch.randn(3, 6)
left, right = inp @ W[:, :4], inp @ W[:, 4:]                # each "GPU" holds half of the columns
tp_ok = torch.allclose(inp @ W, torch.cat([left, right], dim=1), atol=1e-6)
print("tensor parallel matches:", tp_ok)
assert dp_ok and tp_ok

# %% [markdown]
# ### ✏️ Your turn: when does the average break?
#
# Split the same batch unevenly: 2 examples on one worker and 6 on the other. Average the two gradients with equal weights (a plain `/ 2`) and put in `uneven_matches` whether the result still equals the full-batch gradient (`True` or `False`). Why does the equal-size split matter when the loss is a mean?

# %%
uneven_matches = None  # YOUR CODE HERE: True or False

# %%
uneven = (grad(X[:2], y_t[:2]) + grad(X[2:], y_t[2:])) / 2
truth = torch.allclose(full, uneven, atol=1e-6)
if uneven_matches is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("uneven split", uneven_matches == truth, "compute the uneven average and compare it with `full`")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# uneven = (grad(X[:2], y_t[:2]) + grad(X[2:], y_t[2:])) / 2
# uneven_matches = torch.allclose(full, uneven, atol=1e-6)   # False
# ```
#
# The full-batch loss is the mean over 8 examples. A mean over a slice is its sum divided by the slice size, so the average of two slice means equals the full mean only when both slices have the same size. With 2 and 6 examples you would need weights 2/8 and 6/8 instead of 1/2 and 1/2.
# </details>

# %% [markdown]
# ## Step 6 — Pipeline parallelism: the idle share
#
# In pipeline parallelism, the layers are divided into p stages and a batch is split into m micro-batches. In the simple GPipe-style schedule (all forward passes, no overlap with backward), a stage works on m of the p + m - 1 time slots in total, so its **idle share** is (p - 1) / (m + p - 1). The cell below checks the formula by counting the idle slots on a grid.

# %%
def idle_share(p, m):
    return (p - 1) / (m + p - 1)

def idle_by_counting(p, m):
    slots = p + m - 1
    busy = np.zeros((p, slots), dtype=bool)
    for stage in range(p):
        for micro in range(m):
            busy[stage, stage + micro] = True        # micro-batch i reaches stage s at time s + i
    return 1 - busy.mean()

for p, m in [(4, 4), (4, 8), (4, 16)]:
    print(f"p={p}, m={m}: formula {idle_share(p, m):.3f} | counted {idle_by_counting(p, m):.3f}")
    assert abs(idle_share(p, m) - idle_by_counting(p, m)) < 1e-12
assert abs(idle_share(4, 4) - 3 / 7) < 1e-12

# %% [markdown]
# ### ✏️ Your turn: p = 4, m = 8
#
# Use the formula to compute the idle share for p = 4 stages and m = 8 micro-batches, and put it in `answer` (a fraction between 0 and 1).

# %%
answer = None  # YOUR CODE HERE: idle share for p=4, m=8

# %%
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("idle share for p=4, m=8", abs(answer - 3 / 11) < 1e-6, "(p - 1) / (m + p - 1)")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer = idle_share(4, 8)   # 3/11, about 0.273
# ```
#
# (4 - 1) / (8 + 4 - 1) = 3/11, about 27%. More micro-batches make the idle share smaller, at the price of smaller pieces of work per step.
# </details>

# %% [markdown]
# ## Step 7 — Put it together: a deployment decision
#
# A hospital must serve a 13 B assistant to 50 users on two GPUs with 24 GB each. Use the numbers to choose.

# %%
for name, b in [("FP16", 2), ("INT8", 1), ("INT4", 0.5)]:
    gb = weight_gb(13e9, b)
    print(f"{name}: weights {gb:.1f} GB | fits in one 24 GB GPU: {gb < 24} | per GPU if split in two: {gb / 2:.1f} GB")
assert weight_gb(13e9, 2) == 26.0

# %% [markdown]
# The weights are not the whole story: the KV cache and activations also need memory, and they grow with the number of users and the sequence length. Write your decision below.
#
# ### ✏️ Your turn: five sentences
#
# Choose a precision and a parallelism strategy and justify them with the numbers above.

# %%
decision = ""   # YOUR CODE HERE: five sentences as a string

# %%
if not decision.strip():
    print("Write your answer above, then run this cell again.")
else:
    check("you wrote a reasoned answer", len(decision.split()) >= 40, "aim for about five sentences that use the numbers")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# FP16 weights are 26 GB, which does not fit on a single 24 GB GPU, so FP16 needs the model split over both GPUs (13 GB of weights each), for example with tensor parallelism. INT8 weights are 13 GB and fit on one GPU, leaving about 11 GB for the KV cache and activations, and the second GPU can hold a second copy to serve more users (data parallelism for the requests). I would choose INT8 with one copy per GPU if the accuracy loss is acceptable after testing, and FP16 split over both GPUs if it is not. The final choice must be checked by measuring accuracy and latency, not only by arithmetic.
# </details>

# %%
# Free GPU memory when you are done
del model, x
if device == "cuda":
    torch.cuda.empty_cache()
print("memory released")

# %% [markdown]
# ## Wrap-up
#
# - Weight memory is parameters times bytes per parameter, so lower precision (FP16, INT8, INT4) shrinks it in a predictable way; training needs much more (about 16 bytes per parameter with Adam).
# - Autocast keeps FP32 weights and gradients while running operations in BF16; dynamic INT8 quantization cut the model file to about a quarter, but speed gains are smaller and vary.
# - Data and tensor parallelism give the same result as one device when the split is done correctly, and a simple pipeline wastes (p - 1) / (m + p - 1) of the time.
#
# **Reflection.** Double-click to write two sentences: which of the three ideas (lower precision, quantization, parallelism) would help most for the model you want to deploy, and why?
#
# Files in `/content` vanish when the Colab session ends, so download anything you want to keep (File → Download .ipynb). Submit your notebook the way your instructor asks.
