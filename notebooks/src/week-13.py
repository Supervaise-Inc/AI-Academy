# accelerator: GPU

# %% [markdown]
# # Week 13 — Diffusion Models: lab notebook
#
# Built for **Google Colab**. Before you start, choose **File → Save a copy in Drive** so your work is saved in your own copy.
#
# **Runtime:** Parts 1 to 4 run on a CPU in a couple of minutes. Part 5 (Stable Diffusion) needs a GPU: **Runtime → Change runtime type → T4 GPU**. Without a GPU the notebook still runs top to bottom and simply skips Part 5. **Time:** about 3 hours with the exercises and the reflection.
#
# ### What you will do
# - Build the noise schedule and check the numbers from the lesson (ᾱ at t = 300 and t = 1000).
# - Noise a picture at several steps and see when it disappears.
# - See why recovering the image from a noise guess gets harder at large t.
# - Train a tiny network that predicts the noise, in about a minute on a CPU.
# - Use Stable Diffusion and compare 5, 15 and 30 steps and different guidance scales (GPU only).
#
# **How this notebook works:** run the cells from top to bottom with ▶. **Your turn** cells are for you to fill in. A **Check** cell tells you if you are right. Open the **Show a possible answer** block only after you have tried.

# %% [markdown]
# ## Setup
#
# One install (`diffusers`, the library that runs Stable Diffusion, and `accelerate`). Colab already has PyTorch, NumPy, scikit-learn and matplotlib. The first cell may take about a minute.

# %%
#!pip install -q diffusers accelerate

import os, time, gc
import numpy as np
import matplotlib.pyplot as plt
import torch

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
QUICK = bool(os.environ.get("NB_TEST"))   # shortens long loops when the notebook is tested

def check(name, ok, hint=""):
    print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))

# %% [markdown]
# ## Step 1: the noise schedule
#
# From the lesson section **The forward process: adding noise**: the noise schedule β grows in a straight line from 0.0001 to 0.02 over T = 1000 steps, and ᾱ_t ("alpha-bar") is the running product of (1 − β). The image is multiplied by √ᾱ_t and the noise by √(1 − ᾱ_t). Below we build both and check the table from the lesson.

# %%
T = 1000
betas = np.linspace(1e-4, 0.02, T)        # beta_1 ... beta_T
alpha_bar = np.cumprod(1.0 - betas)       # alpha_bar[t-1] is the value at step t

print("alpha_bar(300)  =", alpha_bar[299])
print("alpha_bar(1000) =", alpha_bar[999])
print(f"{'t':>5} {'alpha_bar':>10} {'signal':>8} {'noise':>8}")
for t in (1, 100, 300, 500, 1000):
    a = alpha_bar[t - 1]
    print(f"{t:>5} {a:>10.4f} {np.sqrt(a):>8.4f} {np.sqrt(1 - a):>8.4f}")

# Sanity checks against the numbers in the lesson
assert abs(alpha_bar[299] - 0.3964) < 5e-4, "alpha_bar(300) should be about 0.3964"
assert 3e-5 < alpha_bar[999] < 5e-5, "alpha_bar(1000) should be about 4.0e-5"

# %%
steps = np.arange(1, T + 1)
plt.figure(figsize=(5, 3))
plt.plot(steps, np.sqrt(alpha_bar), label="signal  √ᾱ")
plt.plot(steps, np.sqrt(1 - alpha_bar), label="noise  √(1−ᾱ)")
plt.xlabel("step t"); plt.legend(); plt.tight_layout(); plt.show()

# %% [markdown]
# ### ✏️ Your turn: where do signal and noise cross?
#
# The two curves meet where ᾱ = 0.5 (then √ᾱ = √(1 − ᾱ)). Find the first step t at which ᾱ_t falls **below** 0.5. Remember that `alpha_bar[299]` is step 300, because arrays start at 0.

# %%
answer = None  # YOUR CODE HERE: the step number t (an integer between 1 and 1000)

# %% [markdown]
# ### Check

# %%
true_t = int(np.argmax(alpha_bar < 0.5) + 1)
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("first step with alpha_bar < 0.5", answer == true_t, "use np.argmax(alpha_bar < 0.5) and add 1")
    if answer == true_t:
        print(f"Yes: t = {true_t}, between 200 and 300, so at t = 300 noise already outweighs the picture.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer = int(np.argmax(alpha_bar < 0.5) + 1)
# ```
# `alpha_bar < 0.5` is a True/False array, `np.argmax` returns the index of the first True, and the `+ 1` converts the array index to the step number.
# </details>

# %% [markdown]
# ## Step 2: noise a picture
#
# Now we use the shortcut from the lesson, **x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε**, on a real 8×8 handwritten digit from scikit-learn (no download). Pixel values are scaled to the range −1 to 1, which is the usual choice for diffusion models. The same noise ε is reused at every t, so you can see one picture dissolve.

# %%
from sklearn.datasets import load_digits
digits = load_digits()
x0 = digits.images[0] / 16 * 2 - 1        # 8x8 array, values in [-1, 1]
rng = np.random.default_rng(SEED)
eps = rng.standard_normal(x0.shape)

ts = [1, 50, 100, 300, 500, 1000]
fig, ax = plt.subplots(1, len(ts) + 1, figsize=(14, 2.5))
ax[0].imshow(x0, cmap="gray_r", vmin=-3, vmax=3); ax[0].set_title("x0")
for a, t in zip(ax[1:], ts):
    ab = alpha_bar[t - 1]
    xt = np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps
    a.imshow(xt, cmap="gray_r", vmin=-3, vmax=3); a.set_title(f"t={t}")
for a in ax:
    a.axis("off")
plt.show()

# %% [markdown]
# ### ✏️ Your turn: how much of the picture is left?
#
# Compute the two numbers that multiply x₀ and ε at **t = 500**. Store them as `answer = (signal_weight, noise_weight)`. Compare with the picture above: at which of the columns can you no longer recognise the digit?

# %%
answer = None  # YOUR CODE HERE: a tuple (signal_weight, noise_weight)

# %% [markdown]
# ### Check

# %%
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    ab = alpha_bar[499]
    ok = abs(answer[0] - np.sqrt(ab)) < 0.005 and abs(answer[1] - np.sqrt(1 - ab)) < 0.005
    check("signal and noise weights at t = 500", ok, "signal = sqrt(alpha_bar[499]), noise = sqrt(1 - alpha_bar[499])")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# ab = alpha_bar[499]            # step 500
# answer = (np.sqrt(ab), np.sqrt(1 - ab))   # about (0.2803, 0.9599)
# ```
# At t = 500 the picture is multiplied by about 0.28 and the noise by about 0.96, so noise has almost taken over. Your own answer to "when is the digit gone" depends on the seed: try `digits.images[3]` and another seed.
# </details>

# %% [markdown]
# ## Step 3: recover the image from the noise
#
# If a network predicted ε perfectly, we could rewrite the forward formula to get x₀ back: **x₀ = (x_t − √(1−ᾱ_t) · ε̂) / √ᾱ_t**. Here we give it a perfect guess and then a slightly wrong one (an error of standard deviation 0.1) and see what a small mistake costs.

# %%
def predict_x0(xt, eps_hat, t):
    ab = alpha_bar[t - 1]
    return (xt - np.sqrt(1 - ab) * eps_hat) / np.sqrt(ab)

for t in (50, 300, 1000):
    ab = alpha_bar[t - 1]
    xt = np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps
    exact = np.abs(predict_x0(xt, eps, t) - x0).max()
    wrong = eps + 0.1 * rng.standard_normal(eps.shape)    # a slightly wrong noise estimate
    err = np.abs(predict_x0(xt, wrong, t) - x0).mean()
    factor = np.sqrt(1 - ab) / np.sqrt(ab)
    print(f"t={t:>4} | exact error: {exact:.1e} | error with a 0.1 mistake: {err:.3f} | amplification: {factor:.3f}")

assert np.abs(predict_x0(np.sqrt(alpha_bar[299]) * x0 + np.sqrt(1 - alpha_bar[299]) * eps, eps, 300) - x0).max() < 1e-9

# %% [markdown]
# ### ✏️ Your turn: the amplification factor
#
# The factor √(1−ᾱ)/√ᾱ says how much a mistake in ε is multiplied when we rebuild x₀. Compute it at **t = 50** and at **t = 1000**, and store `answer = (factor_50, factor_1000)`. Which one is easier to predict accurately, and why?

# %%
answer = None  # YOUR CODE HERE: a tuple (factor at t=50, factor at t=1000)

# %% [markdown]
# ### Check

# %%
def amp(t):
    ab = alpha_bar[t - 1]
    return np.sqrt(1 - ab) / np.sqrt(ab)

if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    ok = abs(answer[0] - amp(50)) < 0.01 and abs(answer[1] - amp(1000)) < 0.5
    check("amplification at t = 50 and t = 1000", ok, "factor = sqrt(1 - ab) / sqrt(ab) with ab = alpha_bar[t-1]")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer = (amp(50), amp(1000))   # about (0.173, 157.4)
# ```
# At t = 1000 almost nothing of x₀ is left in x_t, so the tiny signal must be divided by √ᾱ ≈ 0.0064: any small error in ε becomes about 157 times bigger in the recovered image. At t = 50 the factor is about 0.17, so the same mistake barely matters. This is why the sampler recovers the image gradually, in many small steps, instead of in one jump.
# </details>

# %% [markdown]
# ## Step 4: a tiny noise-prediction network
#
# From the lesson section **Training: predict the noise**: the network sees x_t and the step t and must output ε. The loss is the mean squared error between the real ε and the guess. Real diffusion models use a UNet with a time embedding; here we use a small multilayer network on the 8×8 digits so that it trains in about a minute on a CPU. It receives the 64 pixels plus a simple time feature (sine and cosine of t, like a positional encoding).

# %%
import torch.nn as nn

X = torch.tensor(digits.images / 16 * 2 - 1, dtype=torch.float32).reshape(-1, 64)   # 1797 images, 64 pixels
ab_t = torch.tensor(alpha_bar, dtype=torch.float32)

def time_feats(t):                       # t: LongTensor of steps 1..T -> (batch, 8) features
    f = t.float().unsqueeze(1) / T * torch.arange(1, 5).float() * np.pi
    return torch.cat([torch.sin(f), torch.cos(f)], dim=1)

class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(64 + 8, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 64))
    def forward(self, xt, t):
        return self.net(torch.cat([xt, time_feats(t)], dim=1))

def make_batch(x, t):
    e = torch.randn_like(x)
    ab = ab_t[t - 1].unsqueeze(1)
    return torch.sqrt(ab) * x + torch.sqrt(1 - ab) * e, e

model = Denoiser()
print("parameters:", sum(p.numel() for p in model.parameters()))

# %%
torch.manual_seed(SEED)
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
n_steps = 300 if QUICK else 1500
losses = []
t0 = time.time()
for step in range(n_steps):
    idx = torch.randint(0, len(X), (128,))
    t = torch.randint(1, T + 1, (128,))
    xt, e = make_batch(X[idx], t)
    loss = ((model(xt, t) - e) ** 2).mean()         # MSE between the real noise and the guess
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())
print(f"{n_steps} steps in {time.time() - t0:.0f}s. loss: start {np.mean(losses[:20]):.3f} -> end {np.mean(losses[-20:]):.3f}")

# Sanity check: guessing "no noise" everywhere gives a loss near 1 (the variance of the noise)
assert np.mean(losses[-20:]) < np.mean(losses[:20]) * 0.8, "the loss should go down"
plt.figure(figsize=(5, 2.5)); plt.plot(losses); plt.xlabel("training step"); plt.ylabel("MSE"); plt.tight_layout(); plt.show()

# %% [markdown]
# Now look at what the network learned. We take a digit it has not just seen noised in this exact way, add noise at t = 300, let the network guess ε, and rebuild x₀ with the formula from Step 3.

# %%
torch.manual_seed(1)
x_clean = X[5:6]
t_demo = torch.tensor([300])
xt, e = make_batch(x_clean, t_demo)
with torch.no_grad():
    e_hat = model(xt, t_demo)
ab = alpha_bar[299]
x0_hat = (xt - np.sqrt(1 - ab) * e_hat) / np.sqrt(ab)
fig, ax = plt.subplots(1, 3, figsize=(7, 2.6))
for a, img, title in zip(ax, (x_clean, xt, x0_hat), ("clean x0", "noisy x_300", "network's estimate of x0")):
    a.imshow(img.reshape(8, 8).numpy(), cmap="gray_r", vmin=-3, vmax=3); a.set_title(title, fontsize=9); a.axis("off")
plt.show()
print("Mean squared error of the noise guess at t=300:", ((e_hat - e) ** 2).mean().item())

# %% [markdown]
# ### ✏️ Your turn: easy or hard noise levels?
#
# Measure the network's noise-prediction error (MSE, where 1.0 means "no better than guessing zero") at **t = 20** and at **t = 900**, using many random noise samples. Then set `answer` to the step, `20` or `900`, where the error is **lower**. Use `make_batch(X, t)` with `t = torch.full((len(X),), 20)`.

# %%
answer = None  # YOUR CODE HERE: 20 or 900

# %% [markdown]
# ### Check

# %%
def mse_at(tt):
    torch.manual_seed(2)
    t = torch.full((len(X),), tt)
    xt, e = make_batch(X, t)
    with torch.no_grad():
        return ((model(xt, t) - e) ** 2).mean().item()

m20, m900 = mse_at(20), mse_at(900)
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    truth = 20 if m20 < m900 else 900
    check("noise level with the lower error", answer == truth, "compute both MSE values and compare them")
    print(f"MSE at t=20: {m20:.3f}   MSE at t=900: {m900:.3f}")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# t = torch.full((len(X),), 20);  xt, e = make_batch(X, t)
# with torch.no_grad(): m20 = ((model(xt, t) - e) ** 2).mean().item()
# # same for t = 900, then:  answer = 20 if m20 < m900 else 900
# ```
# In this tiny model the error is usually lower at t = 900 (the printout above shows the values you got). At t = 900 x_t is almost pure noise, so "ε ≈ x_t" is already a good guess. At t = 20 the noise is a light dusting on a picture, so the network has to separate noise from the digit's own pixels, which is harder for its noise guess. Compare with Step 3: a small error in ε matters most at large t, even though the noise itself is easiest to guess there. Both ideas are true at once.
# </details>

# %% [markdown]
# ## Step 5: Stable Diffusion (T4 GPU only)
#
# From the lesson section **Making it useful: prompts, guidance and latent space**. Stable Diffusion works in a compressed latent space: a 512×512 RGB image has 512 × 512 × 3 = 786,432 values and its latent is 64 × 64 × 4 = 16,384 values, which is 48 times fewer. This part needs a GPU (**Runtime → Change runtime type → T4 GPU**). The model download is about 4 GB the first time, which can take a few minutes. Without a GPU the next cells print a message and skip.

# %%
# nb-test: skip
# Needs a GPU and a ~4 GB download, so it is not run in the automatic test.
if device == "cuda":
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    prompt = "A watercolour painting of a lighthouse at dusk"
    print("Pipeline loaded in half precision (fp16).")
else:
    print("No GPU found, so Stable Diffusion is skipped. Use Runtime → Change runtime type → T4 GPU to run this part.")

# %% [markdown]
# **Number of steps.** We fix the random seed so that only the step count changes. Each step is one call of the UNet denoiser.

# %%
# nb-test: skip
if device == "cuda":
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, n in zip(ax, (5, 15, 30)):
        g = torch.Generator("cuda").manual_seed(0)
        t0 = time.time()
        img = pipe(prompt, num_inference_steps=n, guidance_scale=7.5, generator=g).images[0]
        a.imshow(img); a.set_title(f"{n} steps, {time.time() - t0:.1f}s"); a.axis("off")
    plt.show()
else:
    print("Skipped: no GPU.")

# %% [markdown]
# **Guidance scale.** Now fix the steps at 20 and the seed, and change only the guidance scale (the default in `diffusers` is 7.5). Higher values usually follow the prompt more closely and give less variety; very high values can look harsh or over-saturated.

# %%
# nb-test: skip
if device == "cuda":
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    for a, gs in zip(ax, (1.0, 4.0, 7.5, 15.0)):
        g = torch.Generator("cuda").manual_seed(0)
        img = pipe(prompt, num_inference_steps=20, guidance_scale=gs, generator=g).images[0]
        a.imshow(img); a.set_title(f"guidance {gs}"); a.axis("off")
    plt.show()
else:
    print("Skipped: no GPU.")

# %% [markdown]
# ### ✏️ Your turn: describe what you see
#
# Write what changed between 5, 15 and 30 steps (sharpness, detail, odd shapes) and how the time changed, and which step count you would use for a quick preview and which for a final image. Then change `prompt` and the seed in the cells above and say which of prompt, seed and scale changes the result most. Write your notes in the string below. (No GPU? Write what you would expect and mark it as a prediction.)

# %%
notes = ""  # YOUR NOTES HERE

# %% [markdown]
# ### Check

# %%
if not notes.strip():
    print("Write a few sentences in `notes` above, then run this cell again.")
else:
    check("you wrote at least 20 words", len(notes.split()) >= 20, "add more detail about sharpness, time and guidance")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# notes = "5 steps looked soft and unfinished; 15 and 30 were sharper, 30 had a little more fine detail. Time grew with the number of steps. I would use 15 for a preview and 30 for a final image. Guidance 1 followed the prompt loosely; 15 looked harsh and over-saturated."
# ```
# Your observations may differ: they depend on the prompt and the seed. The general tendencies are that more steps usually mean better quality and more time, and that very high guidance can look harsh.
# </details>

# %% [markdown]
# ### Free the GPU memory

# %%
# nb-test: skip
if device == "cuda":
    del pipe
    gc.collect(); torch.cuda.empty_cache()
    print("GPU memory released.")
else:
    print("Nothing to free.")

# %% [markdown]
# ## Wrap-up
#
# - The forward process is x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε. With the linear schedule, ᾱ is 0.3964 at t = 300 and about 4.0e-5 at t = 1000, so the end of the forward process is almost pure noise.
# - The network is trained to **predict the noise** with a mean squared error loss, at random noise levels. Sampling then runs it step by step, and more steps usually mean better quality and more time.
# - Stable Diffusion adds text conditioning with guidance and works in a latent space that has 48 times fewer values than the pixels.
#
# **Reflection.** Double-click here to write two sentences: (a) why training a diffusion model is usually more stable than training a GAN, and (b) the capstone needs images classified, not generated: could diffusion still help (for example to create extra practice photos), and what is one risk?
#
# **Capstone opens this week.** The Structural Inspection Assistant capstone (photo in, classify, find checklist items with RAG, draft a summary for an engineer) opens in Week 13. A licensed engineer makes every pass/fail decision. Read the Capstone page in the course app and discuss with your team which model could handle each step.
#
# **Before you leave:** files in `/content` vanish when the Colab session ends. Download anything you want to keep (File → Download .ipynb) and submit the notebook the way your instructor asks.
