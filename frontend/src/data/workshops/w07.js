// Week 7 · Attention mechanisms. Follows Laboratory 3, Part 1: scaled dot-product attention (notebooks/src/week-07.py).
export default {
  week: 7,
  title: 'Attention lab',
  lede: 'Compute scaled dot-product attention by hand, check it in PyTorch, then test the scaling and compare hard with soft attention.',
  lab: 'Laboratory 3, Part 1',
  chips: ['3 hours', 'CPU is enough', 'One query, three keys', 'PyTorch'],
  exercises: [
    {
      tab: '1 · soft vs hard',
      title: 'Why soft attention can be trained',
      min: 15,
      kind: 'prompt',
      goal: 'Explain in words why soft attention can be trained by gradient descent while hard attention cannot (Part A).',
      tasks: [
        'Write your own 3 to 5 sentence answer first.',
        'Ask the tutor the question, mentioning what happens to the output when one score changes by a tiny amount.',
        'Compare the tutor answer with yours and fix any gap.',
      ],
      presets: [
        { l: 'Vague question', u: 'Is attention better?' },
        {
          l: 'Soft vs hard',
          u: 'Why can soft attention be trained by gradient descent but hard attention (argmax) cannot? Think about what happens to the output when one score changes by a tiny amount.',
        },
      ],
      rules: [
        {
          u: 'soft.*hard|hard.*soft|argmax|gradient',
          out: 'Soft attention is a smooth weighted average. If a score changes a little, the weights and the output change a little, so there is a gradient to follow. Hard attention picks the maximum. The output jumps when a different key becomes the maximum and does not change at all otherwise, so the gradient is zero almost everywhere and the model gets no signal to learn from.',
          note: 'matches the lab answer',
          ok: true,
        },
      ],
      fallback: {
        out: 'Better than what, and for what? Tell me which two things you compare (for example soft and hard attention) and what you want to know, such as whether each can be trained with gradients.',
        note: 'vague prompt, vague answer',
        ok: false,
      },
    },
    {
      tab: '2 · by hand',
      title: 'Attention by hand, then in PyTorch',
      min: 40,
      kind: 'code',
      goal: 'Compute scores, weights and output for one query, three keys and three values (lesson Figure 3), first on paper and then in PyTorch.',
      tasks: [
        'On paper, compute the three dot products q . ki, divide each by the square root of 2, then the softmax, then the output.',
        'Run the cell and compare each printed row with your paper work.',
        'Say why keys 1 and 3 get the same weight.',
      ],
      code: `import torch
import torch.nn.functional as F
q = torch.tensor([[1., 0.]])                        # 1 query
K = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])    # 3 keys
V = torch.tensor([[1., 0.], [0., 1.], [2., 2.]])    # 3 values
scores = q @ K.T / (2 ** 0.5)       # step 1: similarity of the query to each key, scaled
w = F.softmax(scores, dim=-1)       # step 2: turn scores into weights that sum to 1
out = w @ V                         # step 3: weighted average of the values
print("scores :", scores.numpy().round(4))
print("weights:", w.numpy().round(4))
print("output :", out.numpy().round(4))`,
      run: 'Run cell',
      out: 'scores : [[0.7071 0.     0.7071]]\nweights: [[0.4011 0.1978 0.4011]]\noutput : [[1.2033 1.    ]]',
      expect: 'Scores 0.7071, 0, 0.7071. Keys 1 and 3 have the same score, so each gets 0.4011; key 2 is less similar, so it gets 0.1978. The output is [1.2033, 1.0].',
    },
    {
      tab: '3 · no scaling',
      title: 'Remove the scaling',
      min: 20,
      kind: 'code',
      goal: 'Wrap the three steps in a function and call it with scale=False on the same q, K and V.',
      tasks: [
        'Run the cell and read both weight rows.',
        'Do the weights move closer together or further apart without the scaling?',
        'Say why in one sentence, using the unscaled scores 1, 0, 1.',
      ],
      code: `def scaled_dot_product_attention(query, key, value, scale=True):
    """Return (output, weights). Set scale=False to skip the 1/sqrt(d_k) factor."""
    d_k = query.shape[-1]
    scores = torch.matmul(query, key.transpose(-2, -1))
    if scale:
        scores = scores / (d_k ** 0.5)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, value), weights

print(scaled_dot_product_attention(q, K, V)[1][0].tolist())
print(scaled_dot_product_attention(q, K, V, scale=False)[1][0].tolist())`,
      run: 'Run cell',
      out: 'scaled  : [0.4011, 0.1978, 0.4011]\nunscaled: [0.4223, 0.1554, 0.4223]\n(the lists print with more decimals; rounded to 4 places they read as above)',
      expect: 'Without scaling the scores are 1, 0, 1 instead of 0.7071, 0, 0.7071, so the weights move further apart: the big scores get more of the weight (0.4223 instead of 0.4011).',
    },
    {
      tab: '4 · why sqrt(d_k)',
      title: 'Why divide by the square root of d_k?',
      min: 30,
      kind: 'code',
      goal: 'Measure the mean largest softmax weight for growing key sizes, with and without scaling.',
      tasks: [
        'Run the cell. For each d_k it uses 2000 random queries and 3 random keys.',
        'Describe how the unscaled column changes as d_k grows, and how the scaled column behaves.',
        'Report both values at d_k = 64. With 3 keys, what would an even split and a one-hot split give?',
      ],
      code: `torch.manual_seed(0)
dims = [2, 8, 32, 64, 256, 1024]
for d in dims:
    Q  = torch.randn(2000, 1, d)    # 2000 queries
    Kr = torch.randn(2000, 3, d)    # 3 keys each
    _, wu = scaled_dot_product_attention(Q, Kr, Kr, scale=False)
    _, ws = scaled_dot_product_attention(Q, Kr, Kr, scale=True)
    print(f"d_k={d:5d}   without scaling: {wu.max(dim=-1).values.mean():.3f}   with scaling: {ws.max(dim=-1).values.mean():.3f}")`,
      run: 'Run cell',
      out: 'Exact values vary with the random draw. What the lab says to expect:\n- Without scaling, the largest weight climbs from about 0.6 towards 1 as d_k grows. At d_k = 64 it is about 0.93; at d_k = 1024 it is above 0.9.\n- With scaling, it stays roughly between 0.55 and 0.6 for every d_k (about 0.60 at d_k = 64).',
      expect: 'An even split of 3 keys gives 0.333 and a one-hot split gives 1.0. Unscaled attention saturates (almost hard attention, tiny gradients); scaled attention keeps blending the keys.',
    },
    {
      tab: '5 · hard vs soft',
      title: 'Hard (argmax) versus soft (softmax) on the tie',
      min: 30,
      kind: 'code',
      goal: 'See what hard attention does with a tie, and how each version reacts to a tiny nudge of key 3.',
      tasks: [
        'Before running, say which value vector hard_attention returns for the original example.',
        'Run the cell and compare hard and soft outputs.',
        'Explain why a 0.001 nudge changes the hard output a lot and the soft output hardly at all.',
      ],
      code: `def hard_attention(query, key, value):
    scores = query @ key.T / (key.shape[-1] ** 0.5)
    idx = scores.argmax(dim=-1)          # pick one key only
    return value[idx], idx

h_out, h_idx = hard_attention(q, K, V)
print("hard: picks key", int(h_idx) + 1, "-> output", h_out.numpy().round(4))
print("soft: weights", w.numpy().round(4), "-> output", out.numpy().round(4))

K_nudged = K.clone()
K_nudged[2, 0] = 1.001               # make key 3 a tiny bit more similar
h2, i2 = hard_attention(q, K_nudged, V)
s2, _ = scaled_dot_product_attention(q, K_nudged, V)
print("after a 0.001 nudge -> hard picks key", int(i2) + 1, "output", h2.numpy().round(4))
print("after a 0.001 nudge -> soft output", s2.numpy().round(4))`,
      run: 'Run cell',
      out: 'hard: picks key 1 -> output [[1. 0.]]\nsoft: weights [[0.4011 0.1978 0.4011]] -> output [[1.2033 1.    ]]\nafter a 0.001 nudge -> hard picks key 3 output [[2. 2.]]\nafter a 0.001 nudge -> soft output (barely different from [[1.2033 1.]], by less than 0.01)',
      expect: 'With a tie, argmax quietly picks key 1 (the first). A 0.001 nudge makes it jump to key 3 and the output jumps from [1, 0] to [2, 2]. The soft output moves only a tiny amount.',
    },
    {
      tab: '6 · check scores',
      title: 'Check your own attention numbers',
      min: 20,
      kind: 'prompt',
      goal: 'Use the tutor to check a hand calculation and to debug the missing-scaling mistake.',
      tasks: [
        'Give the tutor your own query, keys and values and ask it for the scores. Or ask about the lab example.',
        'Ask what changes in the weights when the 1/sqrt(d_k) factor is left out.',
        'Compare with your paper work.',
      ],
      presets: [
        { l: 'Vague question', u: 'What are the attention numbers?' },
        {
          l: 'Lab example',
          u: 'Query (1, 0), keys (1, 0), (0, 1), (1, 1), values (1, 0), (0, 1), (2, 2), d_k = 2. What are the scores, the weights and the output?',
        },
        {
          l: 'No scaling',
          u: 'What happens to the weights for the same example if I forget to divide by sqrt(d_k)?',
        },
      ],
      rules: [
        {
          u: 'forget|without|no scal|skip|leave out|left out|scale\\s*=\\s*false|divide by sqrt',
          out: 'Without dividing by sqrt(d_k) the scores are the raw dot products 1, 0, 1 instead of 0.7071, 0, 0.7071. The softmax then gives weights [0.4223, 0.1554, 0.4223] instead of [0.4011, 0.1978, 0.4011]. The weights move further apart: the big scores get more weight. With a larger d_k the effect grows and the softmax becomes almost one-hot, which gives tiny gradients.',
          note: 'matches the lab: 0.4223, 0.1554, 0.4223',
          ok: true,
        },
        {
          u: '\\(1,\\s*0\\).*\\(0,\\s*1\\).*\\(1,\\s*1\\)|scores?.*weights?|lab example|d_k\\s*=\\s*2',
          out: 'Dot products q . ki: 1, 0, 1. Divide each by sqrt(2) = 1.4142: scores = [0.7071, 0, 0.7071]. Softmax: weights = [0.4011, 0.1978, 0.4011], which sum to 1. Output = 0.4011 x (1, 0) + 0.1978 x (0, 1) + 0.4011 x (2, 2) = (1.2033, 1.0). Keys 1 and 3 tie, so they get the same weight.',
          note: 'matches the lab: output [1.2033, 1.0]',
          ok: true,
        },
      ],
      fallback: {
        out: 'I need the numbers to compute attention: the query, the keys, the values and d_k. Give me those, or say you mean the lab example, and I will show the scores, the weights and the output step by step.',
        note: 'too vague: give q, K, V and d_k',
        ok: false,
      },
    },
    {
      tab: '7 · reflection',
      title: 'Reflection',
      min: 15,
      kind: 'note',
      goal: 'Write two sentences on what surprised you and one thing attention can do that a plain RNN cannot.',
      tasks: ['Fill in both parts.', 'Press Save card.'],
      note: '## Week 7 reflection\n\n### (a) What surprised me\n-\n\n### (b) One thing attention can do that a plain RNN cannot\n-\n\n### Exercises I still find hard\n-',
    },
  ],
};
