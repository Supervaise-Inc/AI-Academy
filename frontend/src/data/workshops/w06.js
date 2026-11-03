// Week 6 · Tokens and word embeddings. Follows GA Lab 2 (backend/content/week-06.md, notebooks/src/week-06.py).
export default {
  week: 6,
  title: 'Tokens and embeddings lab',
  lede: 'Build a bigram model and a BPE tokenizer from scratch, then compare them with real tokenizers and sentence embeddings.',
  lab: 'GA Lab 2',
  chips: ['3 hours', 'CPU is enough', 'GPT-2 and BERT tokenizers', 'MiniLM embeddings'],
  exercises: [
    {
      tab: '1 · bigram',
      title: 'A bigram model from counts',
      min: 35,
      kind: 'code',
      goal: 'Count which word follows which in the 22-word text from the lesson figure, turn counts into probabilities and generate text.',
      tasks: [
        'Run the cell and check the counts for "the" against the figure by hand.',
        'Check that 0.375 x 0.5 = 0.1875 is the probability of "the cat sat".',
        'Replace the text with a paragraph of 100+ words, rebuild the counts and generate five sentences. Where is the output fluent and where does it break?',
        'Add a trigram version (key = the last two words). What does it fix and what does it cost?',
      ],
      code: `import random
from collections import defaultdict, Counter
text = "the cat sat on the mat . the cat saw the dog . the dog sat on the rug . the dog chased the cat ."
words = text.replace(".", " ").split()
counts = defaultdict(Counter)
for a, b in zip(words, words[1:]):
    counts[a][b] += 1
print(counts["the"])
def prob(prev, nxt):
    return counts[prev][nxt] / sum(counts[prev].values())
print(prob("the", "cat"), prob("cat", "sat"), prob("the", "cat") * prob("cat", "sat"))
def generate(start, n=8, seed=0):
    random.seed(seed)
    out = [start]
    for _ in range(n):
        nxt = counts.get(out[-1])
        if not nxt:
            break
        out.append(random.choices(list(nxt), weights=list(nxt.values()))[0])
    return " ".join(out)
print(generate("the"))`,
      run: 'Run cell',
      out: "Counter({'cat': 3, 'dog': 3, 'mat': 1, 'rug': 1})\n0.375 0.5 0.1875\n(generated text: plausible for a few words, then it drifts or repeats)",
      expect: '"the" is followed by cat 3 times, dog 3, mat 1 and rug 1, so P(cat | the) = 3/8 = 0.375. The generated text is plausible for a few words and then drifts or repeats. The exact words depend on the seed.',
    },
    {
      tab: '2 · BPE',
      title: 'Byte-pair encoding from scratch',
      min: 45,
      kind: 'code',
      goal: 'Start from characters plus the end-of-word symbol, merge the most frequent pair nine times and compare with the lesson figure.',
      tasks: [
        'Before running, work out the first three merges and their counts on paper.',
        'Run the cell and compare. The first three should match the figure exactly.',
        'Replace the corpus with six words of your own that share pieces, for example walk, walked, walking, talk, talked, talking, and list the merges.',
      ],
      code: `from collections import Counter
corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
vocab = {tuple(w) + ("</w>",): f for w, f in corpus.items()}
def pair_counts(vocab):
    pc = Counter()
    for sym, f in vocab.items():
        for a, b in zip(sym, sym[1:]):
            pc[(a, b)] += f
    return pc
def merge(vocab, pair):
    new = {}
    for sym, f in vocab.items():
        out, i = [], 0
        while i < len(sym):
            if i < len(sym) - 1 and sym[i] == pair[0] and sym[i + 1] == pair[1]:
                out.append(sym[i] + sym[i + 1]); i += 2
            else:
                out.append(sym[i]); i += 1
        new[tuple(out)] = f
    return new
merges = []
for step in range(9):
    best, n = pair_counts(vocab).most_common(1)[0]
    merges.append(best)
    vocab = merge(vocab, best)
    print(step + 1, best, n)`,
      run: 'Run cell',
      out: "1 ('e', 's') 9\n2 ('es', 't') 9\n3 ('est', '</w>') 9\n4 ('l', 'o') 7\n5 ('lo', 'w') 7\n6 ('n', 'e') 6\n7 ('ne', 'w') 6\n8 ('new', 'est</w>') 6\n9 ('low', '</w>') 5",
      expect: 'Nine merges. The first three are the pairs from "newest" (6) plus "widest" (3), so each counts 9.',
    },
    {
      tab: '3 · check merges',
      title: 'Check your BPE by hand',
      min: 15,
      kind: 'prompt',
      goal: 'Use the tutor to check a hand calculation of BPE pair counts and to debug a common mistake.',
      tasks: [
        'Ask the tutor which pair merges first in this corpus and what its count is. Give it the corpus and the word frequencies.',
        'Ask what goes wrong if you count pairs without multiplying by the word frequency.',
        'Compare each answer with your paper work.',
      ],
      presets: [
        { l: 'Vague question', u: 'How does BPE work?' },
        {
          l: 'First merge',
          u: 'My corpus is low 5, lower 2, newest 6, widest 3, each word split into characters plus </w>. Which pair is merged first and what is its count?',
        },
        {
          l: 'Debug a mistake',
          u: 'My merges do not match the figure. I counted each pair once per word and did not multiply by how often the word occurs. Why is that wrong?',
        },
      ],
      rules: [
        {
          u: 'multipl|once per word|did not .*(frequen|times)|forgot.*(</w>|end)|without.*</w>',
          out: 'BPE counts pairs over the whole corpus, so a pair inside a word that occurs 6 times counts 6, not 1. "newest" occurs 6 times, so each of its pairs counts 6. Without that, (e, s) would be 2 instead of 9 and your merge order and counts would not match the figure. Also keep the end-of-word symbol </w>: without it the first three merges (e, s), (es, t), (est, </w>) cannot come out the same.',
          note: 'the frequency multiplier is the usual slip',
          ok: true,
        },
        {
          u: 'first|which pair|merge.*count|\\(e,\\s*s\\)|newest.*widest',
          out: 'The pair (e, s) is merged first with a count of 9. It appears in "newest" (6 times) and "widest" (3 times): 6 + 3 = 9. The pairs (s, t) and (t, </w>) also count 9, so there is a tie, and the lab output lists (e, s) first. The next merges are (es, t) 9 and (est, </w>) 9. Check your paper work against those three.',
          note: 'matches the lab: (e, s) 9',
          ok: true,
        },
      ],
      fallback: {
        out: 'BPE starts from single characters and repeatedly merges the most frequent adjacent pair. To check a calculation I need your corpus, the word frequencies and what you want to know, for example the first merge and its count.',
        note: 'too vague: give the corpus and the question',
        ok: false,
      },
    },
    {
      tab: '4 · tokenisers',
      title: 'Encode new words and compare real tokenizers',
      min: 35,
      kind: 'code',
      goal: 'Apply your merges to words the corpus never contained, then count the pieces GPT-2 and BERT produce.',
      tasks: [
        'Run the cell. Read what your own BPE does with "lowest" and "newer".',
        'Record the pieces and the counts for each real tokenizer. Which one splits "unbelievably" more?',
        'How does GPT-2 mark a leading space and how does BERT mark a word continuation?',
        'Which sentence needs the most tokens per word, and what might that cost a speaker of that language? Report only what you see.',
      ],
      code: `def encode(word):
    v = {tuple(word) + ("</w>",): 1}
    for pair in merges:
        v = merge(v, pair)
    return list(next(iter(v)))
print(encode("lowest"), encode("newer"))

from transformers import AutoTokenizer
sentences = ["unbelievably good", "Mabilis ang tren.", "Tokenization matters."]
for name in ["gpt2", "bert-base-uncased"]:
    tok = AutoTokenizer.from_pretrained(name)
    for s in sentences:
        pieces = tok.tokenize(s)
        print(name, len(pieces), pieces)`,
      run: 'Run cell',
      out: "['low', 'est</w>'] ['new', 'e', 'r', '</w>']\n(six lines follow, one per tokenizer and sentence: the name, the number of pieces and the piece list. The details differ between GPT-2 and BERT, so read them from your own run.)",
      expect: '"lowest" becomes [low, est</w>] and "newer" becomes [new, e, r, </w>]. The real tokenizers print piece lists whose details differ; do not guess the counts in advance.',
    },
    {
      tab: '5 · embeddings',
      title: 'Sentence embeddings and cosine similarity',
      min: 45,
      kind: 'code',
      goal: 'Mean-pool the token vectors of the MiniLM model and compare three sentences.',
      tasks: [
        'Run the cell and read the shape and the 3x3 matrix.',
        'Explain why the diagonal is 1.0 and which pair is most similar.',
        'Add three sentence pairs of your own, including one with the same words in a different meaning, and predict the similarities before you run.',
      ],
      code: `import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
name = "sentence-transformers/all-MiniLM-L6-v2"
tok = AutoTokenizer.from_pretrained(name)
enc = AutoModel.from_pretrained(name).eval()
def embed(texts):
    batch = tok(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        h = enc(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).float()
    return F.normalize((h * mask).sum(1) / mask.sum(1), dim=1)
sents = ["A cat sat on the mat.", "A kitten rested on the rug.", "The stock market fell sharply today."]
E = embed(sents)
print(E.shape)
print((E @ E.T).numpy().round(2))`,
      run: 'Run cell',
      out: 'torch.Size([3, 384])\n(a 3x3 matrix with 1.0 on the diagonal. The cat and kitten sentences are more similar to each other than either is to the stock-market sentence. The off-diagonal values come from your run.)',
      expect: 'Shape [3, 384], ones on the diagonal, and the cat and kitten sentences closest to each other.',
    },
    {
      tab: '6 · predict',
      title: 'Predict and explain similarities',
      min: 15,
      kind: 'prompt',
      goal: 'Ask the tutor to explain the similarity matrix and a pair of your own before you run it.',
      tasks: [
        'Ask why the diagonal is 1.0 and which pair is most similar.',
        'Ask why the lab normalises the vectors instead of using a raw dot product.',
        'Write down your prediction for a same-words, different-meaning pair, ask the tutor, then run it yourself.',
      ],
      presets: [
        { l: 'Vague question', u: 'Are embeddings good?' },
        { l: 'Diagonal and pair', u: 'In the 3x3 cosine matrix, why is the diagonal 1.0 and which pair of sentences is most similar?' },
        { l: 'Cosine vs dot', u: 'Why do we normalise the embeddings and use cosine similarity instead of a raw dot product?' },
        { l: 'Same words', u: 'What should I expect for "The dog bit the man" and "The man bit the dog" in the embedding similarity?' },
      ],
      rules: [
        {
          u: 'same words|bit the (man|dog)|different meaning|word order',
          out: 'Both sentences use the same words, so a mean-pooled embedding will usually rate them as very similar, even though the meaning is reversed. The model does see word order through attention, so the score is not exactly 1, but expect it to be high. Predict a value, then run it and compare. A high score here is a limit of sentence embeddings, not proof that the two sentences mean the same.',
          note: 'predict first, then check by running',
          ok: true,
        },
        {
          u: 'cosine|dot product|normali[sz]',
          out: 'A raw dot product depends on the length of the vectors, so a longer vector scores higher even if it points the same way. Cosine similarity compares only the direction. The lab normalises each vector to length 1 first, so a plain matrix product E @ E.T then equals cosine similarity. Using a raw dot product on unnormalised vectors is a common mistake in this lab.',
          note: 'cosine compares direction only',
          ok: true,
        },
        {
          u: 'diagonal|most similar|cat.*kitten|which pair',
          out: 'The diagonal compares each sentence with itself. The vectors are normalised to length 1, so the dot product of a vector with itself is exactly 1.0. The most similar pair should be "A cat sat on the mat." and "A kitten rested on the rug.": cat and kitten, mat and rug are related words in a similar scene. The stock-market sentence is about something else, so it scores lower with both.',
          note: 'matches the lab expectation',
          ok: true,
        },
      ],
      fallback: {
        out: '"Good" depends on the task. Tell me which sentences you compare, which similarity you use (cosine or dot product) and what you want to know about the result.',
        note: 'vague prompt, vague answer',
        ok: false,
      },
    },
    {
      tab: '7 · reflection',
      title: 'Reflection',
      min: 20,
      kind: 'note',
      goal: 'Write down what a bigram model cannot do, one risk of web-text embeddings and one limit of an analogy.',
      tasks: ['Answer (a), (b) and (c) in a few sentences each.', 'Press Save card.'],
      note: '## Week 6 reflection\n\n### (a) One thing a bigram model cannot do that BPE plus embeddings can\n-\n\n### (b) One risk of embeddings learned from web text\n-\n\n### (c) One limit of the king - man + woman example\n-\n\n### Exercises I still find hard\n-',
    },
  ],
};
