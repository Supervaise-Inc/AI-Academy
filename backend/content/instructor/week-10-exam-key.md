# Week 10 — Midterm Exam: instructor answer key and marking scheme

**Confidential: for instructors only.** The student notebook (notebooks/week-10.ipynb) is a public download and contains no answers; its self-checks compare against hashed fingerprints. Part A runs in the app (exam-mode quiz id week-10-midterm, 40 questions, 1 point each). Part B (60 points) is marked by the instructor from the submitted notebook.

---

# Instructor key: Part A (written exam), Week 10 Midterm

Not for students. 40 questions, 1 point each, 60 minutes, pass mark 70 percent of Part A as a quiz (28 of 40). Calculation answers were recomputed with Python by build_exam_mid.py, which asserts every correct value and that the three distractors differ from it and from each other.

| # | Topic | Answer | Type | Level |
|---|---|---|---|---|
| 1 | Week 6: tokens and language models | C | concept | recall |
| 2 | Week 6: tokens and language models | D | calculation | recall |
| 3 | Week 6: tokens and language models | D | calculation | apply |
| 4 | Week 6: tokens and language models | A | calculation | apply |
| 5 | Week 6: tokens and language models | C | calculation | apply |
| 6 | Week 6: tokens and language models | B | concept | analyse |
| 7 | Week 6: tokens and language models | C | concept | analyse |
| 8 | Week 6: tokens and language models | C | calculation | apply |
| 9 | Week 6: tokens and language models | A | calculation | apply |
| 10 | Week 6: tokens and language models | C | calculation | apply |
| 11 | Week 7: sequences and attention | B | calculation | apply |
| 12 | Week 7: sequences and attention | D | calculation | apply |
| 13 | Week 7: sequences and attention | B | calculation | apply |
| 14 | Week 7: sequences and attention | A | concept | analyse |
| 15 | Week 7: sequences and attention | B | calculation | apply |
| 16 | Week 7: sequences and attention | A | calculation | recall |
| 17 | Week 7: sequences and attention | D | concept | recall |
| 18 | Week 7: sequences and attention | A | calculation | apply |
| 19 | Week 7: sequences and attention | C | concept | recall |
| 20 | Week 7: sequences and attention | B | calculation | recall |
| 21 | Week 8: the Transformer | D | concept | recall |
| 22 | Week 8: the Transformer | B | concept | analyse |
| 23 | Week 8: the Transformer | C | calculation | apply |
| 24 | Week 8: the Transformer | A | calculation | apply |
| 25 | Week 8: the Transformer | B | calculation | apply |
| 26 | Week 8: the Transformer | C | concept | analyse |
| 27 | Week 8: the Transformer | D | calculation | apply |
| 28 | Week 8: the Transformer | B | calculation | apply |
| 29 | Week 8: the Transformer | B | calculation | apply |
| 30 | Week 8: the Transformer | D | calculation | apply |
| 31 | Week 9: LLM families and scaling | A | concept | analyse |
| 32 | Week 9: LLM families and scaling | C | calculation | apply |
| 33 | Week 9: LLM families and scaling | C | calculation | apply |
| 34 | Week 9: LLM families and scaling | D | calculation | apply |
| 35 | Week 9: LLM families and scaling | A | calculation | recall |
| 36 | Week 9: LLM families and scaling | A | calculation | apply |
| 37 | Week 9: LLM families and scaling | D | calculation | apply |
| 38 | Week 9: LLM families and scaling | B | concept | analyse |
| 39 | Week 9: LLM families and scaling | D | concept | analyse |
| 40 | Week 9: LLM families and scaling | A | calculation | apply |

Level counts: recall 8, apply 24, analyse 8

Answer position counts: A 10, B 10, C 10, D 10

Calculation questions: 28

## Working for calculation questions

**Q2** (Week 6: tokens and language models), correct D: 0.4

- Working: 12/30 = 0.40. Distractors: 12/1000 = 0.012; 1-0.4 = 0.6; 30/12 = 2.5.
- Options: A 0.012; B 0.6; C 2.5; D 0.4

**Q3** (Week 6: tokens and language models), correct D: 0.0172

- Working: 1/(8+50) = 0.0172. Distractors: 1/8 = 0.125; 1/50 = 0.02; 1/(8+1) = 0.1111.
- Options: A 0.125; B 0.0200; C 0.1111; D 0.0172

**Q4** (Week 6: tokens and language models), correct A: 4

- Working: P = 0.00390625; P^(-1/4) = 4.000000. Distractors: 1/P = 256; P^(-1/2) = 16; P^(1/4) = 0.25.
- Options: A 4; B 256; C 16; D 0.25

**Q5** (Week 6: tokens and language models), correct C: 0.158

- Working: 0.6*0.25 + 0.4*0.02 = 0.158. Distractors: bigram part only 0.150; plain mean 0.135; weights swapped 0.112.
- Options: A 0.150; B 0.135; C 0.158; D 0.112

**Q8** (Week 6: tokens and language models), correct C: b + a (count 10)

- Working: Pair counts: {'b a': 10, 'a n': 9, 'n </w>': 9, 'c a': 3, 'a t': 4, 't </w>': 4}. Top: b a = 10. a n = 9, n </w> = 9, c a = 3.
- Options: A a + n (count 9); B n + </w> (count 9); C b + a (count 10); D c + a (count 3)

**Q9** (Week 6: tokens and language models), correct A: 16,384,000

- Working: 32000*512 = 16,384,000. Distractors: V+d = 32,512; d*d = 262,144; V*V = 1,024,000,000.
- Options: A 16,384,000; B 32,512; C 262,144; D 1,024,000,000

**Q10** (Week 6: tokens and language models), correct C: 0.889

- Working: dot = 8; |a| = 3.0; |b| = 3.0; cos = 0.8889. Distractors: dot 8; dot/(|a|+|b|) = 1.333; dot/(|a|^2|b|^2) = 0.099.
- Options: A 8; B 1.333; C 0.889; D 0.099

**Q11** (Week 7: sequences and attention), correct B: 6,208

- Working: 4096 + 2048 + 64 = 6208. Distractors: x100 = 620,800; no bias 6144; LSTM 4x = 24,832.
- Options: A 620,800; B 6,208; C 6,144; D 24,832

**Q12** (Week 7: sequences and attention), correct D: 0.035

- Working: 0.8^15 = 0.0352. Distractors: 0.8^14 = 0.0440; 1-0.8^15 = 0.965; 0.8*15 = 12.
- Options: A 0.044; B 0.965; C 12; D 0.035

**Q13** (Week 7: sequences and attention), correct B: 0.296

- Working: 0.97^40 = 0.2957. Distractors: 1-v = 0.704; 0.97; 0.97/40 = 0.0243.
- Options: A 0.704; B 0.296; C 0.970; D 0.024

**Q15** (Week 7: sequences and attention), correct B: 3.88

- Working: weights [0.5761, 0.2119, 0.2119]; sum w*v = 3.8806. Distractors: score/sum(score) weights 3.50; hard pick 6; mean 2.67.
- Options: A 3.50; B 3.88; C 6.00; D 2.67

**Q16** (Week 7: sequences and attention), correct A: 3

- Working: 24/sqrt(64) = 3.0. Distractors: 24; 24/64 = 0.375; 24*8 = 192.
- Options: A 3; B 24; C 0.375; D 192

**Q18** (Week 7: sequences and attention), correct A: 0.87

- Working: exp(4)=54.598, exp(2)=7.389, exp(0)=1; top = 0.8668. T=1 gives 0.6652; scores/2 gives 0.5065; uniform 0.3333.
- Options: A 0.87; B 0.67; C 0.51; D 0.33

**Q20** (Week 7: sequences and attention), correct B: 9,000,000

- Working: 3000^2 = 9,000,000. Distractors: n = 3,000; 2n = 6,000; n^3 = 27,000,000,000.
- Options: A 3,000; B 9,000,000; C 6,000; D 27,000,000

**Q23** (Week 8: the Transformer), correct C: 0.479

- Working: 10000^(2/8) = 10.0; angle = 0.5; sin = 0.4794. Distractors: cos(0.5) = 0.8776; sin(5) = -0.9589; sin(5/100) = 0.0500.
- Options: A 0.878; B −0.959; C 0.479; D 0.050

**Q24** (Week 8: the Transformer), correct A: 2,359,296 weights; d_head = 64

- Working: 4*768^2 = 2,359,296; d_head = 64. Distractors: 768^2 = 589,824; 3*768^2 = 1,769,472; 12 heads is not d_head.
- Options: A 2,359,296 weights; d_head = 64; B 589,824 weights; d_head = 64; C 1,769,472 weights; d_head = 768; D 2,359,296 weights; d_head = 12

**Q25** (Week 8: the Transformer), correct B: About 67%

- Working: attention 2,359,296; FFN 4,718,592; share = 0.6667. Distractors: attention share 0.333; equal 0.5; 4/5 = 0.8.
- Options: A About 33%; B About 67%; C About 50%; D About 80%

**Q27** (Week 8: the Transformer), correct D: 0.73

- Working: masked softmax over (1,2): 0.7311 (token 1 gets 0.2689). Unmasked softmax weight 0.0347.
- Options: A 0.03; B 0.50; C 0.27; D 0.73

**Q28** (Week 8: the Transformer), correct B: 1.56

- Working: -ln p = [0.693, 1.386, 2.079, 2.079]; sum 6.238; mean 1.5596. Distractors: sum 6.24; -ln(mean p) = 1.386; mean p = 0.25.
- Options: A 6.24; B 1.56; C 1.39; D 0.25

**Q29** (Week 8: the Transformer), correct B: 197 MB

- Working: 2*24*2000*1024*2 = 196,608,000 bytes = 196.6 MB. Distractors: no K/V factor 98.3 MB; 4 bytes 393.2 MB; one layer 8.2 MB.
- Options: A 98 MB; B 197 MB; C 393 MB; D 8 MB

**Q30** (Week 8: the Transformer), correct D: 162,304

- Working: emb 64,000; blocks 98,304; total 162,304. Distractors: blocks only 98,304; one block 113,152; untied 226,304.
- Options: A 98,304; B 113,152; C 226,304; D 162,304

**Q32** (Week 9: LLM families and scaling), correct C: BERT: 60; GPT: 399

- Working: 0.15*400 = 60; GPT n-1 = 399; ratio 6.65. Distractors: swapped; both 60; both 400.
- Options: A BERT: 399; GPT: 60; B BERT: 60; GPT: 60; C BERT: 60; GPT: 399; D BERT: 400; GPT: 400

**Q33** (Week 9: LLM families and scaling), correct C: 51 billion total; 15 billion active

- Working: total 51B; active 15B. Distractors: no shared 48/12; all experts 51/51; top-1 3+6=9.
- Options: A 48 billion total; 12 billion active; B 51 billion total; 51 billion active; C 51 billion total; 15 billion active; D 51 billion total; 9 billion active

**Q34** (Week 9: LLM families and scaling), correct D: 2.25

- Working: 10^(-0.152) = 0.7047; 3.2 * that = 2.2550. Distractors: 3.2*0.839 = 2.686; 3.2*0.924 = 2.957; half = 1.60.
- Options: A 2.69; B 2.96; C 1.60; D 2.25

**Q35** (Week 9: LLM families and scaling), correct A: 1.08 × 10²¹

- Working: 6*3e9*6e10 = 1.080e+21. Distractors: ND = 1.80e+20; 2ND = 3.60e+20; x10 = 1.08e+22.
- Options: A 1.08 × 10²¹; B 1.8 × 10²⁰; C 3.6 × 10²⁰; D 1.08 × 10²²

**Q36** (Week 9: LLM families and scaling), correct A: N = 10 billion parameters, D = 200 billion tokens

- Working: N = sqrt(1.2e+22/120) = 1.000e+10; D = 2.000e+11; 6ND = 1.20e+22. Distractors: sqrt(C/60) = 1.414e+10, D 2.83e+11; D = 2N; N x10.
- Options: A N = 10 billion parameters, D = 200 billion tokens; B N = 14 billion parameters, D = 282 billion tokens; C N = 10 billion parameters, D = 20 billion tokens; D N = 100 billion parameters, D = 2 trillion tokens

**Q37** (Week 9: LLM families and scaling), correct D: 0.60

- Working: 0.95^10 = 0.5987. Distractors: 0.95; 1-v = 0.40; 1-10*0.05 = 0.50.
- Options: A 0.95; B 0.40; C 0.50; D 0.60

**Q40** (Week 9: LLM families and scaling), correct A: About 0.048 (4.8 points)

- Working: sqrt(0.2304/100) = 0.0480. Distractors: var/n = 0.002304; p/n = 0.0064; sqrt(var) = 0.48.
- Options: A About 0.048 (4.8 points); B About 0.0023; C About 0.0064; D About 0.48

---

# Instructor key: Week 10 Midterm practical exam (Part B), Tasks 1 and 2 (26 points)

Not for students. All numbers were computed by running the reference solution (seed 42, CPU, PyTorch of the test venv). Reference notebook: `w10/testP1/week-10.py`; every self-check prints a tick for it, and the unmodified student starter runs without errors (all crosses). The self-checks accept the value rounded to the stated decimals (plus or minus 0.4 of the last digit) and never show it. Marks are in whole or half points. Rule for all code questions: correct method but wrong number earns half of the numeric points; a hard-coded number with no working earns at most half. No points are deducted for a wrong answer.

## Task 1. Tokens and language models (12 points)

The text is `TEXT1` (30 words, plus 4 full stops). With the leading full stop the token list has 35 tokens; V = 17 different tokens (the full stop included): `.` and, at, buyer, buys, carries, ferry, fish, leaves, market, port, rice, sells, six, the, to, vendor.

### B1.1 Counting bigrams (2)
- Model answer: counts after "the": fish 3, ferry 2, market 2, port 1, vendor 1, buyer 1 (total 10). V = 17. P(fish | the) = 3/10 = 0.300. Most common word after "the": fish.
- Tolerance: V exact; probability 0.300 (3 decimals); word exact (case-insensitive).
- Marking: V 0.5; `p_fish_after_the` 1; `most_common_after_the` 0.5. `bigram_counts` must be real code (loop over adjacent pairs). Common errors: V = 16 (full stop forgotten) or 18 (counted "The" and "the" separately, or the start token twice): 0, half a mark if the reasoning is visible. Dividing by the total count of all bigrams (34) instead of the count of "the" gives 0.088: 0.
- If `bigram_counts` is missing but the numbers are right with a hard-coded or manual count: at most half of each numeric mark.

### B1.2 Add-one smoothing and perplexity (3)
- Test sentence: `. the vendor buys the fish .` (six transitions). Counts and probabilities:

| transition | count | count of previous | raw P | add-one P (V = 17) |
|---|---|---|---|---|
| . to the | 4 | 4 | 1.000 | 5/21 = 0.2381 |
| the to vendor | 1 | 10 | 0.100 | 2/27 = 0.0741 |
| vendor to buys | 0 | 1 | 0 | 1/18 = 0.0556 |
| buys to the | 1 | 1 | 1.000 | 2/18 = 0.1111 |
| the to fish | 3 | 10 | 0.300 | 4/27 = 0.1481 |
| fish to . | 0 | 3 | 0 | 1/20 = 0.0500 |

- Model answer: `p_test_raw` = 0 (two unseen pairs, "vendor buys" and "fish ."; the product is 0). `p_buys_after_vendor` = 1/18 = 0.0556. Perplexity = exp(−mean ln P) = 10.365 (the product of the six smoothed probabilities is 8.1e-7; 1/(8.1e-7) to the power 1/6 = 10.365).
- Tolerance: `p_test_raw` 0 (exactly); `p_buys_after_vendor` 4 decimals (0.0556); `ppl_smooth` 2 decimals (10.37, 10.36 or 10.37 accepted).
- Marking: `p_test_raw` 0.5; `p_buys_after_vendor` 1; `ppl_smooth` 1.5. Typical errors: using V = 16 or 18 in the denominator (the check fails, but if the formula (count + 1) / (count of previous + V) is clear: 0.5 for the probability and 1 for the perplexity if it is otherwise right); adding 1 to the numerator only (1/1 = 1.0 for the denominator): 0 with half a mark for the idea; using only the five transitions that exclude the leading full stop: half of the perplexity mark; forgetting the root (product of 1/P, not the Nth root): half.
- The written sentence is not worth separate points but is read: expected idea is that one unseen pair makes the probability 0 and the perplexity infinite (so the number is unusable). If missing or wrong, deduct half a mark from the perplexity mark only when the student also reports an infinity or crashes without commenting.

### B1.3 A BPE merge trace (3)
- Corpus: rain 6, train 4, main 3, mango 5, range 2; each word is written as letters plus `</w>`. At the start the corpus has 111 symbols. Trace (ties go to the pair seen first when the words are scanned in list order, left to right):

| Merge | Pair | Count | Runner-up pairs | Symbols afterwards |
|---|---|---|---|---|
| 1 | a + i | 13 | i n 13, n </w> 13 (tie; a i is seen first), r a 12 | 98 |
| 2 | ai + n | 13 | n </w> 13 (tie; ai n is seen first), r ai 10 | 85 |
| 3 | ain + </w> | 13 | r ain 10, a n 7 | 72 |
| 4 | r + ain</w> | 10 | a n 7, n g 7 | 62 |
| 5 | a + n | 7 | n g 7 (tie; a n is seen first, in "mango"), m a 5 | 55 |
| 6 | an + g | 7 | m an 5, g o 5, o </w> 5 | 48 |

- Model answer: `merge_string` = `a+i ai+n ain+</w> r+ain</w> a+n an+g`. `total_tokens` = 48 (rain = 1 symbol, train = 2, main = 2, mango = 4, range = 4; 6·1 + 4·2 + 3·2 + 5·4 + 2·4 = 48).
- Tolerance: the self-check needs the string exactly in the stated format (single spaces, `a+b`, `</w>` as the end marker); a differently formatted but clearly correct list is accepted by the marker. `total_tokens` exact.
- Marking: `merge_string` 2 (fully right). Partial credit: first three merges right 1; first two right 0.5. A student who breaks ties differently (for example, alphabetically) and gets a different but internally consistent trace: 1 of the 2 and the total tokens mark if consistent with their own merges. `total_tokens` 1: right number for the student's own correct merges gets the mark. Code: `learn_bpe` must count pairs weighted by word count and merge in all words; an unweighted count (each word counted once) gives different counts and, after the first three merges, a different order of the later merges: 0.5 for the string.

### B1.4 Encode new words (1.5)
- Model answer: with the six merges in order: "drain" = d rain</w> (2 pieces); "mangoes" = m ang o e s </w> (6 pieces; only the merges a+n and an+g apply); "strain" = s t rain</w> (3 pieces). Mean = (2 + 6 + 3) / 3 = 3.667.
- Tolerance: `pieces_drain` exactly "d rain</w>"; `tokens_per_word` 3 decimals (3.667).
- Marking: `pieces_drain` 0.5; `tokens_per_word` 1. Common errors: forgetting the end marker `</w>` (then the merges that contain the marker do not apply and the pieces change): 0.5 if the method is right and the marker omission is the only error; applying the merges in frequency order of the new word instead of the learned order: 0.
- The code must apply the merges in the learned order; a hard-coded answer: half.

### B1.5 An embedding table (1.5)
- Model answer: cosine(mango, banana) = 0.9915 (dot 0.8·0.7 + 0.1·0.2 + 0.6·0.5 = 0.88; norms 1.0100 and 0.8832). Cosines with "jeepney": bus 0.965, banana 0.110, mango −0.011, rice −0.443; the closest is **bus**. `emb_params` = 50,000 × 768 = 38,400,000.
- Tolerance: 0.991 (3 decimals; the exact value is 0.99146); word exact; integer exact (38,400,000; accept 38.4 million).
- Marking: 0.5 + 0.5 + 0.5. Common errors: using the dot product without dividing by the norms (0.88): 0; adding the bias or the position (50,000 × 768 + 768): 0.

### B1.6 Which tokenizer? (1)
- Full marks for 3 to 4 sentences that (a) choose a tokenizer and give a reason tied to the scenario and (b) give the two costs with numbers, 0.5 each.
- The best answer is B (BPE): a word-level vocabulary meets the many word forms as unknown tokens (about 6 percent of the words here, a loss of information), and BPE can spell any word from pieces. Costs of B: longer sequences, 1,500 against 1,000 tokens, 50 percent more steps and a longer share of the context window, and attention grows with the square (about 2.25 times as many scores); its embedding table is smaller: 8,000 × 512 = 4,096,000 parameters against 30,000 × 512 = 15,360,000 for A.
- A defensible answer for A is accepted if it names the unknown-token problem and still justifies the choice with numbers (for example "if the community language has little variation"); marking is on correct numbers and correct reasoning, not on the choice.
- Marking: 0.5 for a reasoned choice that names the unknown tokens or the word forms; 0.5 for the two numbers (embedding table size 4,096,000 and 15,360,000, and the sequence length trade-off). If the student reverses the trade-off (says BPE has shorter sequences): 0 for that half.

Task 1 total: 2 + 3 + 3 + 1.5 + 1.5 + 1 = 12.

## Task 2. Sequences and attention (14 points)

### B2.1 Vanishing gradient through time (3)
- Model answer: f = 0.9·(1 − 0.36) = 0.576. g12 = 0.576^12 = 0.001334 (1.334e-3). The smallest L with 0.576^L < 0.0001: 0.576^16 = 1.468e-4 (not yet) and 0.576^17 = 8.46e-5, so `L_min` = 17 (from ln(0.0001)/ln(0.576) = 16.7, rounded up).
- Tolerance: `f` 0.5760 (4 decimals); `g12` 5 decimals (0.00133); `L_min` exact.
- Marking: `f` 0.5; `g12` 0.5; `L_min` 1 (0.5 if 16 or 16.7 is given with the right method: they stopped at "at most", without rounding up). Written sentence 1: expected idea, with |f| below 1 the product shrinks exponentially (vanishing: the early steps get no signal); with |f| above 1 it grows exponentially (exploding; w = 1.8 gives f = 1.152 and 1.152^12 = 5.46); a remedy for the exploding case: gradient clipping (also accepted: a smaller recurrent weight, a better initialisation, an LSTM, a lower learning rate). 0.5 for the two cases, 0.5 for a correct remedy.

### B2.2 The LSTM cell highway (2)
- Model answer: the running products are 0.97, 0.8924, 0.8835, 0.7510, 0.7134, so `cell_grad` = 0.6991 (0.699139). `f_const` = 0.5^(1/40) = 0.98282 (4 decimals: 0.9828).
- Tolerance: 4 decimals.
- Marking: 1 + 1. Common errors: adding or averaging the gates instead of multiplying (mean 0.943): 0; 0.5^(1/40) computed as 0.5/40 = 0.0125: 0; using 20 steps: 0.5 for method. For comparison (not asked): the same six steps in the RNN of B2.1 give 0.576^6 = 0.0365, so the LSTM keeps about 19 times more.

### B2.3 Attention by hand (4)
- Model answer: scores q·k = (2.0, 3.0, 0.5): key 1: 1·1 + 0.5·0 + (−1)·1 + 2·1 = 2.0; key 2: 0 + 1.0 + 1.0 + 1.0 = 3.0; key 3: 2.0 + 0.5 + 0 − 2.0 = 0.5. Divided by √4 = 2: (1.0, 1.5, 0.25); softmax gives `w_scaled` = (0.3204, 0.5283, 0.1513). Output = 0.3204·(1, 0) + 0.5283·(0, 2) + 0.1513·(3, 1) = (0.7744, 1.2079). Unscaled softmax of (2, 3, 0.5) = (0.2537, 0.6897, 0.0566): `w_unscaled_max` = 0.6897 (the output would be (0.4236, 1.4360)).
- Tolerance: 3 decimals on every number.
- Marking: scores 1; weights 1; output 1 (the right output for the student's own weights earns the point); `w_unscaled_max` 0.5. Written 0.5: with large d_k the dot products have a large variance (they grow like √d_k), the softmax saturates and becomes nearly hard (one weight near 1), and the gradient is nearly 0; dividing by √d_k keeps the scores near the scale of 1. Half for "makes the numbers smaller" without the saturation or the slope.
- Common errors: dividing by d_k = 4 instead of √d_k (weights 0.3366, 0.4321, 0.2313; output (1.0305, 1.0956)): 0 for the weights, 1 for the output if consistent with their weights; softmax along the wrong axis; using K·V in place of the weights (shape error).

### B2.4 Soft against hard attention, with autograd (2)
- Model answer: softmax(1, 2, 0.5) = (0.2312, 0.6285, 0.1402); the output is 0.5769. The gradient with respect to the first score is p₁·(v₁ − output) = 0.2312·(4 − 0.5769) = 0.7915 (the full gradient is (0.7915, −0.9911, 0.1996)). For hard attention the output is `vals_hard[s_hard.argmax()]`; backward runs (it flows into `vals_hard`), but `s_hard.grad` is None: `hard_grad_exists` = False.
- Tolerance: `grad_soft_s0` 4 decimals (0.7915); the boolean exactly.
- Marking: 1 + 0.5. Written sentence 0.5: the argmax is a step function with no slope, so no gradient reaches the scores, so hard attention cannot be trained by ordinary backpropagation (it needs sampling methods such as REINFORCE), and soft attention can; this is why LLMs use soft attention. If a student gets `s_hard.grad` as zeros (for example by using a one-hot with softmax first) and says "zero gradient", accept it as `hard_grad_exists` True with 0.5 for the idea if explained; the check will show a cross, the marker overrides.

### B2.5 A memory experiment (3)
- Model answer: the test accuracies are `rnn8` = 1.00, `lstm8` = 1.00, `rnn60` = 0.107, `lstm60` = 0.120 (chance is 0.125). The run takes about 15 to 25 seconds on a CPU. In a fresh network of 40 steps, the gradient of the last output with respect to the first input is 1.5e-11 times the one at the last step for the RNN (`decades_rnn` = 10.82, so 11 powers of ten weaker) and 4.0e-9 for the LSTM (`decades_lstm` = 8.40): the LSTM keeps the signal about 270 times better but still loses about 8 powers of ten, because a fresh LSTM has forget gates near 0.5, not near 1.
- Tolerance: the numbers vary in the last digit between machines; accuracy at length 60 anywhere below 0.3 is acceptable; `decades_*` are checked to the nearest whole number (the check accepts 11 and 8; a student's 10.8 and 8.4 pass).
- Marking: (a) the four accuracies recorded 1 (0.5 if only two of them, or if the code does not run the experiment but types numbers); (b) `decades_rnn` and `decades_lstm` 0.5 (0.25 each; the sign must be positive, that is −log10); (c) the written interpretation 1.5, in three parts of 0.5:
  1. Reads their own numbers correctly: both models learn at length 8; at length 60 both are near chance (about 0.125); the RNN fails at the long length as the lesson predicts.
  2. Explains with the gradient: the signal from step 1 is many powers of ten weaker than the signal from the last step (vanishing gradient), so the early symbol cannot be learned from the error at the end; the LSTM's number is better (smaller loss of signal), and its gates help but did not rescue the task in 50 epochs at length 60.
  3. A sensible change: train much longer or with a larger learning-rate schedule; initialise the forget-gate bias near 1; gradient clipping (helps the exploding side only); a shorter distance; or attention, which connects the last step directly to the first (the lesson's solution). "Use a bigger model" earns 0.25.
  A student who writes that the LSTM "solves" the task at length 60 contradicts their own number: 0 for part 2.
- Note for markers: the result for the LSTM at length 60 depends on the epochs and learning rate; with 50 epochs at 0.003 it stays at chance on every seed tried (0 and 42). In the lesson lab an LSTM trained longer does learn. Do not penalise a student who explains this.

Task 2 total: 3 + 2 + 4 + 2 + 3 = 14. Tasks 1 and 2 together: 26 points of the 60 in Part B.

---

# Week 10 — Midterm Exam: instructor key, Part B Tasks 3 to 5

**Confidential: for instructors only.** All numbers below were computed by running the reference solution (the same code that produced the fingerprints in the public notebook). Self-checks in the notebook only print a tick or a cross; the mark is given from the code, the numbers and the written answers. Tasks 3 to 5 are worth 16 + 12 + 6 = 34 of the 60 points of Part B (Tasks 1 and 2 are keyed separately). Where a student's earlier answer is wrong but later work is right for the wrong input, give the later marks (own-error carried forward).

---

## Task 3: The Transformer and autoregressive training (16 points)

### B3.1 Sinusoidal positions (3 points)
Model answer. `sinusoid(pos, d)`: for i in 0..d/2-1, angle = pos / 10000^(2i/d), entry 2i = sin(angle), entry 2i+1 = cos(angle). Test values: `sinusoid(13, 6)` = [0.42017, 0.90745, 0.56745, 0.82341, 0.02800, 0.99961]; `sinusoid(2, 4)` = [0.90930, -0.41615, 0.02000, 0.99980]. `dot_near` (positions 20 and 21, d = 8) = 3.5353; `dot_far` (20 and 60) = 0.5997. `wavelength` = 2π · 10000^(6/8) = 2π · 1000 = 6283.19 positions.
Tolerance: 3 decimals for the vectors and dot products; the wavelength to the nearest whole number (6283).
- 1 point: the function (sine on even, cosine on odd entries, exponent 2i/d, base 10000).
- 1 point: both dot products (0.5 each). The near pair has the larger dot product, as the lesson says nearby positions look alike.
- 1 point: wavelength. Full marks for 2π · 10000^(2·3/8); 0.5 if they give the period of the angle in the wrong pair (for example i = 3 counted as pair index 6) or forget the 2π (1000).
Typical errors: sin and cos swapped (loses the function point, dot products follow their own function: carry forward); using the index of the entry (2i) instead of i in the exponent; angles in degrees; exponent i/d instead of 2i/d (the function is then wrong but consistent: 0 for the function, full for the others if carried forward).

### B3.2 Multi-head shapes and parameters (2 points)
Model answer. `d_head` = 96 / 6 = 16; `q_shape` = (4, 6, 10, 16), obtained with `.view(4, 10, 6, 16).transpose(1, 2)` (or `reshape`). `attn_formula` = 4·96² + 4·96 = 36,864 + 384 = 37,248; `attn_torch` = 37,248 (`nn.MultiheadAttention(96, 6)`: in-projection 3·96² + 3·96 plus output projection 96² + 96); `attn_12` = 37,248 (the number of heads does not appear).
- 1 point: `d_head` = 16 and the shape (4, 6, 10, 16) (0.5 each). A shape (4, 10, 6, 16) (before the transpose) earns 0.25.
- 1 point: the three counts, all 37,248 (formula 0.5, torch 0.25, 12 heads 0.25).
Typical errors: leaving out the biases (36,864); 3d² (forgetting the output projection); 12 heads giving half the parameters, or twice (counting per head); d_head = 6.

### B3.3 Layer normalisation (2 points)
Model answer. mean = 4, variance (divisor 5) = 18, so `v_hat` = (v − 4) / √(18 + 1e-5) = [0, 1.41421, −1.41421, 0.70711, −0.70711]; `ln_out` = gamma · v_hat + beta = [0, 1.41421, −0.91421, −0.64645, −1.16421].
Tolerance: 3 decimals.
- 1 point: `v_hat` (mean over the five numbers of the token, variance with divisor 5, eps inside the square root).
- 1 point: `ln_out` (scale then shift, in that order, element-wise).
Typical errors: sample variance (divisor 4): v_hat = [0, 1.26491, −1.26491, 0.63246, −0.63246] (0.5 off; ln_out carried forward gets its point); normalising over a different axis; beta applied before gamma; forgetting the square root (huge values).

### B3.4 One row of masked attention (2 points)
Model answer. Scaled scores = raw / √16 = [0.5, −0.3, 0.9, 0.2, 1.3]; the key at position 4 is in the future of position 3 and gets −∞ before the softmax; `w_row` = [0.27159, 0.12203, 0.40517, 0.20120, 0.0]; `out_row` = w_row · V_mat = [1.28590, 0.85044].
Tolerance: 3 decimals.
- 1 point: `w_row` (scale by √d_k, mask before the softmax so that the weights sum to 1 over positions 0 to 3, and the last weight is exactly 0).
- 1 point: `out_row` (weighted sum of the value rows, shape (2,)); carried forward from a wrong `w_row`.
Typical errors: no mask (w = [0.16928, 0.07606, 0.25253, 0.12540, 0.37673]); setting the weight of position 4 to 0 after the softmax so the weights do not sum to 1 (0.5 off); no scaling (w = [0.15886, 0.00648, 0.78682, 0.04785, 0]); scaling by d_k instead of √d_k; multiplying with V_mat.T.

### B3.5 The shifted-target loss (2 points)
Model answer. `targets` = [0, 3, 1, 4] (tokens[1:]); the probabilities of the targets are 0.30, 0.45, 0.10, 0.25 (P[i, targets[i]]); `loss_shift` = mean(−ln p) = 1.42284 nats; `ppl` = exp(1.42284) = 4.14889. `loss_bug` (targets = tokens[:-1] = [2, 0, 3, 1], probabilities 0.25, 0.10, 0.30, 0.30) = 1.52421.
Tolerance: 3 decimals for the losses, 2 for the perplexity.
- 1 point: `targets`, `loss_shift` and `ppl` (0.25, 0.5, 0.25).
- 1 point: `loss_bug` (0.5) and the sentence in `ANSWERS["B3.5"]` (0.5): the bug trains the model to reproduce the token it is reading (copy its input) instead of predicting the next token, so the loss is meaningless for generation; the shift by one is what makes the task prediction.
Typical errors: log base 10 or 2 (0.618 or 2.053); mean of the probabilities instead of the mean of −ln p; using the product instead of the mean (perplexity then wrong); targets as tokens[:4] (that is the bug) or tokens[2:].

### B3.6 Train two settings and compare (3 points)
Model answer (CPU, seed 42, 250 steps, the numbers vary slightly between machines and PyTorch versions; the checks are tolerant). Vocabulary 22 characters, floor = 0.148 nats per character. Model A (d = 16, 2 heads, 1 layer): 4,176 parameters, validation loss about 1.39, gap about 1.24. Model B (d = 48, 4 heads, 2 layers): 59,232 parameters, validation loss about 0.38, gap about 0.23. `n_ratio` = 59,232 / 4,176 = 14.18 (exact, 2 decimals). For reference ln 22 = 3.09 is the loss of a model that knows nothing.
- 1 point: both models built with the right settings (seed set before each), trained with `train_lm(..., steps=250, block_size=32)` and `loss_a`, `loss_b` equal to `eval_lm` on `val_data`.
- 1 point: `gap_a`, `gap_b` (loss minus floor) and `n_ratio` = 14.18. 0.5 for the gaps, 0.5 for the ratio.
- 1 point: `ANSWERS["B3.6"]`: compares the two with the numbers (B lower loss and smaller gap, with about 14 times more parameters), explains the floor (the entropy of the grammar: the lowest loss any model can reach, so a gap of zero would be perfect), and suggests one sensible change for A (more width or layers, more training steps). 0.5 for the comparison with numbers, 0.5 for the floor and the change.
Typical errors: training on the validation data or evaluating on the training data (0 for the loss point); forgetting to reseed so that the numbers differ (accept if the conclusions are the same); calling the floor "the best validation loss we got" or "zero loss"; claiming that a bigger model always wins (full credit here only if tied to the data: the gap shows B is closer; a good answer notes that B could overfit with more steps on 5,000 characters).

### B3.7 Count a whole configuration (2 points)
Model answer. V = 22, d = 40, L = 3, block = 24. Per block: attention 4d² + 4d = 6,560; feed-forward 2·d·4d + 4d + d = 12,800 + 160 + 40 = 13,000; layer norms 4d = 160; total per block 6,560 + 13,000 + 160 = 19,720 (= 12d² + 13d). Whole model: V·d = 880; positions 24·40 = 960; blocks 3 · 19,720 = 59,160; final norm 2d = 80; output tied (0). `n_formula` = `n_model` = 61,080. The rule 12·L·d² = 57,600 is below the exact 59,160 by 1,560, so `rule_err_pct` = 100 · 1,560 / 59,160 = 2.64.
Tolerance: exact integers; 2 decimals for the percentage.
- 1 point: `n_formula` and `n_model` = 61,080 (0.5 each; a model count that differs from the formula count and is not explained gets 0.5 for the model count only).
- 1 point: `rule_err_pct` = 2.64 (0.5 if the percentage is measured against the rule instead of the exact count, 2.71, or if it is against the whole model).
Typical errors: an untied output layer (61,960); no position table (60,120); no biases and no layer norms (59,440); counting only one layer norm per block; 12 L d² as the exact count.

**Task 3 total: 3 + 2 + 2 + 2 + 2 + 3 + 2 = 16.**

---

## Task 4: LLM families, scaling laws and prompting (12 points)

### B4.1 Training signals (1 point)
Model answer. `n_mlm` = 0.15 · 400 = 60; `n_causal` = 399 (every position except the last has a next token); `signal_ratio` = 399 / 60 = 6.65.
- 1 point: three values (0.5 for the two counts, 0.5 for the ratio). Accept `n_causal` = 400 with the ratio 6.67 for 0.5.
Typical errors: 400 targets for the causal model; the masked share applied to the wrong model; ratio inverted (0.15).

### B4.2 Routing and parameters in a mixture of experts (2 points)
Model answer. Top-2 scores are 2.3 (expert 1) and 1.7 (expert 3); `top_idx` = [1, 3]; `top_w` = softmax([2.3, 1.7]) = [0.64566, 0.35434]. One expert has 2·256·1024 + 1024 + 256 = 525,568 parameters; `total_params` = 6 · 525,568 = 3,153,408; `active_params` = 2 · 525,568 = 1,051,136 (total / active = 3).
- 1 point: indices and weights (0.5 each).
- 1 point: total and active (0.5 each).
Typical errors: softmax over all six scores (weights 0.42935 and 0.23563 do not sum to 1) loses the weights point; indices from the sorted order of scores [3, 1] (accept with weights in the matching order, as long as consistent: the task asks for increasing order, deduct 0.25); omitting biases (total 3,145,728); counting only one expert as active.

### B4.3 Training compute and the Chinchilla rule (2 points)
Model answer. N = √(2.4e22 / 120) = √(2.0e20) = 1.41421e10, so `N_opt_b` = 14.142 billion parameters; D = 20 N, `D_opt_b` = 282.84 billion tokens. R1: 6 · 12e9 · 96e9 = 6.912e21 FLOPs, `flops_r1` = 6.912; tokens per parameter: `tpp_r1` = 96e9 / 12e9 = 8, `tpp_r2` = 150e9 / 3e9 = 50.
Tolerance: 3 decimals for N, 2 for D.
- 1 point: `N_opt_b`, `D_opt_b` (0.5 each).
- 1 point: `flops_r1` (0.5) and the two ratios (0.25 each).
Typical errors: C = 2ND; N = C/120 without the square root; units confused (billions versus millions; trillions); D = N / 20.

### B4.4 The fitted loss formula (2 points)
Model answer. `chin_loss(N, D)` = 1.8172 + 482.01 / N^0.3478 + 2085.43 / D^0.3658. Test values: `chin_loss(7e9, 1.1e11)` = 2.18938; `chin_loss(2e10, 4e11)` = 2.06209. Design A (14e9, 280e9): `L_a` = 2.09530. Design B: same compute C = 6 · 14e9 · 280e9 = 2.352e22, so D = C / (6 · 56e9) = 70e9 tokens, `L_b` = 2.13021 and `gap` = 0.03491 (B, the bigger and under-trained model, is worse by about 0.035).
Tolerance: 3 decimals.
- 1 point: the function (both test values).
- 1 point: `L_a`, `L_b` and `gap` (0.5 for the two losses, 0.5 for the gap with the right sign).
Typical errors: using the same D = 280e9 for B (loss 2.0407, lower than A, but wrong because the compute is then four times larger); sign of the gap reversed; N^alpha and D^beta swapped; using billions in the formula (the constants are for raw counts of parameters and tokens).

### B4.5 Exact match and per-token accuracy (1 point)
Model answer. `p_exact` = 0.93^12 = 0.419 (0.41860); `p_needed` = 0.5^(1/12) = 0.9439 (0.94387).
- 1 point: both (0.5 each).
Typical errors: 0.93 × 12 = 11.2; 12 × 0.93 / ... ; p_needed = 0.5 / 12 or 1 − 0.5/12 = 0.9583.

### B4.6 Validate a reply (2 points)
Model answer.
```python
def check_reply(reply):
    try:
        obj = json.loads(reply.strip())
    except Exception:
        return False
    if not isinstance(obj, dict):
        return False
    name, score = obj.get("name"), obj.get("score")
    return isinstance(name, str) and isinstance(score, int) and not isinstance(score, bool)
```
`results` = [True, False, False, True]. On the hidden replies of the check (`{"name": "Lea", "score": 3.5}`, `[1, 2, 3]`, `not json at all`, `{"name": "Max", "score": 10}`, `{"name": 7, "score": 2}`, empty text, `{"name": "Zed"}`) the function must give F, F, F, T, F, F, F.
- 1 point: the four results.
- 1 point: a function that parses with a real JSON parser inside try/except, checks the object type, the key names and the value types, and returns False instead of raising (0.5 if it passes the four tests by a hack such as string matching or checking for the characters of the replies, 0.5 if it raises on non-JSON text).
Typical errors: `eval` instead of `json.loads` (0 for the method); accepting a float score; not catching the exception for `''`; accepting the chatty reply by extracting the braces (the task says the whole reply must be JSON: 0.5 off); `True` accepted as score (not tested; no deduction).

### B4.7 Is the difference real? (2 points)
Model answer. p_A = 63/90 = 0.7, p_B = 54/90 = 0.6. `se_a` = √(0.7 · 0.3 / 90) = 0.0483; `se_b` = √(0.6 · 0.4 / 90) = 0.0516; `z` = 0.1 / √(0.0483² + 0.0516²) = 0.1 / 0.07071 = 1.41. Interpretation: a gap of 10 points is only about 1.4 standard errors of the difference, so with 90 questions it is not convincing (it could easily be noise); next: more questions or repeated runs on the same questions, read the wrong replies, test again.
Tolerance: 4 decimals for the standard errors, 2 for z.
- 1 point: `se_a`, `se_b` (0.25 each) and `z` (0.5).
- 1 point: `ANSWERS["B4.7"]`: verdict (not convincingly better, or within noise) tied to z = 1.41 (0.5), and a sensible next step (more or repeated questions, same questions, read the replies) (0.5). A student who says "A is better because 70% is more than 60%" without z gets 0.
Typical errors: dividing by n instead of taking the square root (0.0023); using n − 1; z with the sum of the two standard errors instead of the root of the sum of squares (1.00); z = 10 (the difference in points divided by 1); saying "z > 1 so significant".

**Task 4 total: 1 + 2 + 2 + 2 + 1 + 2 + 2 = 12.**

---

## Task 5: Synthesis (6 points, 2 each)

Marking: read the answer, give the points in the lines below. The model answers are in the reference notebook as well.

### B5.1 (Weeks 6 and 7), 2 points
Model answer. (a) Sub-word: 32,000 · 512 = 16,384,000 parameters; character-level: 100 · 512 = 51,200. (b) Sub-word: 100 · 100 = 10,000 attention scores per head per layer; character-level: 400 · 400 = 160,000, which is 16 times more (the cost grows with the square of the length: 4² = 16). (c) A cost of the large vocabulary: a big embedding (and output) table, and rare tokens whose vectors are seen few times and are poorly trained, or odd splits of rare words; a cost of the character-level design: sequences four times longer (16 times the attention cost, longer training and generation), and the model must learn to spell words from letters, and long-range links are harder.
- 1 point: (a) and (b) numbers (0.25 for the two table sizes, 0.25 for the two score counts, 0.25 for the ratio 16, 0.25 for showing the working).
- 1 point: (c), one valid cost each (0.5 each).
Typical errors: 32,000 + 512 (adding); forgetting the n² (ratio 4); a vague "it is slower".

### B5.2 (Weeks 7 and 8), 2 points
Model answer. (a) 0.9^999 = 1.9e-46: nothing reaches the first token, a vanishing gradient through time (a product of 999 factors below 1). Attention links token 1000 to token 1 directly, in one step, so the path between two tokens does not grow with their distance (and residual connections keep a short path to the input through the layers). (b) Positional information: the sinusoidal encoding PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(...), added to the token embeddings (or a learned position table), because attention itself is blind to the order. (c) 8 · 1000 · 1000 = 8,000,000 scores per layer for one document. The KV cache stores the keys and values of the tokens already processed, so that when generating, each new token computes only its own query, key and value and attends to the cache, instead of recomputing the whole sequence at every step.
- 1 point: (a) the number (about 2e-46; accept 1e-46 to 1e-45 from rounding of 0.9^999, or the statement "essentially zero" with the product written) 0.5 and the reason attention avoids it 0.5.
- 1 point: (b) positional encoding with its idea 0.5; (c) 8,000,000 and the cache 0.5.
Typical errors: 0.9 · 999 = 899; 8 · 1000 = 8,000 (forgetting n²); saying the cache saves memory (it costs memory and saves computation); "positional encoding makes the model faster".

### B5.3 (Weeks 8 and 9), 2 points
Model answer. (a) N = √(1e23 / 120) = 2.887e10 = 28.9 billion parameters; D = 20 N = 5.77e11 = 577 billion tokens. (b) The causal model learns from 2,047 targets (every position except the last); a BERT-style model with 15% masking from 0.15 · 2048 = 307, so the causal model gets about 6.7 times as many. (c) Partly right: with top-2 routing over 8 experts a token uses 2/8 of the expert (feed-forward) parameters, so the compute per token is that of a much smaller model; but all 100B parameters must be stored in memory, the attention, embeddings and router are always active so the active count is above 25B, and routing and load balancing add cost and engineering. (A good answer says "cost" means compute per token, not memory.)
- 1 point: (a) 28.9 and 577 billion (0.5) and (b) 2,047 and about 307 (0.5).
- 1 point: (c): active against total parameters, 0.5; what must still be stored or paid (all parameters in memory, always-active attention, load balancing), 0.5.
Typical errors: N = C/120 (no square root); D = N/20; 2,048 targets for the causal model (accept, 0.25 off); saying the MoE model needs only 25B of memory.

**Task 5 total: 2 + 2 + 2 = 6.**

---

## Totals and pass mark
Task 3: 3 + 2 + 2 + 2 + 2 + 3 + 2 = 16. Task 4: 1 + 2 + 2 + 2 + 1 + 2 + 2 = 12. Task 5: 2 + 2 + 2 = 6. Together with Tasks 1 and 2 (12 and 14 points) Part B is 60; with Part A (40) the Midterm is out of 100 and the pass mark is 70.

## Timing evidence
The reference solution runs the whole of Tasks 3 to 5 in about 8 seconds of CPU compute through `tools/test-notebooks.py --run` (the two 250-step trainings in B3.6 take about 5 seconds together); the unmodified starter runs without error and every check prints a cross. Writing time is the limit: the 30-minute boxes assume that the data, the tensors and the helper functions are given, as they are.
