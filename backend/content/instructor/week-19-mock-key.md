# Instructor key: Week 19 mock NCA-GENL exam (50 questions)

**Confidential: for instructors only.** The mock runs in the app (exam-mode quiz id week-19-mock): 50 multiple-choice questions, 1 point each, 60 minutes, closed book, pass mark 70 percent (35 of 50). It is written from the course content (Weeks 1 to 18), not from the official NCA-GENL blueprint, and it does not reproduce real exam questions. Calculation answers were recomputed with Python by build_mock.py, which asserts every correct value, that the four options of each question differ, the answer-position counts, the topic counts and the number of calculation questions.

Counts: topics 9/14/12/15 (Weeks 1–5 / 6–10 / 11–15 / 16–18); 24 calculation questions; level recall 17, apply 24, analyse 9; correct answers A 13, B 13, C 12, D 12.

| # | Topic | Answer | Type | Level |
|---|---|---|---|---|
| 1 | Weeks 1–5: learning, networks and CNNs | C | concept | recall |
| 2 | Weeks 1–5: learning, networks and CNNs | B | concept | recall |
| 3 | Weeks 1–5: learning, networks and CNNs | C | calculation | apply |
| 4 | Weeks 1–5: learning, networks and CNNs | D | situation | analyse |
| 5 | Weeks 1–5: learning, networks and CNNs | D | calculation | apply |
| 6 | Weeks 1–5: learning, networks and CNNs | B | concept | recall |
| 7 | Weeks 1–5: learning, networks and CNNs | A | calculation | apply |
| 8 | Weeks 1–5: learning, networks and CNNs | C | concept | recall |
| 9 | Weeks 1–5: learning, networks and CNNs | D | concept | recall |
| 10 | Weeks 6–10: language models, Transformers and scaling | C | calculation | apply |
| 11 | Weeks 6–10: language models, Transformers and scaling | B | calculation | apply |
| 12 | Weeks 6–10: language models, Transformers and scaling | B | calculation | apply |
| 13 | Weeks 6–10: language models, Transformers and scaling | A | concept | recall |
| 14 | Weeks 6–10: language models, Transformers and scaling | B | calculation | apply |
| 15 | Weeks 6–10: language models, Transformers and scaling | C | concept | recall |
| 16 | Weeks 6–10: language models, Transformers and scaling | A | calculation | apply |
| 17 | Weeks 6–10: language models, Transformers and scaling | B | concept | recall |
| 18 | Weeks 6–10: language models, Transformers and scaling | C | situation | analyse |
| 19 | Weeks 6–10: language models, Transformers and scaling | D | concept | recall |
| 20 | Weeks 6–10: language models, Transformers and scaling | C | calculation | apply |
| 21 | Weeks 6–10: language models, Transformers and scaling | B | situation | analyse |
| 22 | Weeks 6–10: language models, Transformers and scaling | D | calculation | apply |
| 23 | Weeks 6–10: language models, Transformers and scaling | A | concept | recall |
| 24 | Weeks 11–15: images, generation and fine-tuning | D | calculation | apply |
| 25 | Weeks 11–15: images, generation and fine-tuning | B | calculation | apply |
| 26 | Weeks 11–15: images, generation and fine-tuning | A | situation | analyse |
| 27 | Weeks 11–15: images, generation and fine-tuning | A | calculation | apply |
| 28 | Weeks 11–15: images, generation and fine-tuning | D | calculation | apply |
| 29 | Weeks 11–15: images, generation and fine-tuning | C | concept | recall |
| 30 | Weeks 11–15: images, generation and fine-tuning | B | situation | analyse |
| 31 | Weeks 11–15: images, generation and fine-tuning | A | calculation | apply |
| 32 | Weeks 11–15: images, generation and fine-tuning | C | concept | recall |
| 33 | Weeks 11–15: images, generation and fine-tuning | A | calculation | apply |
| 34 | Weeks 11–15: images, generation and fine-tuning | C | concept | recall |
| 35 | Weeks 11–15: images, generation and fine-tuning | B | situation | analyse |
| 36 | Weeks 16–18: RAG, agents, trust and deployment | D | calculation | apply |
| 37 | Weeks 16–18: RAG, agents, trust and deployment | D | concept | recall |
| 38 | Weeks 16–18: RAG, agents, trust and deployment | A | calculation | apply |
| 39 | Weeks 16–18: RAG, agents, trust and deployment | A | calculation | apply |
| 40 | Weeks 16–18: RAG, agents, trust and deployment | C | situation | analyse |
| 41 | Weeks 16–18: RAG, agents, trust and deployment | B | concept | recall |
| 42 | Weeks 16–18: RAG, agents, trust and deployment | D | situation | analyse |
| 43 | Weeks 16–18: RAG, agents, trust and deployment | D | calculation | apply |
| 44 | Weeks 16–18: RAG, agents, trust and deployment | A | concept | recall |
| 45 | Weeks 16–18: RAG, agents, trust and deployment | C | calculation | apply |
| 46 | Weeks 16–18: RAG, agents, trust and deployment | A | calculation | apply |
| 47 | Weeks 16–18: RAG, agents, trust and deployment | D | situation | analyse |
| 48 | Weeks 16–18: RAG, agents, trust and deployment | A | concept | recall |
| 49 | Weeks 16–18: RAG, agents, trust and deployment | B | calculation | apply |
| 50 | Weeks 16–18: RAG, agents, trust and deployment | B | calculation | apply |

## Working for calculation questions

**Q3** (Weeks 1–5: learning, networks and CNNs), correct C: 5.6

- Working: dL/dw = 4(w − 4) = 8. New w = 6 − 0.05·8 = 5.6. Distractors: 6 − 0.05·4 = 5.8 (gradient 2(w−4)); 6 − 8 = −2 (no learning rate); 6 + 0.4 = 6.4 (sign).
- Distractor errors: A 5.8 (used 2(w − 4) as the gradient and dropped the constant 2); B -2 (forgot to multiply the gradient by the learning rate); D 6.4 (added the step instead of subtracting it)
- Options: A 5.8; B -2; C 5.6; D 6.4

**Q5** (Weeks 1–5: learning, networks and CNNs), correct D: 9

- Working: z = 0.5·3 + 0.5 = 2; a = 2; dL/da = 2(2 − 0.5) = 3; da/dz = 1; dz/dw = x = 3; dL/dw = 3·1·3 = 9. Distractors: 3 (no x); 1.5·3 = 4.5 (no factor 2); 3·0.5 = 1.5 (w instead of x).
- Distractor errors: A 3 (forgot the factor dz/dw = x); B 4.5 (forgot the 2 from the derivative of the square); C 1.5 (used w instead of x for dz/dw)
- Options: A 3; B 4.5; C 1.5; D 9

**Q7** (Weeks 1–5: learning, networks and CNNs), correct A: 4,096

- Working: conv: (64+4−5)//2+1 = 32; pool: 16; flatten 16·16·16 = 4096. Distractors: 32·32·16 = 16384 (no pool); 16·16·3 = 768 (3 channels); 16·16 = 256 (no channels).
- Distractor errors: B 16,384 (skipped the pooling layer); C 768 (used the 3 input channels instead of the 16 filters); D 256 (forgot the channel dimension)
- Options: A 4,096; B 16,384; C 768; D 256

**Q10** (Weeks 6–10: language models, Transformers and scaling), correct C: a + p, 14 occurrences

- Working: Pair counts: (a,p) = 4+3+5+2 = 14; (l,a) = 4+3 = 7; (t,a) = 5+2 = 7; (p,s) = (s,e) = 3; (p,e) = (e,s) = 2. Distractors: 4 distinct words; l+a 7; 14 − 2 = 12.
- Distractor errors: A a + p, 4 occurrences (counted the 4 distinct words, not their frequencies); B l + a, 7 occurrences (a real pair, but it occurs only in lap and lapse (4 + 3)); D a + p, 12 occurrences (left out the 2 occurrences in tapes)
- Options: A a + p, 4 occurrences; B l + a, 7 occurrences; C a + p, 14 occurrences; D a + p, 12 occurrences

**Q11** (Weeks 6–10: language models, Transformers and scaling), correct B: 0.44

- Working: u·v = 2 − 2 + 4 = 4; |u| = |v| = 3; cos = 4/9 = 0.444. Distractors: 4 (dot); 8/9 = 0.889 (no sign); 4/3 = 1.333 (one norm).
- Distractor errors: A 4 (stopped at the dot product and did not divide by the lengths); C 0.89 (dropped the minus sign: 2 + 2 + 4 = 8, then 8/9); D 1.33 (divided by only one length: 4/3 (a cosine can never exceed 1))
- Options: A 4; B 0.44; C 0.89; D 1.33

**Q12** (Weeks 6–10: language models, Transformers and scaling), correct B: 4

- Working: prod = 1/256; PPL = 256^(1/4) = 4. Distractors: 1/mean(p) = 1/0.28125 = 3.56; 256; 256^(−1/4) = 0.25.
- Distractor errors: A 3.56 (inverted the arithmetic mean of the probabilities instead of the geometric mean); C 256 (inverted the product but did not take the 4th root); D 0.25 (took the geometric mean of the probabilities but forgot to invert it)
- Options: A 3.56; B 4; C 256; D 0.25

**Q14** (Weeks 6–10: language models, Transformers and scaling), correct B: 0.73

- Working: scaled = [2, 1]; w1 = e²/(e²+e) = 0.7311. Distractors: softmax([4,2])[0] = 0.881; 4/6 = 0.667; w2 = 0.269.
- Distractor errors: A 0.88 (forgot to scale by √d_k); C 0.67 (divided by the sum of the scores instead of using a softmax); D 0.27 (reported the weight of the second key)
- Options: A 0.88; B 0.73; C 0.67; D 0.27

**Q16** (Weeks 6–10: language models, Transformers and scaling), correct A: 786,432

- Working: attn = 4·65,536 = 262,144; ffn = 2·256·1024 = 524,288; sum 786,432 = 12·256². Distractors: 3·65,536+524,288 = 720,896; 262,144+131,072 = 393,216; 262,144+262,144 = 524,288.
- Distractor errors: B 720,896 (counted only three attention matrices and left out Wo); C 393,216 (took d_ff = d_model (256) in the feed-forward network); D 524,288 (counted only one of the two feed-forward matrices)
- Options: A 786,432; B 720,896; C 393,216; D 524,288

**Q20** (Weeks 6–10: language models, Transformers and scaling), correct C: 1.08 × 10²¹ FLOPs

- Working: D = 6e10; C = 6·3e9·6e10 = 1.08e21. Distractors: N·D = 1.8e20; 2ND = 3.6e20; D = 200N = 6e11 gives 1.08e22.
- Distractor errors: A 1.80 × 10²⁰ FLOPs (forgot the factor 6 (used N·D)); B 3.60 × 10²⁰ FLOPs (used 2·N·D, the cost of a forward pass only); D 1.08 × 10²² FLOPs (used 200 tokens per parameter instead of 20)
- Options: A 1.80 × 10²⁰ FLOPs; B 3.60 × 10²⁰ FLOPs; C 1.08 × 10²¹ FLOPs; D 1.08 × 10²² FLOPs

**Q22** (Weeks 6–10: language models, Transformers and scaling), correct D: 0.87

- Working: [4,2,0] → e = [54.598, 7.389, 1]; top = 0.8668. Distractors: T=1 0.665; logits·T = [1,0.5,0] 0.506; /0.25 = [8,4,0] 0.982.
- Distractor errors: A 0.67 (ignored the temperature (T = 1)); B 0.51 (multiplied the logits by T instead of dividing); C 0.98 (divided by T² (0.25) instead of T)
- Options: A 0.67; B 0.51; C 0.98; D 0.87

**Q24** (Weeks 11–15: images, generation and fine-tuning), correct D: 50

- Working: 224/32 = 7; 7·7 = 49; +1 CLS = 50. Distractors: 49; 7; 224²/32 = 1568.
- Distractor errors: A 49 (forgot the CLS token); B 7 (counted patches along one side only); C 1,568 (divided the number of pixels by the patch side, not by the patch area)
- Options: A 49; B 7; C 1,568; D 50

**Q25** (Weeks 11–15: images, generation and fine-tuning), correct B: 16,256

- Working: 128² = 16384; minus the 128 diagonal = 16256. Distractors: 128·127/2 = 8128; 127; 16384.
- Distractor errors: A 8,128 (counted unordered pairs; the matrix has a row and a column for every ordered pair); C 127 (counted the negatives for a single image only); D 16,384 (counted every entry, including the 128 correct pairs on the diagonal)
- Options: A 8,128; B 16,256; C 127; D 16,384

**Q27** (Weeks 11–15: images, generation and fine-tuning), correct A: 14

- Working: (7−1)·2 − 2·1 + 3 + 1 = 12 − 2 + 4 = 14. Distractors: 13; 16; 12.
- Distractor errors: B 13 (left out the output padding); C 16 (ignored the padding (−2p)); D 12 (kept only (n − 1)·stride)
- Options: A 14; B 13; C 16; D 12

**Q28** (Weeks 11–15: images, generation and fine-tuning), correct D: 0.4

- Working: abar = 0.36; sqrt = 0.6; sqrt(0.64) = 0.8; x3 = 1.2 − 0.8 = 0.4. Distractors: 0.72 − 0.64 = 0.08; 0.707·2 − 0.707 = 0.71; 1.2 + 0.8 = 2.0.
- Distractor errors: A 0.08 (used ᾱ and 1 − ᾱ without square roots); B 0.71 (used only the last α (0.5) instead of the product of all three); C 2.0 (flipped the sign of the noise)
- Options: A 0.08; B 0.71; C 2.0; D 0.4

**Q31** (Weeks 11–15: images, generation and fine-tuning), correct A: 0.82

- Working: σ(1.5) = 0.8176. Distractors: 1 − p = 0.182; σ(2.7) = 0.937; 2.1/2.7 = 0.778.
- Distractor errors: B 0.18 (gave the probability of the rejected answer winning); C 0.94 (used the sum of the rewards instead of the difference); D 0.78 (normalised the two rewards as a ratio instead of using the sigmoid of the gap)
- Options: A 0.82; B 0.18; C 0.94; D 0.78

**Q33** (Weeks 11–15: images, generation and fine-tuning), correct A: 8,388,608

- Working: 2·4096·16 = 131,072 per adapter; ×64 = 8,388,608. Distractors: half = 4,194,304; four matrices = 16,777,216; full q,v = 1,073,741,824.
- Distractor errors: B 4,194,304 (counted only matrix A of each adapter); C 16,777,216 (adapted four matrices per layer (q, k, v, o) instead of two); D 1,073,741,824 (counted the full 4,096×4,096 matrices, which LoRA leaves frozen)
- Options: A 8,388,608; B 4,194,304; C 16,777,216; D 1,073,741,824

**Q36** (Weeks 16–18: RAG, agents, trust and deployment), correct D: 0.763

- Working: 0.95·0.90 = 0.855; ·0.92 = 0.7866; ·0.97 = 0.763. Distractors: min 0.900; mean 0.935; 1 − 0.26 = 0.740.
- Distractor errors: A 0.900 (took the weakest step); B 0.935 (averaged the four success rates); C 0.740 (subtracted the sum of the failure rates from 1)
- Options: A 0.900; B 0.935; C 0.740; D 0.763

**Q38** (Weeks 16–18: RAG, agents, trust and deployment), correct A: 7

- Working: (1000−50)/(200−50) = 6.33 → 7. Distractors: 1000/200 = 5; floor 6; 1000/250 = 4.
- Distractor errors: B 5 (ignored the overlap (1,000 / 200)); C 6 (rounded down and lost the last, shorter chunk); D 4 (added the overlap to the chunk size instead of subtracting it)
- Options: A 7; B 5; C 6; D 4

**Q39** (Weeks 16–18: RAG, agents, trust and deployment), correct A: Precision 0.60, recall 0.75

- Working: P = 6/10 = 0.60; R = 6/8 = 0.75. Distractors: swapped; 6/10 twice; 6/8 twice.
- Distractor errors: B Precision 0.75, recall 0.60 (swapped the two denominators); C Precision 0.60, recall 0.60 (divided by the 10 returned passages both times); D Precision 0.75, recall 0.75 (divided by the 8 relevant passages both times)
- Options: A Precision 0.60, recall 0.75; B Precision 0.75, recall 0.60; C Precision 0.60, recall 0.60; D Precision 0.75, recall 0.75

**Q43** (Weeks 16–18: RAG, agents, trust and deployment), correct D: 20

- Working: 0.2·0.1·0.25 = 0.005; ×4000 = 20. Distractors: 0.55·4000 = 2200; 0.1833·4000 = 733; 0.1·4000 = 400.
- Distractor errors: A 2,200 (added the three pass rates (0.55)); B about 733 (used the average pass rate (0.183)); C 400 (counted only the best rail (0.10))
- Options: A 2,200; B about 733; C 400; D 20

**Q45** (Weeks 16–18: RAG, agents, trust and deployment), correct C: 0.24

- Working: rA = 0.60; rB = 0.36; DPD = 0.24. Distractors: 0.36/0.60 = 0.60; 84 − 45 = 39; 84/265 − 45/265 = 0.147.
- Distractor errors: A 0.60 (computed the ratio B/A instead of the difference); B 39 (subtracted the counts of selected people, ignoring group sizes); D 0.15 (divided both counts by the combined total of 265)
- Options: A 0.60; B 39; C 0.24; D 0.15

**Q46** (Weeks 16–18: RAG, agents, trust and deployment), correct A: (0.25, 1.00, 0.00)

- Working: x + 0.2·sign(g) = (0.25, 1.00, −0.10) → clip → (0.25, 1.00, 0.00). Distractors: unclipped; x + 0.2·g = (−0.15, 0.84, 0.098) → clip (0.00, 0.84, 0.10); x − 0.2·sign = (0.65, 0.60, 0.30).
- Distractor errors: B (0.25, 1.00, -0.10) (forgot to clip the third pixel back into 0 to 1); C (0.00, 0.84, 0.10) (multiplied ε by the gradient itself instead of only its sign); D (0.65, 0.60, 0.30) (stepped against the gradient, which lowers the loss)
- Options: A (0.25, 1.00, 0.00); B (0.25, 1.00, -0.10); C (0.00, 0.84, 0.10); D (0.65, 0.60, 0.30)

**Q49** (Weeks 16–18: RAG, agents, trust and deployment), correct B: Stored integer 17, read back as 0.51

- Working: scale = 0.03; 0.5/0.03 = 16.67 → 17; 17·0.03 = 0.51. Distractors: floor 16 (0.48); round(0.5·127) = 64 (1.92); round(0.5/3.81) = 0 (0.00).
- Distractor errors: A Stored integer 16, read back as 0.48 (truncated 16.67 to 16 instead of rounding to the nearest integer); C Stored integer 64, read back as 1.92 (multiplied by 127 without dividing by the largest value); D Stored integer 0, read back as 0.00 (divided by 3.81 but forgot to multiply by 127 (everything rounds to 0))
- Options: A Stored integer 16, read back as 0.48; B Stored integer 17, read back as 0.51; C Stored integer 64, read back as 1.92; D Stored integer 0, read back as 0.00

**Q50** (Weeks 16–18: RAG, agents, trust and deployment), correct B: 1.57 GB

- Working: 2·24·1000·2048·2·8 = 1,572,864,000 B = 1.57 GB. Distractors: 0.79 GB (no K and V); 0.20 GB (batch 1); 3.15 GB (4-byte values).
- Distractor errors: A 0.79 GB (left out the factor 2 for keys and values); C 0.20 GB (computed it for one request, not 8); D 3.15 GB (used 4 bytes per value (FP32) instead of 2)
- Options: A 0.79 GB; B 1.57 GB; C 0.20 GB; D 3.15 GB

## Notes for marking and use

- Pass mark: 35 of 50. The app shows a breakdown by the four topic strings; use the weakest row to target revision.
- Situation questions (analyse) have one best answer; the wrong options each contain the mistake named in the explanation shown to the student.
- Do not describe this mock as the official exam or as a prediction of it. NCA-GENL is issued by NVIDIA after its own proctored exam; Supervaise cannot issue or guarantee it.
