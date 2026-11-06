# Week 5 — Prelim Exam: instructor answer key and marking scheme

**Confidential: for instructors only.** The student notebook (notebooks/week-05.ipynb) is a public download and contains no answers; its self-checks compare against hashed fingerprints. Part A runs in the app (exam-mode quiz id week-05-prelim, 40 questions, 1 point each). Part B (60 points) is marked by the instructor from the submitted notebook.

---

# Instructor key: Part A (written exam), Week 5 Prelim

Not for students. 40 questions, 1 point each. Calculation answers were recomputed with Python (build_exam.py asserts every correct value and that each distractor differs).

| # | Topic | Answer | Type | Level |
|---|---|---|---|---|
| 1 | Week 1: generative AI | C | concept | recall |
| 2 | Week 1: generative AI | C | calculation | apply |
| 3 | Week 1: generative AI | D | calculation | apply |
| 4 | Week 1: generative AI | C | calculation | apply |
| 5 | Week 1: generative AI | A | concept | analyse |
| 6 | Week 1: generative AI | B | calculation | apply |
| 7 | Week 1: generative AI | B | concept | recall |
| 8 | Week 1: generative AI | C | concept | analyse |
| 9 | Week 2: supervised learning | A | concept | recall |
| 10 | Week 2: supervised learning | C | calculation | apply |
| 11 | Week 2: supervised learning | B | calculation | apply |
| 12 | Week 2: supervised learning | D | calculation | apply |
| 13 | Week 2: supervised learning | A | calculation | apply |
| 14 | Week 2: supervised learning | D | calculation | analyse |
| 15 | Week 2: supervised learning | D | calculation | apply |
| 16 | Week 2: supervised learning | A | concept | analyse |
| 17 | Week 2: supervised learning | D | concept | analyse |
| 18 | Week 2: supervised learning | A | concept | recall |
| 19 | Week 3: deep learning and backprop | C | concept | recall |
| 20 | Week 3: deep learning and backprop | A | calculation | apply |
| 21 | Week 3: deep learning and backprop | D | calculation | apply |
| 22 | Week 3: deep learning and backprop | A | calculation | apply |
| 23 | Week 3: deep learning and backprop | B | calculation | apply |
| 24 | Week 3: deep learning and backprop | D | calculation | apply |
| 25 | Week 3: deep learning and backprop | B | calculation | apply |
| 26 | Week 3: deep learning and backprop | C | concept | recall |
| 27 | Week 3: deep learning and backprop | B | calculation | apply |
| 28 | Week 3: deep learning and backprop | A | concept | recall |
| 29 | Week 3: deep learning and backprop | D | concept | analyse |
| 30 | Week 4: convolutional networks | B | calculation | apply |
| 31 | Week 4: convolutional networks | C | concept | recall |
| 32 | Week 4: convolutional networks | A | calculation | apply |
| 33 | Week 4: convolutional networks | B | calculation | apply |
| 34 | Week 4: convolutional networks | C | calculation | apply |
| 35 | Week 4: convolutional networks | A | concept | recall |
| 36 | Week 4: convolutional networks | C | calculation | apply |
| 37 | Week 4: convolutional networks | B | calculation | apply |
| 38 | Week 4: convolutional networks | D | calculation | apply |
| 39 | Week 4: convolutional networks | D | calculation | apply |
| 40 | Week 4: convolutional networks | B | concept | analyse |

Level counts: recall 9, apply 24, analyse 7

## Working for calculation questions

**Q2** (Week 1: generative AI), correct C: 0.004

- Working: 0.02 x 0.5 x 0.4 = 0.0040. Distractors: sum 0.92; mean 0.3067; first two only 0.010.
- Options: A 0.01; B 0.307; C 0.004; D 0.92

**Q3** (Week 1: generative AI), correct D: 8

- Working: P = 0.001953125; P^(-1/3) = 8.000000. Distractors: 1/P = 512; geometric mean prob = 0.1250; 1/arithmetic mean = 4.57.
- Options: A 512; B 0.125; C 4.57; D 8

**Q4** (Week 1: generative AI), correct C: 0.63

- Working: exp(1.5)=4.4817, exp(0.5)=1.6487, exp(0)=1; total 7.1304; top = 0.6285. T=1 gives 0.8438; 3/(3+1+0)=0.75; 1/3=0.333 (T to infinity).
- Options: A 0.84; B 0.75; C 0.63; D 0.33

**Q6** (Week 1: generative AI), correct B: −0.2

- Working: sqrt(0.64)=0.8, sqrt(0.36)=0.6; 0.8*0.5 + 0.6*(-1) = -0.20. No roots: 0.64*0.5+0.36*(-1) = -0.04. eps sign lost: 0.4+0.6=1.0; sign of result lost 0.2.
- Options: A 0.2; B −0.2; C −0.04; D 1.0

**Q10** (Week 2: supervised learning), correct C: 2.75

- Working: squared errors 1,1,0,9 sum 11; MSE = 11/4 = 2.75. Distractors: sum 11; MAE=(1+1+0+3)/4=1.25; RMSE=1.658.
- Options: A 11; B 1.25; C 2.75; D 1.66

**Q11** (Week 2: supervised learning), correct B: 1.61

- Working: -ln(0.2) = 1.6094. Distractors: 1-0.2=0.8; -log10(0.2)=0.699; -ln(0.3)=1.204.
- Options: A 0.80; B 1.61; C 0.70; D 1.20

**Q12** (Week 2: supervised learning), correct D: 0.85

- Working: z = 1.70; sigmoid = 0.8455. Distractors: z itself 1.70; no bias sigmoid(3.2)=0.9608; sign error sigmoid(4.7)=0.9910.
- Options: A 1.70; B 0.96; C 0.99; D 0.85

**Q13** (Week 2: supervised learning), correct A: 203

- Working: 104+72+27 = 203. Distractors: weights only 184; 12+8+8+3 = 31; 203+12 = 215.
- Options: A 203; B 184; C 31; D 215

**Q14** (Week 2: supervised learning), correct D: (5, 3)

- Working: W2 @ W1 = [[5.0, 3.0]]. Check x=(1,2): W1x=(2,3), W2(W1x)=11; (5,3).x = 11. Distractors: swapped (3,5); W2 @ W1.T = [[2.0, 4.0]]; last layer only (1,3).
- Options: A (3, 5); B (2, 4); C (1, 3); D (5, 3)

**Q15** (Week 2: supervised learning), correct D: 0.87

- Working: 0.8 - 0.2*(-0.35) = 0.87. Distractors: sign error 0.8+(-0.07)=0.73; no eta 0.8-(-0.35)=1.15; no eta wrong sign 0.8+(-0.35)=0.45.
- Options: A 0.73; B 1.15; C 0.45; D 0.87

**Q20** (Week 3: deep learning and backprop), correct A: −0.5

- Working: z1 = [-0.5, 1.0]; a1 = [0.0, 1.0]; y = -0.5. Distractors: no ReLU -1.5; no b2 -1.0; no b1 5.5.
- Options: A −0.5; B −1.5; C −1.0; D 5.5

**Q21** (Week 3: deep learning and backprop), correct D: 6

- Working: z=2.0, a=2.0; dL/da=2(a-y)=2; da/dz=1; dz/dw=x=3; product 6.0. Distractors: (a-y)x=3; 2(a-y)=2; 2*a*x=12.
- Options: A 3; B 2; C 12; D 6

**Q22** (Week 3: deep learning and backprop), correct A: −0.032

- Working: (0.8-1)*0.8*0.2 = -0.032. Distractors: a-y=-0.2 (cross-entropy shortcut, not this loss); slope only 0.16 (sign -0.16 is wrong too); sign error 0.032.
- Options: A −0.032; B −0.2; C −0.16; D 0.032

**Q23** (Week 3: deep learning and backprop), correct B: (1, 0)

- Working: W^T delta = [1.0, -0.5]; relu' = [1, 0]; delta = [1.0, -0.0].
- Options: A (1, −0.5); B (1, 0); C (0.5, 0); D (0, −0.5)

**Q24** (Week 3: deep learning and backprop), correct D: −6

- Working: outer(delta, a) = [[1.5, 0.5, 0.0], [-6.0, -2.0, -0.0]]; entry [2nd unit, 1st input] = -6.0. Distractors: dW[1st unit,1st input]=1.5; delta alone -2; dW[:,3rd]=0.
- Options: A 1.5; B −2; C 0; D −6

**Q25** (Week 3: deep learning and backprop), correct B: About 1.5 × 10⁻⁵

- Working: 0.25^8 = 1.526e-05. Distractors: 8*0.25=2; 0.25^4=3.9062e-03; 0.25/8=0.03125.
- Options: A About 2; B About 1.5 × 10⁻⁵; C About 3.9 × 10⁻³; D About 0.031

**Q27** (Week 3: deep learning and backprop), correct B: (−1.8, 2.4)

- Working: norm = 15.0; factor = 0.2; clipped = [-1.8, 2.4000000000000004] (norm 3.0). Distractors: componentwise clip (-3,3); unit vector (-0.6,0.8); unchanged.
- Options: A (−3, 3); B (−1.8, 2.4); C (−0.6, 0.8); D (−9, 12)

**Q30** (Week 4: convolutional networks), correct B: 1,400

- Working: conv: 3*3*3*50+50 = 1400. Distractors: no bias 1350; no depth 450; dense 3072*50+50 = 153650.
- Options: A 1,350; B 1,400; C 450; D 153,650

**Q32** (Week 4: convolutional networks), correct A: 1

- Working: sum = -3.0; +b = 1.0; ReLU = 1.0. Flipped kernel: 3.0+4 = 7.0. No bias: ReLU(-3.0) = 0. Raw sum = -3.
- Options: A 1; B 7; C 0; D −3

**Q33** (Week 4: convolutional networks), correct B: 32

- Working: floor((64+4-5)/2)+1 = floor(31.5)+1 = 32. Distractors: 31 (no +1); 33 (ceil+1); stride 1 gives 64.
- Options: A 31; B 32; C 33; D 64

**Q34** (Week 4: convolutional networks), correct C: 7,146

- Working: conv1 208; conv2 1168; flatten 576; dense 5770; total 7146. Distractors: no biases 7112; no second pool (flatten 2304) 24426; conv only 1376.
- Options: A 7,112; B 24,426; C 7,146; D 1,376

**Q36** (Week 4: convolutional networks), correct C: 10

- Working: rf trace: conv1 3 (j1); conv2 5 (j1); pool 6 (j2); conv3 6+2*2 = 10 (j2). Distractors: ignoring jump 3+2+1+2 = 8; pool adds nothing 5+4 = 9; k*jump 6+3*2 = 12.
- Options: A 8; B 9; C 10; D 12

**Q37** (Week 4: convolutional networks), correct B: 1

- Working: dK[0,1] = 1.0. Others: dK[0,0]=0.0, dK[1,0]=3.0, dK[1,1]=3.0; first window only = 2.0.
- Options: A 0; B 1; C 3; D 2

**Q38** (Week 4: convolutional networks), correct D: 3

- Working: dx[1,1] = 3.0. Distractors: unflipped sum(G*K) = -3.0; transposed index 1.0; last window only -2.0.
- Options: A −3; B 1; C −2; D 3

**Q39** (Week 4: convolutional networks), correct D: (0, 0, 0, 1.5)

- Working: a = [0.0, 2.0, 0.5, 3.0]; argmax cell 4; gradient [0.0, 0.0, 0.0, 1.5]. Distractors: avg-pool style [0.0, 0.375, 0.375, 0.375] (1.5/4 = 0.375); wrong cell; all active cells.
- Options: A (0, 0.375, 0.375, 0.375); B (0, 1.5, 0, 0); C (0, 1.5, 1.5, 1.5); D (0, 0, 0, 1.5)

---

# Instructor key: Week 5 Prelim practical exam (Part B), Tasks 1 and 2 (22 points)

Not for students. All numbers were computed by running the reference solution (seed 42, CPU). Reference notebook: `w5/testA/week-05.py`; every self-check prints a tick for it. The self-checks accept the value rounded to the stated decimals (plus or minus 0.4 of the last digit). Marks are in whole points unless a half point is shown. Rule for all code questions: method correct but wrong number earns half the numeric point; a hard-coded number with no working earns at most half.

## Task 1. Generative AI and language models (10 points)

### B1.1 Softmax with temperature (2)
- Model answer: p(hot) at T = 1 is 0.6684 (all four: 0.6684, 0.2717, 0.0449, 0.0150); at T = 0.5 it is 0.8545 (all four: 0.8545, 0.1412, 0.0039, 0.0004).
- Tolerance: 3 decimals (0.668, 0.854 or 0.855).
- Marking: 1 for `p_hot_T1`, 1 for `p_hot_T05`. A student who forgets the max-subtraction is still right. Dividing the logits by T after exp (instead of before) gives 0.6684 and a wrong second value: half of the second point if the formula is otherwise clear.
- The written sentence is not worth separate points; read it as support: expected idea is that the distribution becomes sharper (more peaked, less random). If it says the opposite (flatter), deduct half a point from the part of the numeric points it contradicts.

### B1.2 Top-k and top-p (2)
- Model answer: `n_kept` = 3 (cumulative probabilities 0.668, 0.940, 0.985, 1.0; 0.940 is below 0.95, so the third word is needed). `p_hot_topk2` = 0.6684 / (0.6684 + 0.2717) = 0.7109.
- Tolerance: `n_kept` exact; `p_hot_topk2` 0.711 (0.710 to 0.712).
- Marking: 1 + 1. Common errors: n_kept = 2 (stopped when the sum is below 0.95, should include the word that crosses it): 0, but give half a point for a clear explanation of the rule; forgetting to renormalise gives 0.668 for the second: 0.
- A working `filter_top` that returns a normalised vector is evidence for method credit.

### B1.3 Chain rule and perplexity (2)
- Model answer: ln p = ln 0.12 + ln 0.30 + ln 0.25 + ln 0.08 + ln 0.50 = −7.929 (probability 3.6e-4); perplexity = exp(7.929 / 5) = 4.884; the most surprising word is "every" (p = 0.08).
- Tolerance: log_prob −7.93 (2 decimals); perplexity 4.88; word exact.
- Marking: 0.5 + 1 + 0.5. Typical errors: log10 instead of ln (log_prob −3.444): 0 for log_prob, but accept the perplexity if it is computed as 10 to the power of the negative mean (also 4.88); dividing by 5 without the minus sign (0.2): 0; product to the power 1/5 (not inverted) gives 0.2047: 0, half for the right idea.

### B1.4 Bigram model (2)
- Model answer: counts after "makes": rice 2, eggs 1, so P(rice | makes) = 2/3 = 0.6667. Test sentence ". the cook makes eggs ." has the five transition probabilities P(the | .) = 1, P(cook | the) = 3/7 = 0.4286, P(makes | cook) = 1, P(eggs | makes) = 1/3, P(. | eggs) = 1. The product is 0.1429 (1/7), the perplexity is 7^(1/5) = 1.4758.
- Tolerance: 3 decimals (0.667; 1.476).
- Marking: 1 + 1. `bigram_counts` and `p_next` must be real code (dictionary of counts; division by total after the previous token). Half a point for the perplexity if the transition probabilities are right but the exponent or root is wrong. Using only four transitions (dropping the leading full stop) gives perplexity 1.6266: half.
- Note for markers: "the" is followed by canteen twice, cook three times and students twice (total 7).

### B1.5 Fluent is not the same as true (2)
- Model answer (1 + 1):
  - (a) 1 point. The model chooses a likely next token one at a time from learned patterns; it is trained to produce plausible text, not to check facts, and has no database of real books. At temperature 0.9 it samples, so it can pick a title that sounds right but does not exist (a hallucination). Full mark for: fluent is not the same as true, plus either next-token prediction or sampling. Half for "the AI made a mistake" without the mechanism.
  - (b) 1 point. Two sensible actions, 0.5 each: look each source up in a library, catalogue or official site; ask the chatbot or tool for links and open them; cross-check with a second reliable source; tell the teacher that AI was used and verify before citing. "Ask the same chatbot whether it is sure" earns 0 (it can repeat the error).
- Wrong or missing answer: 0. Under 30 words is still marked on content.

## Task 2. Supervised learning, neurons and generalisation (12 points)

### B2.1 One neuron (2)
- Model answer: z = 0.7·0.6 + 0.2·(−1.5) + (−0.5)·2.0 + 0.3 = 0.42 − 0.30 − 1.00 + 0.30 = −0.58. a_sigmoid = 1 / (1 + e^0.58) = 0.3589. (ReLU would give 0.)
- Tolerance: z −0.580; a 0.359 (0.358 to 0.360).
- Marking: 1 + 1. If z is wrong but the sigmoid is applied correctly to the student's own z: award the second point.

### B2.2 Counting parameters (2)
- Model answer: (a) 6·10 + 10 + 10·8 + 8 + 8·3 + 3 = 70 + 88 + 27 = 185. (b) 4·5 + 5 + 5·2 + 2 = 37 parameters; the single equivalent layer 4 to 2 has 4·2 + 2 = 10. Sentence: without an activation, stacked layers collapse to one linear layer (the product of the weight matrices), so depth adds parameters but no extra power.
- Marking: 1 for (a) exact 185. 1 for (b): 37 and 10 both correct (0.5 if only one, or if the sentence is missing or wrong). Forgetting biases gives 164 and 28 and 8: 0.5 for the whole of (a) if the weights-only count is shown and labelled; otherwise 0.

### B2.3 Two losses (2)
- Model answer: squared errors 0.16, 0.16, 0.25, 0.64 sum to 1.21, so MSE = 0.3025. Softmax of (1.2, −0.3, 0.5): sum of exponentials 3.3201 + 0.7408 + 1.6487 = 5.7096; p(class 2) = 1.6487 / 5.7096 = 0.2888; cross-entropy −ln 0.2888 = 1.2422.
- Tolerance: mse 0.3025 (4 decimals; 0.302 to 0.303 accepted); ce 1.242 (1.241 to 1.243).
- Marking: 1 + 1. Using the logit as a probability or taking ln of the logit: 0. Using log base 10 (0.5394): 0 but half for the right structure.

### B2.4 Gradient descent (2)
- Model answer: the gradient at w = 0 is (2/3)·(−1.8 − 9.0 − 17.1) = −18.6, so w1 = 0.05·18.6 = 0.93. Continuing: w2 = 1.4260, w3 = 1.6905, w4 = 1.8316 (1.83162). The loss at w4 is 0.2478 (0.24775).
- Tolerance: w_final 1.832 (1.831 to 1.833); loss_final 0.248 (0.247 to 0.249).
- Marking: 1 + 1. One or three steps (an off-by-one in the number of steps) gives a wrong w: half for correct update rule shown in a loop. Plus sign in the update (moving uphill): 0 with half for the right gradient. The loss is computed with the student's own final w: if w is wrong but loss is consistent with it, give the loss point.
- Intermediate values (computed): w after 1 step 0.9300, 2 steps 1.4260, 3 steps 1.6905, 4 steps 1.8316.

### B2.5 Why a hidden layer? (2)
- Reference numbers (data `make_xor(n=300, noise=0.12, seed=7)`, 200 training / 100 validation, 300 epochs, Adam, lr 0.05, seed 42): validation accuracy of the linear neuron 0.36, of the tanh network (2 hidden layers of 8, Xavier) 0.92. Final training losses: 0.68 and 0.12. Other seeds give 0.85 to 0.90 for the hidden network.
- Tolerance: the linear neuron must be clearly poor (below about 0.75 and often near or below 0.5; XOR cannot be split by one line); the hidden network clearly better (above 0.8).
- Marking: 1 for both accuracies obtained by really training with `train_model` and reading `"val_acc"` (training accuracy instead of validation: 0.5). 1 for the explanation: a single neuron draws one straight line (linear boundary) and XOR needs two or a bent boundary; the hidden layer with a non-linear activation can combine lines. Half for "the hidden net is better because it is bigger" without lines and non-linearity.

### B2.6 Overfitting (2)
- Reference numbers (`make_moons(n=40, noise=0.40, seed=3)` train, `make_moons(n=400, noise=0.40, seed=4)` test, 800 epochs, Adam, lr 0.01): `big` train accuracy 1.000, test 0.830, gap 0.170; `small` train 0.875, test 0.865, gap 0.010.
- Tolerance: `big` gap between about 0.10 and 0.25 if the student changes nothing else; accept any run in which gap_big is clearly larger than gap_small and equals train accuracy minus validation accuracy returned by `train_model`.
- Marking: 1 for the two gaps computed from the returned `"acc"` and `"val_acc"` (0.5 if one is right). 1 for the written answer: `big` overfits (memorises the 40 points, training accuracy 1.0, test accuracy lower, large gap), and one valid remedy: more data, a smaller network, early stopping, weight decay/dropout (the last two beyond the labs: accept if correctly described), or lower learning rate / fewer epochs. Half if the diagnosis is right but no remedy, or a remedy without the diagnosis. Saying the small network underfits is not correct for these numbers (train 0.875 and test 0.865 are close): 0 for that sentence.

## Total
| Question | Points |
|---|---|
| B1.1 | 2 |
| B1.2 | 2 |
| B1.3 | 2 |
| B1.4 | 2 |
| B1.5 | 2 |
| Task 1 | 10 |
| B2.1 | 2 |
| B2.2 | 2 |
| B2.3 | 2 |
| B2.4 | 2 |
| B2.5 | 2 |
| B2.6 | 2 |
| Task 2 | 12 |

## Notes for the markers
- The self-check cells use fingerprints, so students cannot read the expected values from the notebook; a tick means the number matches to the stated decimals, nothing more.
- The expected time for the reference solution is about 5 seconds of compute in total; most students need 20 and 25 minutes mostly for typing and thinking.

---

# Instructor key: Part B, Tasks 3 to 5 (Week 5 Prelim)

Not for students. All numbers below were computed by running the reference solution (`testB/week-05.py`; seed 42, CPU). The notebook's ✓/✗ checks accept the reference solution (31 of 31). They compare against fingerprints, so a student whose number rounds to the same 4 or 5 decimals (plus or minus one unit of the last decimal) gets a ✓. The check is a guide; the mark follows the scheme below. Marking principle for every sub-question: result, method (visible code), interpretation (where asked); award partial credit; apply "own-error carried forward" (if an earlier wrong value is used correctly later, do not penalise twice).

## Task 3: Backpropagation and deep networks (16 points)

### B3.1 Forward pass (3 points)
Model answer: z1 = [1.47, 0.07]; a1 = [0.81306, 0.51749]; z2 = [0.37909]; a2 = [0.59365]; loss = 0.17621 (L = ½ (0.59365 − 0)²).
Tolerance: 4 decimals (about plus or minus 0.0002).
- 1 point: hidden layer z1, a1 (W1 @ x + b1 with W1 rows = hidden units; sigmoid).
- 1 point: z2, a2.
- 1 point: loss with the ½ factor.
Typical errors: W1 transposed (x @ W1 or W1.T) gives z1 = [−0.5, ...] (lose point 1, carry on); forgetting the ½ (loss 0.35242: 0.5 point off; award 2 of 3); using target 1 instead of 0; forgetting the sigmoid on the output (a2 = 0.37909).

### B3.2 Backward pass (4 points)
Model answer: delta2 = (a2 − y) a2 (1 − a2) = [0.14321]; dW2 = [ [0.11644, 0.07411] ]; db2 = [0.14321]; delta1 = (W2ᵀ delta2) ⊙ a1 (1 − a1) = [0.01306, −0.01788]; dW1 = [ [0.01045, −0.01959], [−0.01430, 0.02682] ] (outer(delta1, x)); db1 = [0.01306, −0.01788].
Tolerance: 5 decimals (about plus or minus 0.00002).
- 1 point: delta2 (loss slope times sigmoid slope).
- 1 point: dW2 and db2 (delta2 times a1; shape (1, 2) and (1,)).
- 1 point: delta1 (carries delta2 back through W2ᵀ, then multiplies by the sigmoid slope).
- 1 point: dW1 and db1 (outer product with x; shape (2, 2)).
Partial credit: own-error carried forward from B3.1 earns full marks if the method is right (the ✗ in the check does not matter then). Typical errors: using a1 instead of x for dW1; using W2 instead of W2ᵀ (shape error or wrong values); forgetting the sigmoid slope at either layer (delta2 would be 0.59365); dropping the ½ derivative handling (a2 − y is the loss slope for the ½ loss; a factor 2 means they differentiated without the ½); shape (2,) instead of (2, 2) for dW1 (lose 0.5).

### B3.3 Verification (2 points)
Model answer: `max_diff` about 1e-17 (anything below 1e-10 expected; the check demands below 1e-6); `fd` = −0.01959 (matches dW1[0, 1] = −0.01959 to about 1e-9).
- 1 point: autograd comparison: float64 tensors with `requires_grad`, `backward()`, a dictionary of four NumPy gradients, `max_diff` below 1e-6. Using float32 gives about 1e-8, which also passes.
- 1 point: central difference (L(W + eps) − L(W − eps)) / (2 eps) using `loss_at` with only the entry [0, 1] changed. A one-sided difference gives about 1e-6 accuracy; accept if the code is right and the number is within 1e-4 of dW1[0, 1] (0.5 point).
Typical errors: perturbing the whole matrix instead of one entry; dividing by eps instead of 2 eps; reporting a difference rather than the derivative.

### B3.4 Reading three deep networks (3 points)
Model answer. Gradient norms per layer, first layer to last (data `Xm, ym`, seed 7):
- N1 (ReLU, large initialisation): 1.02e+03, 1.13e+03, 8.90e+02, 1.17e+03, 1.68e+03, 1.05e+03, 1.11e+03, 1.13e+03, 9.51e+02: exploding.
- N2 (sigmoid, default initialisation): 2.53e-08, 1.86e-07, 1.18e-06, 7.57e-06, 5.71e-05, 4.53e-04, 3.65e-03, 3.06e-02, 2.24e-01: vanishing.
- N3 (ReLU, He): 1.77e-01, 4.88e-01, 3.32e-01, 4.44e-01, 5.67e-01, 3.37e-01, 3.33e-01, 3.03e-01, 2.04e-01: healthy.
- 2 points: classification (N1 exploding, N2 vanishing, N3 healthy): 2 for three correct, 1 for two correct, 0 otherwise.
- 1 point: `ANSWERS["B3.4_why"]` quotes the numbers: for example the factor of about 10^7 between the first and last layer of N2, norms near 1,000 for N1, norms of the same size (0.2 to 0.6) for N3. Half a point if the reasoning is right but no numbers are quoted.
Typical errors: calling N1 "healthy" because the norms are the same across layers (the size is the clue: about a thousand times too large for a loss gradient); calling N3 "exploding" because the ratio of first to last is not exactly one.

### B3.5 Repairing the vanishing network (4 points)
Model answer (one acceptable repair): ReLU with He initialisation, `torch.manual_seed(7); fixed = make_deep(8, 16, "relu", "he")`. Before: ratio_before = 1.13e-07, acc_before = 0.500 (validation accuracy after 300 epochs of SGD at 0.05). After: ratio_after = 0.867, acc_after = 0.990.
- 1 point: before numbers (ratio 1.1e-07; accuracy 0.500).
- 2 points: a repaired network that is valid (8 hidden layers of 16, seed 7) and measured with the same function: ratio_after at least 1000 times ratio_before and validation accuracy at least 0.9 (1 point for the change and the measured ratio, 1 point for the accuracy). Other acceptable repairs: `residual=True` network with ReLU or tanh, `norm=True` (LayerNorm) with ReLU, tanh with Xavier. Values will differ; accept any repair whose numbers meet the thresholds. If a repair improves the ratio but not the accuracy (for example a residual sigmoid network, ratio about 1e-2, which may still be at 0.5 accuracy after 300 epochs), give 1 of the 2.
- 1 point: `ANSWERS["B3.5"]` names the change and explains the mechanism (sigmoid slope at most 0.25 per layer so the product shrinks; ReLU slope 1; He or Xavier keeps the size of signals) and quotes the numbers.
Typical errors: changing the learning rate or the number of epochs only (no effect on the ratio; 0 for the change); changing the optimiser to Adam without addressing the cause (may raise accuracy but not the ratio: 1 of 2, and 0 for the explanation if it claims the gradients are fixed); comparing different data splits for before and after.

## Task 4: Convolutional networks (16 points)

### B4.1 A convolution and its sizes (4 points)
Model answer: `out1` = [ [−8, −1, −2, 2], [−5, 4, −4, 10], [−1, −2, −2, 3], [−7, 2, 0, 11] ] (shape 4 × 4); `n_out` = ⌊(7 + 2·1 − 3)/2⌋ + 1 = 4; `conv_params` = 5·5·3·12 + 12 = 912; `out_side` = ⌊(40 + 2·2 − 5)/2⌋ + 1 = 20; `dense_params` = (3·40·40)·(12·20·20) + 12·20·20 = 4,800 · 4,800 + 4,800 = 23,044,800.
Tolerance: exact (integers).
- 1 point: `out1` (conv2d_np with stride=2, pad=1). Half a point if only one value is wrong.
- 1 point: `n_out` from the formula (4). Give the point if the formula is shown or the code uses it, even if `out1` is wrong.
- 1 point: `conv_params` 912.
- 1 point: `out_side` 20 and `dense_params` 23,044,800 (0.5 each).
Typical errors: 5·5·12 + 12 = 312 (forgot input channels); out_side 22 (padding not doubled) or 21 (floor/ceil confusion); dense layer without biases (23,040,000: 0.5 point); using the 1-channel picture of part (a) for part (b).

### B4.2 Backward pass of a convolution (5 points)
Model answer, L = sum(G2 ⊙ conv(x2, K2)), stride 1, no padding:
dL/dK = [ [15, 13, 17], [3, 22, 10], [10, 1, 10] ];
dL/dx = [ [1, 2, −2, −3, 2, 0], [1, 6, 7, −4, −7, 3], [−2, −2, 9, 15, −3, −2], [1, 1, −1, −2, 12, 2], [−1, −1, 8, −2, −2, 1], [0, −2, −3, 4, −1, 0] ].
`err_auto` is 0 or below 1e-12; `fd_K` = 10.0000 (equals dL/dK[1, 2] = 10).
Reference loops: dK[a, b] += G[i, j] · x[i + a, j + b]; dx[i + a, j + b] += G[i, j] · K[a, b].
- 2 points: `conv_backward_kernel` (1 point for the correct sum over output positions with the correct input offset, 1 point for correct values). Reward vectorised solutions if correct; the task says explicit loops, so deduct nothing for the loop style if the logic is visible.
- 2 points: `conv_backward_input` (1 point for the scatter of G · K into the window, 1 point for correct values). Using a full convolution with the flipped kernel is correct (the lesson shows it): full marks.
- 1 point: verification: PyTorch gradients (`F.conv2d` on float64 with the (1, 1, H, W) reshape) compared with both gradients, and the central difference of K2[1, 2]. 0.5 for autograd only, 0.5 for finite difference only.
Typical errors: dK[a, b] using x[i, j] instead of x[i + a, j + b] (every cell gets the same number: 0 for correctness, 0.5 for structure); kernel flipped in dx (values differ at the border); autograd compared with a shape mismatch or with `float32` and tolerance fudged; `fd_K` computed with eps = 1e-6 in float32 (noisy: ask for float64; accept if within 1e-3).

### B4.3 CNN against a dense network under shift (5 points)
Model answer (one design): CNN = Conv(1,6,3,pad 1) – ReLU – MaxPool(2) – Conv(6,12,3,pad 1) – ReLU – MaxPool(2) – Flatten – Linear(300, 5): 2,225 parameters. MLP = Flatten – Linear(400, 5) – ReLU – Linear(5, 5): 2,035 parameters (8.5 % below the CNN's). Trained 25 epochs, Adam lr 0.01, seed 0 inside the helper:
- cnn_centre = 1.000, cnn_shift = 0.450, mlp_centre = 1.000, mlp_shift = 0.193 (chance is 0.20).
- drop_cnn = 0.550, drop_mlp = 0.807.
Other seeds and designs give a CNN shift accuracy between about 0.40 and 0.55 and an MLP shift accuracy between about 0.18 and 0.30; the ordering (CNN above MLP on shifted pictures) should hold for every sensible design. Different numbers are fine; the check only requires the structural conditions and consistency.
- 2 points: (a) a valid CNN (at least two convolution layers, at most 3,000 parameters, output 5, a Linear layer sized from the pooled map: 12·5·5 = 300 for the design above) and a matched MLP (within 15 % of the CNN's parameter count). 1 point for each network. Typical error: the Linear layer's input size is wrong (shape error at the first forward pass; the starter then fails: give 0.5 if the student fixes it later).
- 2 points: (b) both trained with the given call; RES filled with test accuracies measured by `accuracy` on `X0` and `X4` (1 point for training both, 1 point for the four measurements computed from the trained models and not typed in).
- 1 point: (c) drops computed as centre minus shifted for each model.
Typical errors: an unfair comparison (MLP with 64 hidden units, 25,000 parameters: lose the matching point and comment); measuring the accuracy on the training set; training with shifted pictures (not asked); drops reported as shift minus centre (negative; 0.5).

### B4.4 Interpretation (2 points)
Model answer: for the reference numbers, the CNN falls from 1.00 to 0.45 (drop 0.55) and the MLP from 1.00 to 0.19 (drop 0.81), so the drops differ by about 26 percentage points and the CNN is more robust. Reason: a dense layer has one weight per pixel position, so a shape that moves meets weights that never saw it; convolution kernels are shared across positions (weight sharing) and so detect a pattern wherever it appears, and pooling adds a little tolerance to small shifts. A change that helps the weaker network: train on shifted pictures (data augmentation), or use a CNN with more pooling or a global pool, or increase the training data covering shifts.
- 0.5 point: names the more robust network (CNN) and quotes the two drops or their difference from the student's own numbers (consistent with their RES).
- 1 point: explanation with weight sharing and position-specific weights (the idea of local connectivity and sharing; accept "a kernel looks for a pattern anywhere").
- 0.5 point: a sensible remedy (augmentation with shifted pictures, more shifted data, pooling, global pooling, a bigger receptive field).
Typical errors: "the CNN has more parameters" (wrong: the matched sizes are the point; 0 for the reason); claiming the CNN is invariant (it drops to 0.45 too: equivariance plus pooling gives only partial tolerance; accept "more robust", not "unaffected"); mixing up which network was weaker.

## Task 5: Synthesis (6 points, 2 each)

### B5.1 (Weeks 2 and 3), 2 points
Model answer: (a) 0.693 = ln 2 is the binary cross-entropy of always predicting 0.5, so the network has learned nothing and predicts 0.5 for every row. (b) A sigmoid layer passes back at most 0.25 of the gradient (the slope is at most 0.25), so over 10 layers at most 0.25^10 = 9.5e-07 (about 1e-6) of it reaches the first layers; more data gives the same tiny gradients, and a lower learning rate makes the steps smaller still, so neither plan helps. (c) Any two of: ReLU instead of sigmoid, He (or Xavier) initialisation, residual connections, normalisation, fewer layers, gradient clipping is not relevant (it is for exploding gradients).
- 1 point: (a) and (b) together: ln 2 identified and the 0.25 per layer and about 1e-6 over 10 layers, with the reason both plans fail (0.5 for each half).
- 1 point: two valid changes with a reason (0.5 each).
Typical errors: saying "overfitting" or "needs more epochs"; using 0.25 × 10 = 2.5 instead of 0.25^10; naming clipping or a higher learning rate as the remedy for vanishing gradients.

### B5.2 (Weeks 2 and 4), 2 points
Model answer: (a) dense 784·128 + 128 + 128·3 + 3 = 100,867 parameters; convolution layer 5·5·1·8 + 8 = 208. (b) Overfitting (100 % train, 62 % test; too many parameters for 300 photos); a CNN uses local connectivity and one shared kernel for all positions, so it has far fewer parameters and finds a pattern wherever the leaf lies, which helps with off-centre leaves. (c) Data side: augmentation (shifted, flipped, rotated photos) or collecting more photos; model side: a CNN with pooling, or regularisation (dropout, weight decay, early stopping), or a smaller model.
- 1 point: the two parameter counts (0.5 each; accept 100,480 + 387 shown as separate layers) and the diagnosis "overfitting" (generalisation gap) with the Week 4 reason (weight sharing, local connectivity, fewer parameters). If the counts are right but the diagnosis is missing, 0.5.
- 1 point: one valid data-side fix and one valid model-side fix (0.5 each).
Typical errors: forgetting the biases (100,352 + 384 = 100,736: 0.25 off, treat as half credit for that count); calling it underfitting; "use a bigger dense network".

### B5.3 (Weeks 1 and 2), 2 points
Model answer: (a) softmax of (3, 1, 0): top word 0.8438 at temperature 1 (probabilities 0.8438, 0.1142, 0.0420); at temperature 0.5 the logits become (6, 2, 0) and the top word gets 0.9796. (b) The cross-entropy loss of the second word at temperature 1 is −ln 0.1142 = 2.170; training raises that word's score (and lowers the others'), which raises its probability and lowers the loss. (c) A low temperature (for example 0.2 to 0.5, or greedy decoding), possibly with a small top-p or top-k, for accurate and repeatable text; a very high temperature flattens the distribution so unlikely words are chosen and the output becomes random, inconsistent and often false.
Tolerance: 0.844 and 0.980 (3 decimals); 2.17 (2 decimals).
- 1 point: numbers (0.5 for the two probabilities, 0.5 for the loss 2.17) and the direction of training (the correct word's score goes up).
- 1 point: part (c): a low (or zero, greedy) temperature, with the reason about the flatter distribution at high temperature (0.5 for the setting, 0.5 for the consequence).
Typical errors: dividing the probabilities by T instead of the logits; using log base 10 (0.943); choosing "temperature 1.5 for creativity" for an exam tool; claiming a high temperature makes the model "more accurate".

## Totals and pass mark
Task 3: 3 + 4 + 2 + 3 + 4 = 16. Task 4: 4 + 5 + 5 + 2 = 16. Task 5: 2 + 2 + 2 = 6. Tasks 1 and 2 (10 and 12 points) are keyed separately. Part B total 60; with Part A (40) the Prelim is out of 100 and the pass mark is 70.

## Timing evidence
The reference solution runs the whole of Tasks 3 to 5 in about 6 seconds of compute on a CPU (the slowest cells are the two 300-epoch trainings in B3.5 and the two 25-epoch CNN trainings in B4.3). Writing time is the limit: the 30-minute boxes assume the helper code (the `build` function, `loss_at`, the data, `accuracy`) is given, as it is.
