# Week 15 — Pre-final Exam: instructor answer key and marking scheme

**Confidential: for instructors only.** The student notebook (notebooks/week-15.ipynb) is a public download and contains no answers; its self-checks compare against hashed fingerprints. Part A runs in the app (exam-mode quiz id week-15-prefinal, 40 questions, 1 point each). Part B (60 points) is marked by the instructor from the submitted notebook.

---

# Instructor key: Part A (written exam), Week 15 Pre-final

Not for students. 40 questions, 1 point each, 60 minutes, pass mark 70 percent of Part A as a quiz (28 of 40). Calculation answers were recomputed with Python by build_partA.py, which asserts every correct value and that the three distractors differ from it and from each other.

| # | Topic | Answer | Type | Level |
|---|---|---|---|---|
| 1 | Week 11: images as tokens and CLIP | A | concept | recall |
| 2 | Week 11: images as tokens and CLIP | B | calculation | apply |
| 3 | Week 11: images as tokens and CLIP | B | calculation | apply |
| 4 | Week 11: images as tokens and CLIP | C | calculation | apply |
| 5 | Week 11: images as tokens and CLIP | B | concept | recall |
| 6 | Week 11: images as tokens and CLIP | C | calculation | apply |
| 7 | Week 11: images as tokens and CLIP | A | calculation | apply |
| 8 | Week 11: images as tokens and CLIP | D | concept | analyse |
| 9 | Week 11: images as tokens and CLIP | A | calculation | apply |
| 10 | Week 11: images as tokens and CLIP | D | concept | analyse |
| 11 | Week 12: GANs and transpose convolution | B | concept | recall |
| 12 | Week 12: GANs and transpose convolution | A | calculation | recall |
| 13 | Week 12: GANs and transpose convolution | D | calculation | apply |
| 14 | Week 12: GANs and transpose convolution | C | calculation | apply |
| 15 | Week 12: GANs and transpose convolution | A | calculation | apply |
| 16 | Week 12: GANs and transpose convolution | D | calculation | apply |
| 17 | Week 12: GANs and transpose convolution | B | concept | recall |
| 18 | Week 12: GANs and transpose convolution | B | concept | analyse |
| 19 | Week 12: GANs and transpose convolution | C | concept | analyse |
| 20 | Week 12: GANs and transpose convolution | D | calculation | recall |
| 21 | Week 13: diffusion models | B | calculation | recall |
| 22 | Week 13: diffusion models | C | calculation | apply |
| 23 | Week 13: diffusion models | C | calculation | apply |
| 24 | Week 13: diffusion models | C | concept | recall |
| 25 | Week 13: diffusion models | D | calculation | apply |
| 26 | Week 13: diffusion models | A | calculation | apply |
| 27 | Week 13: diffusion models | D | calculation | recall |
| 28 | Week 13: diffusion models | B | concept | analyse |
| 29 | Week 13: diffusion models | A | concept | analyse |
| 30 | Week 13: diffusion models | A | calculation | apply |
| 31 | Week 14: training LLMs | D | calculation | apply |
| 32 | Week 14: training LLMs | A | calculation | recall |
| 33 | Week 14: training LLMs | B | concept | recall |
| 34 | Week 14: training LLMs | A | calculation | apply |
| 35 | Week 14: training LLMs | D | calculation | apply |
| 36 | Week 14: training LLMs | C | calculation | apply |
| 37 | Week 14: training LLMs | D | concept | recall |
| 38 | Week 14: training LLMs | B | calculation | apply |
| 39 | Week 14: training LLMs | C | concept | analyse |
| 40 | Week 14: training LLMs | C | concept | analyse |

Level counts: recall 12, apply 20, analyse 8

Answer position counts: A 10, B 10, C 10, D 10

Calculation questions: 25

## Working for calculation questions

**Q2** (Week 11: images as tokens and CLIP), correct B: 97

- Working: (192/16)*(128/16)=96; +1 = 97. Distractors: 96; 192*128/16=1536; 12+8+1=21.
- Options: A 96; B 97; C 21; D 1536

**Q3** (Week 11: images as tokens and CLIP), correct B: 226,176

- Working: 14*14*3=588; 588*384+384=226176. Distractors: no bias 225792; no channels 75648; sum 972.
- Options: A 972; B 226,176; C 225,792; D 75,648

**Q4** (Week 11: images as tokens and CLIP), correct C: 25,350

- Working: 65^2=4225; *6=25350. Distractors: one head 4225; 64^2*6=24576; 65*6=390.
- Options: A 4,225; B 390; C 25,350; D 24,576

**Q6** (Week 11: images as tokens and CLIP), correct C: 0.8889

- Working: dot=8; |a|=|b|=3; cos=0.8889. Distractors: 8/18=0.4444; dot 8; 8/6=1.3333.
- Options: A 0.4444; B 8; C 0.8889; D 1.3333

**Q7** (Week 11: images as tokens and CLIP), correct A: 0.7311

- Working: 6 and 5 -> 1/(1+e^-1)=0.7311. Distractors: scale 1 0.5125; 0.30/0.55=0.5455; scale 100 0.9933.
- Options: A 0.7311; B 0.5455; C 0.5125; D 0.9933

**Q9** (Week 11: images as tokens and CLIP), correct A: 0.7675

- Working: (0.2231+0.6931+1.3863)/3=0.7675. Distractors: -ln(mean p)=0.6604; mean p=0.5167; sum=2.3026.
- Options: A 0.7675; B 0.6604; C 2.3026; D 0.5167

**Q12** (Week 12: GANs and transpose convolution), correct A: 0.75

- Working: 0.6/(0.6+0.2)=0.7499999999999999. Distractors: 0.25; 0.40; 3.00.
- Options: A 0.75; B 0.40; C 3.00; D 0.25

**Q13** (Week 12: GANs and transpose convolution), correct D: 0.5798

- Working: 0.2231+0.3567=0.5798. Distractors: -ln0.8-ln0.3=1.4271; ln0.8+ln0.7=-0.5798; 0.56.
- Options: A -0.5798; B 0.5600; C 1.4271; D 0.5798

**Q14** (Week 12: GANs and transpose convolution), correct C: 49

- Working: 0.98/0.02=49. Distractors: 1/0.02=50; the saturating gradient 0.02; 2.
- Options: A 2; B 50; C 49; D 0.02

**Q15** (Week 12: GANs and transpose convolution), correct A: 13

- Working: (5-1)*3-2*2+5=13. Distractors: 17; 15; 9.
- Options: A 13; B 15; C 9; D 17

**Q16** (Week 12: GANs and transpose convolution), correct D: 131,136

- Working: 128*64*16+64=131136. Distractors: no bias 131072; k not squared 32832; no in-channels 1088.
- Options: A 32,832; B 131,072; C 1,088; D 131,136

**Q20** (Week 12: GANs and transpose convolution), correct D: -1.3863

- Working: 2 ln 0.5=-1.3863=-ln4. Distractors: -0.6931; +1.3863; -2.7726.
- Options: A 1.3863; B -0.6931; C -2.7726; D -1.3863

**Q21** (Week 13: diffusion models), correct B: 0.884352

- Working: 0.98*0.96*0.94=0.884352. Distractors: 1-0.12=0.88; 1-ab=0.115648; 0.12.
- Options: A 0.115648; B 0.884352; C 0.120000; D 0.880000

**Q22** (Week 13: diffusion models), correct C: 0.90

- Working: 0.8*1.5+0.6*(-0.5)=0.90. Distractors: 0.64*1.5+0.36*(-0.5)=0.78; 1.50-0.50=1.00; 1.20+0.30=1.50.
- Options: A 0.78; B 1.50; C 0.90; D 1.00

**Q23** (Week 13: diffusion models), correct C: 0.25

- Working: 0.2/0.8=0.25. Distractors: 4.00; 0.20; 0.80.
- Options: A 4.00; B 0.20; C 0.25; D 0.80

**Q25** (Week 13: diffusion models), correct D: 0.8845

- Working: bracket 0.8667; /0.9798 = 0.8845. Distractors: 0.8667; /0.8 = 1.0833; inner 0.04/0.8: 0.8930.
- Options: A 0.8667; B 0.8930; C 1.0833; D 0.8845

**Q26** (Week 13: diffusion models), correct A: 0.0200

- Working: 10000^(0.5)=100; sin(0.02)=0.0200. Distractors: sin2=0.9093; sin(2/10000)=0.0002; cos0.02=0.9998.
- Options: A 0.0200; B 0.0002; C 0.9093; D 0.9998

**Q27** (Week 13: diffusion models), correct D: 2.20

- Working: 0.2+5*0.4=2.20. Distractors: 3.00; 3.20; 0.6+2.0=2.60.
- Options: A 3.00; B 3.20; C 2.60; D 2.20

**Q30** (Week 13: diffusion models), correct A: 48

- Working: 786432/16384=48. Distractors: 64 (spatial only); 8; 192 (=786432/4096).
- Options: A 48; B 192; C 64; D 8

**Q31** (Week 14: training LLMs), correct D: 1.3040

- Working: 3.9120/3=1.3040. Distractors: with prompt 0.8245; -ln(0.3333)=1.0986; sum 3.9120.
- Options: A 1.0986; B 3.9120; C 0.8245; D 1.3040

**Q32** (Week 14: training LLMs), correct A: 0.8455

- Working: sigma(1.7)=0.8455. Distractors: sigma(-1.7)=0.1545; sigma(2.1)=0.8909; sigma(0.4)=0.5987.
- Options: A 0.8455; B 0.5987; C 0.1545; D 0.8909

**Q34** (Week 14: training LLMs), correct A: 0.5291

- Working: Z=2.7933; pi*=[0.179, 0.2919, 0.5291]. Distractors: softmax(r)[3]=0.6652; beta=2 0.3534; ref 0.2.
- Options: A 0.5291; B 0.2000; C 0.6652; D 0.3534

**Q35** (Week 14: training LLMs), correct D: 0.3711

- Working: margin 0.8; -ln sigma(0.8)=0.3711. Distractors: no beta 0.0181; sign flipped 1.1711; ln2 0.6931.
- Options: A 0.0181; B 0.6931; C 1.1711; D 0.3711

**Q36** (Week 14: training LLMs), correct C: 65,536

- Working: 8*(4096+4096)=65536. Distractors: full W 16777216; r*W 134217728; one factor 32768.
- Options: A 32,768; B 16,777,216; C 65,536; D 134,217,728

**Q38** (Week 14: training LLMs), correct B: 1.82 GB

- Working: 1.50+0.32=1.82 GB. Distractors: 1.50; 6.32; 3.32.
- Options: A 6.32 GB; B 1.82 GB; C 3.32 GB; D 1.50 GB

---

# Instructor key: Week 15 Pre-final practical exam (Part B), Tasks 1 and 2 (26 points)

Not for students. All numbers were computed by running the reference solution (seed 42, CPU, PyTorch of the test venv; the whole of Tasks 1 and 2 runs in about 6 seconds, the same in quick mode). Reference notebook: `w15/ref-P1.py`; every self-check prints a tick for it, and the unmodified student starter runs without errors (all crosses). The self-checks accept the value rounded to the stated decimals (plus or minus 0.4 of the last digit) and never show it. Marks are in whole or half points. Rule for all code questions: correct method but wrong number earns half of the numeric points; a hard-coded number with no working earns at most half. No points are deducted for a wrong answer. Where a student's earlier answer is wrong but later work is right for the wrong input, give the later marks (own-error carried forward).

## Task 1. Multimodal AI: patches, ViT and CLIP (12 points)

### B1.1 Counting patch tokens (3)
- Model answer: `n_patches(h, w, p) = (h // p) * (w // p)`. `CASES` give 224 × 224, p 16: 14 × 14 = **196**; 224 × 160, p 16: 14 × 10 = **140**; 200 × 200, p 16: 12 × 12 = **144** (200 / 16 = 12.5, the last 8 pixel columns and rows are dropped); 384 × 384, p 32: 12 × 12 = **144**. `N_list = [196, 140, 144, 144]`. `kept_share` = 144 × 256 / 40,000 = 36,864 / 40,000 = **0.9216**.
- Tolerance: `N_list` exact (list of four integers); `kept_share` 4 decimals (0.9216; accept 92.16 percent written as 0.9216).
- Marking: `n_patches` code plus `N_list` 2 (the function 1, the list 1; a correct list from a hand count with the function missing: 1); `kept_share` 1. Typical errors: rounding up (`ceil`) to 13 × 13 = 169 for the 200 × 200 case (the lesson's patch grid drops the leftover pixels): 0 for that entry, 0.5 for the function if the rest is right; `h * w / p²` (156.25 and 196, 140, 156, 144): 0 for the list, half for the method; `kept_share = 144 / 156.25` or 1 − 144/169: 0, half if the idea "patch area over image area" is clear.

### B1.2 Cutting an image into patches (2)
- Model answer: `x = img.reshape(H // p, p, W // p, p, C).transpose(0, 2, 1, 3, 4).reshape(N, p·p·C)`. For `p = 4` the shape is **(16, 48)**: 4 × 4 = 16 patches of 4 · 4 · 3 = 48 numbers. Patch 5 (counting from 0) is in grid row 1, grid column 1, so it starts at pixel row 4, column 4 and holds the values `48·row + 3·col + channel` for rows 4 to 7, columns 4 to 7, channels 0 to 2: the first numbers are 204, 205, 206, 207, …; the sum is **13,512** (48 numbers with mean 281.5). `lost_nothing = True`: the patches hold the same 768 numbers, only rearranged (`patches.sum() == img.sum()`).
- Tolerance: shape exactly (16, 48); sum exactly 13,512; the Boolean `True`.
- Marking: `patch_shape` 0.5; `patch5_sum` 1; `lost_nothing` 0.5. `patchify` must use `reshape` and `transpose` (a double loop with slicing is accepted and earns the same, if the order is right; flag it as a method difference). Typical errors: forgetting the transpose, so the "patches" are four rows of the image (shape right, `patch5_sum` wrong: 13,512 versus 16,896 or similar): shape 0.5, sum 0, half a mark for the sum if the reasoning is visible; ordering the patches column by column: sum of patch 5 is different, 0; answering `False` for `lost_nothing` (confusing "rearranged" with "lost"): 0.

### B1.3 Sequence length, embedding and cost (2)
- Model answer: `embed_params` = 16 · 16 · 3 · 192 + 192 = 147,456 + 192 = **147,648**. `seq_len` = 14 · 14 + 1 = **197** (196 patches plus the class token). `scores_per_head` = 197² = **38,809**. With patch side 8 there are 28 · 28 = 784 patches, 785 tokens and 785² = 616,225 scores, so `ratio_scores` = 616,225 / 38,809 = **15.88** (about 16: halving the patch side gives 4 times the tokens and about 16 times the scores).
- Tolerance: `embed_params`, `seq_len`, `scores_per_head` exact; `ratio_scores` 2 decimals (15.88; 16 is accepted by the marker with the note that the exact value with the class token is 15.88).
- Marking: 0.5 each. Typical errors: forgetting the bias (147,456): 0 with the half-mark rule only if the reasoning is visible (the bias is in the stated task); 196 for the length (class token forgotten): 0, and then 196² = 38,416 is carried forward as correct for `scores_per_head`; ratio of 4 (tokens, not scores): 0.

### B1.4 CLIP-style zero-shot choice (2.5)
- Model answer: with `img_vec = [0.7, 0.5, 0.2, 0.4]` (norm 0.9487) the cosines in dictionary order are **[0.9785, 0.7302, 0.4962, 0.8734]**. The logits are 10 × cosine = [9.785, 7.302, 4.962, 8.734], and the softmax gives **[0.6939, 0.0579, 0.0056, 0.2426]** (sum 1). The best sentence is **"a photo of a cracked pier"**, `p_best` = **0.6939**. The loose-bolt sentence takes 0.24: the zero-shot answer is a choice among the four given sentences only.
- Tolerance: `cos_sims` and `probs` to 3 decimals (the self-check mixes all entries); `best` as the exact sentence (case-insensitive); `p_best` 3 decimals (0.694).
- Marking: `cos_sims` 0.75; `probs` 0.75; `best` 0.5; `p_best` 0.5. Typical errors: the dot product without the norms ([0.9, 0.69, 0.5, 0.88]): 0 for `cos_sims`, half of the `probs` mark for the right softmax method; no scale (softmax of the cosines: [0.303, 0.237, 0.187, 0.273], a nearly flat result, the winner is the same, `p_best` = 0.303): `probs` 0 and `p_best` 0 but `best` 0.5; scale 20 instead of 10 (p_best 0.886): method half.

### B1.5 The contrastive loss (1.5)
- Model answer: with scale 10 the logits are 10 · S. Row-wise (each image picks its text): the diagonal probabilities are 0.9519, 0.8406, 0.8545, so `loss_i2t` = mean(−ln p) = **0.1268**. Column-wise (each text picks its image): 0.9646, 0.7611, 0.9100, so `loss_t2i` = **0.1345**. `loss_clip` = (0.1268 + 0.1345) / 2 = **0.1306**. (The unaveraged sum of the three row losses is 0.380.)
- Tolerance: 3 decimals each (0.127, 0.134 and 0.131; the exact value of `loss_t2i` is 0.13446).
- Marking: 0.5 each. Typical errors: using `S` without the scale (loss about 1.0): 0; transposing the wrong way (swapping `loss_i2t` and `loss_t2i`): 0 for both, half if the idea is clear; using `.sum()` instead of the mean: half; `loss_clip` as the sum of the two (0.261) instead of the mean: 0, half if the sum is clearly labelled.

### B1.6 Choosing how to fuse (1, written)
- Model answer: the projector option adds (224 / 14)² = 16 × 16 = **256 image tokens**, which is 256 / 2,048 = **12.5 percent** of the context window (1,792 tokens are left for the text). For a question that needs a sentence about a detail in the photo (crack width), the projector with prefix tokens (LLaVA style) or cross-attention are the sensible options; cross-attention saves the context but needs new layers. A CLIP-style shared space only ranks or matches given sentences, so it cannot answer an open question. Risk to name: hallucination (a confident crack width that is not in the photo), the answer must be checked against the photo by a person; also acceptable: image tokens use up context, a wrong scale (a card of unknown size), poor resolution.
- Marking: 1 point in total: 0.5 for the numbers 256 and 12.5 percent (or the equivalent), 0.25 for a justified choice, 0.25 for one valid risk. A choice of the shared space with a good reason (it only needs to rank a fixed list such as crack classes) is accepted if the student says it cannot write a free answer. The self-check only tests that at least 25 words are present.

Task 1 total: 3 + 2 + 2 + 2.5 + 1.5 + 1 = 12.

## Task 2. Image generation: GANs and transpose convolution (14 points)

### B2.1 Transpose convolution sizes (2)
- Model answer: `convT_out(n, k, s, p, op) = (n − 1)·s − 2p + k + op`. The five cases give **[8, 13, 14, 14, 20]**: (4, 4, 2, 1, 0): 3·2 − 2 + 4 = 8; (7, 3, 2, 1, 0): 12 − 2 + 3 = 13; (7, 3, 2, 1, 1): 13 + 1 = 14; (5, 5, 3, 2, 1): 12 − 4 + 5 + 1 = 14; (10, 2, 2, 0, 0): 18 + 2 = 20. The sides produced by `nn.ConvTranspose2d` are the same, so `all_match = True`.
- Tolerance: lists exact; the Boolean `True`.
- Marking: `sizes` 1 (a single wrong entry: 0.5); `torch_sizes` 0.5 (the student must build the real layers; copying the formula result without a layer: 0); `all_match` 0.5. Typical errors: forgetting `out_pad` (13, 13, …): the third and fourth entries are wrong; using the convolution formula (n + 2p − k)/s + 1 (gives 1, 3, 3, 1, 5): 0; passing `output_padding` as a positional argument to the wrong place (a PyTorch error, not an answer error): look at the code.

### B2.2 A small DCGAN generator (3)
- Model answer: noise 64 → fully connected 64 → 4096 (4 × 4 × 256); sides after the layers: **[4, 8, 16, 32]** (each transpose convolution with k 4, stride 2, pad 1 doubles the side). Parameters: fully connected 64 · 4,096 + 4,096 = **266,240**; ConvT 256 → 128: 4 · 4 · 256 · 128 + 128 = **524,416**; 128 → 64: 4 · 4 · 128 · 64 + 64 = **131,136**; 64 → 1: 4 · 4 · 64 · 1 + 1 = **1,025**. `layer_params = [266240, 524416, 131136, 1025]`, `total_params` = **922,817**, and `param_count` of the PyTorch model gives the same 922,817 (`torch_total`). The output is a 32 × 32 picture with 1 channel; the first transpose convolution holds more than half of the parameters.
- Tolerance: all exact.
- Marking: `sides` 1; `layer_params` 1 (3 of 4 right: 0.5); `total_params` 0.5; `torch_total` 0.5 (needs the built model; a copied number: 0). Typical errors: leaving out the biases (262,144 + 524,288 + 131,072 + 1,024 = 918,528): `layer_params` 0, half if the shape of the work is right; forgetting the fully connected layer; treating the channels as the side.

### B2.3 The optimal discriminator (2.5)
- Model answer: `d_star` = p_data / (p_data + p_g) = 0.6 / 0.8 = **0.75**. At p_g = p_data: **0.5**. When the generator is perfect and D* = 1/2 everywhere, V = ln(1/2) + ln(1 − 1/2) = 2 ln(1/2) = **−ln 4 = −1.3863**. For the batch: −mean(ln [0.9, 0.8, 0.7]) = 0.2284 and −mean(ln [0.8, 0.7, 0.9]) = 0.2284, so `d_loss` = **0.4568**.
- Tolerance: `d_star` 3 decimals (0.750); `d_star_eq` 0.500; `v_eq` 4 decimals (−1.3863); `d_loss` 4 decimals (0.4568).
- Marking: 0.5 + 0.5 + 0.5 + 1. Typical errors: `d_star` = 0.6/0.2 = 3 or 0.25 (swapped): 0; `v_eq` = −ln 2 (0.6931) or +1.386 (sign): 0, half if the student says the −ln 4 idea in words; `d_loss` as the mean of the six logs (0.2284): half; forgetting the "1 −" for the fakes (−mean(ln [0.2, 0.3, 0.1]) = 1.7...): 0.

### B2.4 Saturating and non-saturating generator loss (2)
- Model answer: with a = ln(0.01/0.99) = −4.5951, D = sigmoid(a) = 0.01. Autograd gives d/da of ln(1 − D) = −D = **−0.0100** and d/da of (−ln D) = −(1 − D) = **−0.9900**. The ratio of sizes is **99** (0.99 / 0.01). Written: the original loss has almost no gradient when the generator is losing, so the generator hardly learns at the start; the non-saturating loss has a strong gradient exactly there (and the same goal, to push D(G(z)) up).
- Tolerance: 4 decimals for the gradients (−0.0100, −0.9900); ratio to 1 decimal (99.0); sign matters for the two gradients (a student who reports magnitudes only: 0.25 each, mention it in the feedback).
- Marking: `grad_sat` 0.5; `grad_ns` 0.5; `ratio_grad` 0.5; written 0.5 (must say that the saturating gradient vanishes when D(G(z)) is near 0 and that the non-saturating one does not, with a number). Autograd must be used: a hand formula is accepted at half marks for the numbers (the task says autograd). Typical errors: the gradient with respect to D instead of the logit (−1.0101 and −100, ratio 99.0 too): 0 for the two gradients, 0.5 for the ratio, half a mark each if the derivative is correct and the variable is said; a ratio of 100 (from the D derivatives, 100/1.01 = 99.0, rounded): accept for 0.5 if explained.

### B2.5 A tiny GAN on a line (2)
- Model answer: generator step `g_loss = -F.logsigmoid(D(G(z))).mean()` (or BCE-with-logits with label 1); `optG.zero_grad(); g_loss.backward(); optG.step()`. Only G is updated (the discriminator has its own optimiser, and `optD.zero_grad()` is called before its next step, so leftover gradients in D do no harm). The reference run (seed 42, 800 steps) gives a generated mean of **3.76** and a standard deviation of **0.81** for 2,000 samples (real data: 4 and 1). Other seeds in testing gave means between 3.2 and 4.7 and spreads between 0.35 and 1.3, so the checks accept a mean in (3.0, 5.0) and a spread in (0.25, 2.0); an untrained generator gives a mean near 0 and fails.
- Marking: generator step 1 (must use the discriminator's score of the fakes with the non-saturating form: −ln D(G(z)); the saturating `log(1 − D)` form earns 0.5 if the code otherwise works; a missing `zero_grad` or an update of D's parameters in this step: 0.5); `gen_mean` 0.5; `gen_std` 0.5 (the values must be read from the trained generator, not typed). Do not mark down for a different but healthy mean or spread. Typical errors: detaching the fakes in the generator step (no gradient reaches G: the mean stays near 0, 0 marks); minimising `-log(1 - D)` (wrong sign), the generator is pushed the wrong way.

### B2.6 Reading a failing log (1.5)
- Model answer: `d_loss_guess` = −ln 0.5 − ln 0.5 = **1.3863** (= ln 4). Written: the log shows **mode collapse** (only 1 of the 8 kinds of leaf is generated, the spread 0.03 is far below the real 0.9) together with a discriminator that has won (discriminator loss 0.02 is far below the guessing level 1.386; generator loss 4.7 is high, so G is not fooling D). Remedies: weaken D or slow it (lower learning rate, fewer D steps, label smoothing or noise on D's input), use the non-saturating loss, feed more variety (minibatch statistics, bigger batch, a different loss such as Wasserstein), check by counting modes.
- Tolerance: `d_loss_guess` 3 decimals (1.386).
- Marking: `d_loss_guess` 0.5; written 1: name the failure (mode collapse; "the discriminator wins" as the second half is welcome) 0.5; two numbers as evidence 0.25; one remedy 0.25. A student who writes only "oscillation" or "the generator is perfect because D loss is low": 0. Typical errors: reading the 0.02 loss as "everything is fine" (the lesson warns that losses do not tell you whether pictures are good, but a D loss this low means the discriminator wins easily); using 0.693 for `d_loss_guess` (half of 1.386, the per-term value): half a mark if the student says the loss in the B2.3 form is the sum of two terms, otherwise 0.

### B2.7 Transpose convolution or resize-and-convolve? (1)
- Model answer: `params_convT` = 3 · 3 · 64 · 32 + 32 = **18,464** (the side 7 → 14 with k 3, s 2, p 1, out_pad 1: (7 − 1)·2 − 2 + 3 + 1 = 14). Written: with k = 3 and stride 2 the kernel size is not divisible by the stride, so some output cells get more overlapping kernel copies than others and the output shows a checkerboard; resize-and-convolve first enlarges the grid (nearest neighbour or bilinear) and then applies an ordinary convolution, so every cell is treated the same (no uneven overlap); a kernel of 4 with stride 2 (as in the DCGAN) is another cure.
- Tolerance: exact integer.
- Marking: `params_convT` 0.5 (18,432 without the bias: 0, half if the bias omission is the only error); written 0.5 (both the cause and the alternative; the cause must mention the stride and the kernel size or uneven overlap). The self-check only tests that at least 25 words are present.

Task 2 total: 2 + 3 + 2.5 + 2 + 2 + 1.5 + 1 = 14. Tasks 1 and 2 together: 26 of the 60 points of Part B.

---

# Week 15 — Pre-final Exam: instructor key, Part B Tasks 3 to 5

**Confidential: for instructors only.** All numbers below were computed by running the reference solution (the same code that produced the fingerprints in the public notebook, label `w15-b1`). Self-checks in the notebook only print a tick or a cross; the mark is given from the code, the numbers and the written answers. Tasks 3 to 5 are worth 16 + 12 + 6 = 34 of the 60 points of Part B (Tasks 1 and 2 are keyed separately). Where a student's earlier answer is wrong but later work is right for the wrong input, give the later marks (own-error carried forward). Parts B3.2 and B3.3 use a schedule that is given in the notebook, and B3.5 another one, so a mistake in B3.1 never spoils them.

---

## Task 3: Diffusion models (16 points)

### B3.1 The schedule, ᾱ and the signal-to-noise ratio (3 points)
Model answer. `make_abar(T, b0, b1)` = `np.cumprod(1 - np.linspace(b0, b1, T))`. Test values: `make_abar(6, 0.01, 0.1)` = [0.99000, 0.96228, 0.91802, 0.85926, 0.78880, 0.70992]; `make_abar(40, 0.001, 0.05)` has ᾱ_1 = 0.99900, ᾱ_10 = 0.93536, ᾱ_40 = 0.35428. For T1 = 250, β from 0.0004 to 0.04: `abar_60` = 0.7360, `abar_150` = 0.1569 (index t − 1); `snr_60` = 0.7360 / 0.2640 = 2.788, `snr_150` = 0.1569 / 0.8431 = 0.186; `t_cross` = 92 (SNR is 1.004 at t = 91 and 0.975 at t = 92; equivalently ᾱ_t falls below 0.5 at t = 92).
Tolerance: 4 decimals for ᾱ, 3 for SNR, exact integer for `t_cross`.
- 1 point: the function (linear β with `linspace`, running product with `cumprod`, array of length T).
- 0.5 point: `abar_60`, `abar_150` (0.25 each). 0.5 point: `snr_60`, `snr_150` (0.25 each).
- 1 point: `t_cross` = 92. 0.5 for 91 or 93 (off by one from the indexing), or for the correct reasoning with a wrong ᾱ carried forward.
Typical errors: index t instead of t − 1 (`abar_60` = 0.7287); `np.linspace` with T + 1 points or `np.arange`; the sum instead of the product; SNR = ᾱ (forgetting the denominator) or the square root of the SNR (1.670); asking for ᾱ < 1 or SNR < 0.5.

### B3.2 The closed form and the loss (2 points)
Model answer. At t = 150, ᾱ_150 = 0.41982 (schedule of B3.2), so `w_sig` = √0.41982 = 0.6479 and `w_noise` = √0.58018 = 0.7617 (their squares add to 1). x_t = 0.6479 · [0.6, −0.2, 0.9, −0.7] + 0.7617 · [1.1, −0.4, 0.3, −1.5] = [1.2266, −0.4343, 0.8117, −1.5961]. `mse_eps` = mean of (ε − ε̂)² = (0.09 + 0.09 + 0.04 + 0.09) / 4 = 0.0775 (sum of squares 0.31). Function test values: `noise_to([1, −1, 0.5], [0, 1, −1], 50)` = [0.95046, −0.63960, 0.16437]; `noise_to([0.2, 0.8], [−1, 0.5], 399)` = [−0.98923, 0.53786].
Tolerance: 4 decimals for the weights and the mean squared error, 3 for x_t.
- 1 point: the function `noise_to` (both test values; 0.5 if it uses ᾱ_(t+1), that is `ABAR2[t]`, or √ᾱ and √(1−ᾱ) swapped).
- 0.5 point: `w_sig`, `w_noise` and `x_t`.
- 0.5 point: `mse_eps` = 0.0775. 0.25 for the sum of squares (0.31) or for the root mean squared error (0.2784).
Typical errors: using ᾱ instead of its square root (x_t = [0.8901, ...] for the first entry); using α_t = 1 − β_t instead of ᾱ_t; mean over the wrong axis; dividing by 3.

### B3.3 One reverse step, two ways (3 points)
Model answer (schedule of B3.2, t = 200, s = 150). β_200 = 0.015063, α_200 = 0.984937, ᾱ_200 = 0.21567, √(1 − ᾱ_200) = 0.88562, so `beta_t` = 0.015063 and `coef` = 0.015063 / 0.88562 = 0.017008. σ_t = √β_t = 0.12273. `x_prev` = (x_t − coef · ε̂) / √α_t + σ_t · z = [0.93511, −0.39704, 1.40751] (without the fresh noise it would be [0.89829, −0.29886, 1.39524]). DDIM: `x0_hat` = (x_t − 0.88562 · ε̂) / √0.21567 = [0.98446, −0.26459, 1.29831]; with ᾱ_150 = 0.41982: `x_s` = √0.41982 · x̂₀ + √0.58018 · ε̂ = [1.01872, −0.32378, 1.52675].
Tolerance: 6 decimals for β_t, 5 for coef, 4 for the vectors.
- 0.5 point: `beta_t` and `coef` (0.25 each).
- 1 point: `x_prev` (0.5 for forgetting the division by √α_t, [0.92832, −0.39478, 1.39697]; 0.5 if the fresh noise is left out).
- 0.5 point: `x0_hat`.
- 1 point: `x_s` (uses the weights of s = 150, not of t = 200, and the same ε̂; carried forward from a wrong `x0_hat`).
Typical errors: β_t read as `BETA2[t]` (off by one); σ_t = β_t; the coefficient written β_t / (1 − ᾱ_t) without the root; the DDIM step with the weights of t = 200 (the picture does not move); adding fresh noise to the DDIM step (η = 0 means none).

### B3.4 The sinusoidal time embedding (2 points)
Model answer. `time_emb(t, d)`: for k = 0..d/2−1, angle = t / 10000^(2k/d), entry 2k = sin, entry 2k+1 = cos. Test values: `time_emb(7, 6)` = [0.6570, 0.7539, 0.3192, 0.9477, 0.0151, 0.9999]; `time_emb(130, 4)` = [−0.9301, −0.3673, 0.9636, 0.2675]. For d = 16: `cos_near` (steps 100 and 101) = 0.936, `cos_far` (steps 100 and 250) = 0.246. The nearby steps look alike, the far steps do not.
Tolerance: 4 decimals for the vectors, 3 for the cosines.
- 1 point: the function. 0.5 if sine and cosine are swapped (note: the two cosines are then unchanged, 0.936 and 0.246, so the second point is still given), or if the exponent is k/d instead of 2k/d.
- 1 point: `cos_near`, `cos_far` (0.5 each). A student who uses the dot product without dividing by the lengths gets 0.25 each.
Typical errors: angles in degrees; the exponent 2k/d with k the entry index (not the pair index); base e instead of 10000.

### B3.5 Does the denoiser need to know the step? (3 points)
Model answer. A correct `train_denoiser` draws x0 = `sample_x0(256, gen)`, t = `torch.randint(1, T5 + 1, (256,), generator=gen)`, ε = `torch.randn(256, 2, generator=gen)`, builds x_t = √ᾱ_t · x0 + √(1 − ᾱ_t) · ε with `ABAR5[t − 1]` (shape (256, 1)), computes `F.mse_loss(model(x_t, t), eps)` and takes an Adam step (zero_grad, backward, step). Reference results (500 steps, seed 42, CPU; the checks are tolerant and use only the pattern, not exact values): with time input `mse_t_low` ≈ 0.39, `mse_t_high` ≈ 0.03; without time `mse_n_low` ≈ 0.73, `mse_n_high` ≈ 0.26. (Quick test mode, 250 steps, gives the same pattern.) For reference, a model that always guesses 0 has an error of about 1.
- 1 point: a correct training function (random t for every example, closed-form x_t, MSE against the noise that was added, optimiser step). 0.5 for a loop that uses one t for the whole batch, or that predicts x0 instead of ε (then the numbers do not fit).
- 1 point: the four numbers equal `eval_mse` of the two trained models (0.5) and the with-time model is better at both levels, clearly at the high one (0.5).
- 1 point: `ANSWERS["B3.5"]`: compares with the four numbers (0.5): with time lower at both levels, the gap largest at the high-noise step (about 8 times lower); explains that the same-looking noisy point means different things at different noise levels (at step 95 the point is almost pure noise, so the noise guess is the point itself; at step 15 it is mostly picture and the noise is a small correction), and that without the step the network must use one compromise rule (0.5). Accept "the time input tells the network how much of x_t is noise".
Typical errors: evaluating on the training batch instead of `eval_mse`; forgetting `opt.zero_grad()`; building the model without `torch.manual_seed` (accept); claiming that the model without time is better at low noise because its number happens to be close to 1 (a model that guesses zero would also score about 1); saying the time input makes the model bigger or "remember" the data.

### B3.6 Parameters of two UNet blocks (1 point)
Model answer. Block = (9·C_in·C_out + C_out) + (9·C_out² + C_out) + (D·C_out + C_out). Down block, 32 → 64, D = 64: 18,496 + 36,928 + 4,160 = `p_down` 59,584. Up block, 96 → 32: 27,680 + 9,248 + 2,080 = `p_up` 39,008. Without the skip (64 channels in): 18,464 + 9,248 + 2,080 = 29,792, so `skip_extra` = 39,008 − 29,792 = 9,216 = 9 · 32 · 32 (the extra input channels, 32 of them, times the 3 × 3 kernel and the 32 output channels).
- 0.5 point: `p_down`, `p_up` (0.25 each). 0.5 point: `skip_extra` = 9,216 (carried forward if the formula is consistent).
Typical errors: forgetting the biases (p_down 59,392); forgetting the time projection; counting the second convolution with C_in; using 3 · 3 · ... with the wrong channels for the skip (the copy adds input channels, not output channels).

### B3.7 Guidance and the latent space (1 point)
Model answer. `eps_guided` = ε_u + w (ε_c − ε_u) = [0.30, −0.10, 0.50] + 4 · [0.15, −0.30, −0.30] = [0.90, −1.30, −0.70]. Picture: 640 · 384 · 3 = 737,280 numbers; latent: 80 · 48 · 16 = 61,440; `ratio` = 12.0.
Tolerance: 3 decimals for the vector, 2 for the ratio.
- 0.5 point: `eps_guided`. (w = 1 would give back ε_c = [0.45, −0.40, 0.20]; ε_c + w(ε_c − ε_u) is the classic mistake: [1.05, −1.60, −1.00].) 0.5 point: `ratio` = 12 (0.25 for 48, which is the ratio for 4 channels; the 16 channels change it).

### B3.8 Explain the design (1 point)
Model answer. (i) The noise ε is known exactly when x_t is built, so the loss is a plain mean squared error against a known target of the same size at every noise level (predicting x0 directly is hard at high noise, and ε is easy to read off the formula). (ii) DDIM uses ε̂ to estimate x̂₀ and then jumps along the same schedule without adding fresh noise (η = 0), so the trained network can be used at far fewer steps (20 to 50 calls instead of about 1,000). (iii) Guidance costs two network calls per step (with and without the prompt) and a large w makes pictures follow the prompt more but gives less variety and sometimes harsh pictures.
- 1 point: three parts, one third each (0.25 for a part that is partly right). A student who says DDIM is "better" without the number of steps gets 0.25 for that part; one who says guidance is free gets 0.

**Task 3 total: 3 + 2 + 3 + 2 + 3 + 1 + 1 + 1 = 16.**

---

## Task 4: Training LLMs: SFT, preference learning and LoRA (12 points)

### B4.1 The SFT loss with a masked prompt (2 points)
Model answer. −ln p for the 11 tokens: [1.897, 1.204, 2.526, 1.609, 0.916, 2.120, 0.799, 0.357, 1.386, 0.163, 0.511]; the response part (last 5) adds up to 3.2148, the prompt part (first 6) to 10.2728. `sft_loss(p, is_resp)` = sum(−ln p · mask) / sum(mask). `loss_masked` = 3.2148 / 5 = 0.643 (perplexity 1.90); `loss_all` = 13.4877 / 11 = 1.226. Function test values: `sft_loss([0.5, 0.2, 0.9, 0.1], [0, 1, 0, 1])` = 1.9560 and `sft_loss([0.3, 0.6, 0.8], [1, 1, 1])` = 0.6460.
Tolerance: 4 decimals for the function, 3 for the two losses.
- 1 point: the function (divides by the number of response tokens, not by the length of the sequence; uses the natural logarithm).
- 0.5 point each: `loss_masked`, `loss_all`.
Typical errors: dividing the response sum by 11 (0.292); the mean over all positions for the masked loss (1.226); log base 2 or 10; mean of p instead of −ln p.

### B4.2 The reward model and Bradley–Terry (1 point)
Model answer. Gaps r_w − r_l = [2.2, −0.6, 1.6]; `p_chosen` = σ(gap) = [0.90025, 0.35434, 0.83202]; losses −ln p = [0.1051, 1.0375, 0.1839]; `rm_loss` = 0.4422; `worst_pair` = 1 (the model prefers the rejected answer).
- 0.5 point: `p_chosen` (three values). 0.5 point: `rm_loss` and `worst_pair` (0.25 each).
Typical errors: a softmax over the two scores instead of the sigmoid of the gap (same number: accept); σ(r_l − r_w); the mean of the probabilities (0.6955) instead of the mean of −ln p; `worst_pair` = 1 counted from 1 (answer 2: 0.125).

### B4.3 The KL-regularised optimum (2 points)
Model answer. `kl_opt(ref, r, beta)`: weights ref · exp(r / β), divide by their sum (π*), KL = Σ π* ln(π* / ref). Function test: `kl_opt([0.4, 0.4, 0.2], [0, 1, 2], 0.5)` gives π* = [0.02802, 0.20705, 0.76493] and KL = 0.8153. For ref = [0.55, 0.30, 0.15], rewards [0.5, 1.5, 4.0]: β = 0.25: π* = [0.0000, 0.0001, 0.9999], KL 1.896; β = 1: π* = [0.0869, 0.1288, 0.7844], KL 1.028; β = 4: π* = [0.4247, 0.2975, 0.2779], KL 0.0589. So `kl_list` = [1.896, 1.028, 0.059], `top_share` = [0.9999, 0.7844, 0.2779] (the best policy puts 15% of its mass on answer 3 at the start and almost all of it when β is small), `beta_closest` = 4.0 (the largest β keeps the policy closest to the reference).
Tolerance: 4 decimals for the function, 3 for the lists.
- 1 point: the function (0.5 if the KL is computed in the other direction, KL(ref ‖ π*), or without the normalisation; π* alone correct earns 0.5).
- 0.5 point: both lists. 0.5 point: `beta_closest` = 4.0 (the reverse of the rule, the smallest β, is the typical error; 0.25 if the choice agrees with the student's own wrong lists).
Typical errors: exp(r · β) instead of exp(r / β); forgetting to multiply by ref (π* proportional to exp(r / β) only, which ignores the reference model); forgetting to normalise; `beta_closest` the smallest β.

### B4.4 The DPO loss (1 point)
Model answer. ln-ratios: chosen −1.20 − (−1.60) = 0.40; rejected −2.90 − (−2.30) = −0.60; `margin` = 0.2 · (0.40 − (−0.60)) = 0.200; `dpo` = −ln σ(0.2) = 0.5981; `dpo_start` = ln 2 = 0.6931 (the policy equals the reference, so the margin is 0 and σ(0) = 0.5).
- 0.75 point: `margin` (0.25) and `dpo` (0.5). 0.25 point: `dpo_start`.
Typical errors: forgetting β (margin 1.000, loss 0.313); the sign of the margin reversed (loss 0.798); taking the logarithm of numbers that are already log-probabilities; `dpo_start` = 0.

### B4.5 A LoRA layer (3 points)
Model answer.
```python
class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.scale = alpha / r
        self.A = nn.Parameter(0.01 * torch.randn(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)
```
(a) Hidden test: `nn.Linear(6, 5)`, r = 2, alpha = 4: 22 trainable numbers, base frozen, same output as the base at the start (B = 0), and with A = 0.5 and B = 0.25 everywhere the output on `torch.linspace(-1, 1, 12).reshape(2, 6)` must be the base output plus 2 · x A^T B^T (checked through a fingerprint of the whole output). (b) `LoRALinear(nn.Linear(96, 64), 4, 8)`: `n_trainable` = 4 · (96 + 64) = 640; `n_frozen` = 96 · 64 + 64 = 6,208; `share_pct` = 100 · 640 / 6,208 = 10.31. (c) Reference run: `loss_before` = 3.85, `loss_after` ≈ 5e-6 (150 steps of Adam at 0.02), `changed` = ['A', 'B'], the base weight equal to its copy. The checks ask `loss_after < 0.2 · loss_before`, an unchanged base weight and exactly two changed parameters, none of them from the base.
- 1 point, part (a): frozen base and exactly A and B trainable (0.25 for each of the two checks that concern the parameters), starting output equal to the base (0.25), the factor α/r in the forward pass (0.5).
- 0.5 point, part (b): the two counts (0.25) and the share (0.25).
- 1.5 points, part (c): a correct training loop with only the trainable parameters in the optimiser (0.5), a loss that falls a lot with the base untouched (0.5), the snapshot comparison and `changed` (0.5).
Typical errors: forgetting to freeze the base (35 trainable parameters in the test, and the base weight changes during training); initialising B randomly (the layer differs from the base at the start); the scale α · r instead of α / r; B of shape (r, out) (matrix shapes fail); putting all parameters in the optimiser (harmless if the base is frozen, accept); `n_trainable` counted for the base (6,208); `share_pct` as a fraction (0.1031) or measured against the total of base plus adapter (9.35).

### B4.6 Parameters and memory of a bigger model (2 points)
Model answer. `lora_params` = 40 layers · 4 matrices · 16 · (5120 + 5120) = 26,214,400 (about 0.20 percent of 13 billion). `gb_full` = 16 · 13e9 / 1e9 = 208.0. `gb_lora` = (2 · 13e9 + 14 · 26,214,400) / 1e9 = 26 + 0.367 = 26.367. `gb_qlora` = (0.5 · 13e9 + 14 · 26,214,400) / 1e9 = 6.5 + 0.367 = 6.867 (30 times smaller than full fine-tuning; only QLoRA fits a 24 GB card, before activations).
Tolerance: exact integer; 1 decimal for `gb_full`, 2 for the others.
- 1 point: `lora_params`. 0.5 for 13,107,200 (forgot the factor for d_in + d_out, using r · d) or 6,553,600 (forgot the four matrices), with the memory then carried forward.
- 1 point: `gb_full` (0.25), `gb_lora` (0.5), `gb_qlora` (0.25). 
Typical errors: 16 bytes for every parameter in the LoRA case (208); 14 bytes per parameter of the base; 4-bit counted as 4 bytes; 1 GB = 2^30 bytes (accept if consistent, a small difference).

### B4.7 Choose a method (1 point)
Model answer. DPO (or SFT first and then DPO): the school has chosen and rejected pairs, DPO trains on them with a single loss and no reward model or RL loop, which RLHF would need. A short SFT on the polite answers is also acceptable as the first step. LoRA or QLoRA: with 24 GB a 13 B model cannot be fully fine-tuned (208 GB), but with LoRA (about 27 GB) it is borderline, and with QLoRA (about 7 GB) it fits; only the adapters get gradients and optimiser state. Risk: the model drifts, becomes flattering or loses accuracy (reward hacking, forgetting): keep β so that the policy stays near the reference (KL), and test on new questions that a teacher checks.
- 1 point: method with a reason that uses the pairs (0.5); memory help from LoRA or QLoRA and one risk with a guard (0.5). 0.25 for an answer that says RLHF "because it is the best" without reasons; 0 if the answer ignores the data given.

**Task 4 total: 2 + 1 + 2 + 1 + 3 + 2 + 1 = 12.**

---

## Task 5: Synthesis with the capstone (6 points, 2 each)

Marking: read the answer, give the points in the lines below. The model answers are in the reference notebook as well. The Structural Inspection Assistant rule: a licensed engineer makes every pass or fail decision; an answer that lets the assistant sign off on safety loses the guardrail point.

### B5.1 (Weeks 11 and 14), 2 points
Model answer. (a) 384 / 16 = 24 and 256 / 16 = 16, so 24 · 16 = 384 patch tokens and 385 tokens with the class token; one head computes 385² = 148,225 attention scores per layer. (b) One adapted matrix trains 16 · (768 + 768) = 24,576 numbers; two matrices in each of the 12 layers give 24 · 24,576 = 589,824; the head adds 768 · 4 + 4 = 3,076; total 592,900, which is 592,900 / 86,000,000 = 0.69 percent. (c) The position table was learned for 197 positions (a 14 × 14 grid plus the class token); 385 tokens have no embeddings for the extra positions. Ways out: resize the photos to 224 × 224 (squeezes the photo and loses detail); resize or interpolate the position table to the 24 × 16 grid and fine-tune; or crop or tile the photo into 224 × 224 pieces.
- 1 point: (a) numbers 385 and 148,225 (0.5), (b) 592,900 and about 0.69 percent (0.5; 0.25 if the head is forgotten: 589,824, 0.686 percent).
- 1 point: (c) names the problem (the table has too few positions, or the position embeddings do not match the grid) (0.5) and two valid ways out (0.25 each).
Typical errors: 384 tokens for the attention (forgetting the class token: 147,456, accept for 0.25 off); 24 + 16 instead of 24 · 16; counting 16 · 768 for the adapters (one matrix side only); r · 768 · 768; 197 tokens for the photo.

### B5.2 (Weeks 12 and 13), 2 points
Model answer. (a) The picture has 256 · 256 · 3 = 196,608 numbers and the latent 32 · 32 · 4 = 4,096, so the latent is 48 times smaller; the grid of positions is 65,536 against 1,024, 64 times fewer (the table of attention scores is 4,096 times smaller). (b) DDPM: 500 · 1,000 = 500,000 calls; guided DDIM: 500 · 25 · 2 = 25,000 calls, which is 20 times fewer; at 0.04 s a call that is 1,000 seconds, about 17 minutes (DDPM would take 20,000 seconds, about 5.6 hours). (c) A danger: the synthetic pictures may look unlike real early corrosion so the classifier learns the wrong cues (or the labels are wrong, or the pictures repeat); a person must check a sample, label them as synthetic and keep the test set real. For a GAN: mode collapse (only a few kinds of picture; check variety), and unstable training of the generator and discriminator; the transpose-convolution checkerboard artefacts.
- 1 point: (a) 48 and 64 (0.25 each); (b) the call counts 500,000 and 25,000 with the factor 20 (0.25), and the time 1,000 s or about 17 minutes (0.25).
- 1 point: (c) a valid danger (0.5) and a valid GAN check (0.5).
Typical errors: forgetting the factor 2 of guidance (12,500 calls, 40 times fewer); 1,000 calls per image for DDIM; using the channels only (3 / 4); saying that diffusion cannot be used because it is slow.

### B5.3 (Weeks 11 to 14): the design card, 2 points
Model answer (one possible card, 130 words). 1. Photo as tokens: a ViT (Week 11); 448/16 · 336/16 = 28 · 21 = 588 patches, 589 tokens with the class token. 2. Classifier: the ViT adapted with LoRA r = 8 on the query and value matrices (Week 14): 2 · 12 · 8 · 1,536 = 294,912 trainable parameters, plus the head. 3. Extra data: latent diffusion with DDIM (Week 13): 25 steps with guidance, 50 network calls per picture. 4. Summary writer: a language model with LoRA, SFT on engineers' example summaries and then DPO on chosen and rejected drafts (Week 14); watch the masked SFT loss. Evaluation: recall of the crack class on 200 real photos never used for training. Guardrail: a reward model can be hacked by long, confident summaries, so keep a large β (small KL to the reference) and let an engineer read samples. Decides: a licensed engineer makes every pass or fail decision.
Other correct choices are fine: CLIP zero-shot for step 1 (the number can be the image-text similarity or the 589 tokens of its ViT), a GAN or no generator for step 3 (then the number can be a transposed-convolution output size), full fine-tuning with 16 bytes per parameter for step 2, an RAG step (not in these weeks, accept) for the checklist items.
- 1 point: the four step lines: each has a model that fits the step, its week, and a number that is correct for the student's own choice (0.25 each; the 589 tokens and the LoRA count are checked against the formulas of Weeks 11 and 14).
- 1 point: one evaluation with a number and the data it is measured on (real, unseen photos) (0.5 together: 0.25 for the number, 0.25 for the real unseen data); one guardrail with its cause and its stop (reward hacking and β or a human check, hallucination and retrieval, synthetic data kept out of the test set) (0.25); the licensed engineer decides (0.25).
Typical errors: a card without numbers; an evaluation measured on synthetic or training photos (0 for that line); a guardrail that is only "be careful"; the assistant "approves" the structure (0 for the last line and a note to the instructor); a week number that does not match the model (Week 12 for diffusion, for instance).

**Task 5 total: 2 + 2 + 2 = 6.**

---

## Totals and pass mark
Task 3: 3 + 2 + 3 + 2 + 3 + 1 + 1 + 1 = 16. Task 4: 2 + 1 + 2 + 1 + 3 + 2 + 1 = 12. Task 5: 2 + 2 + 2 = 6. Together with Tasks 1 and 2 (12 and 14 points) Part B is 60; with Part A (40) the Pre-final is out of 100 and the pass mark is 70.

## Timing evidence
The reference solution runs the whole of Tasks 3 to 5 in about 3 seconds of CPU compute (the two denoiser trainings of B3.5 take about 1.5 seconds together, the LoRA run of B4.5 well under a second); in quick test mode (`NB_TEST=1`) it takes about 2.5 seconds, and every self-check is still passed. The slowest cell takes about 3 seconds, far below the limit of 90 seconds. The unmodified starter runs without error and every one of the 46 checks of Tasks 3 to 5 prints a cross. Writing time is the limit: the 30-minute boxes of Tasks 3 and 4 assume that the data, the networks and the helper functions are given, as they are.
