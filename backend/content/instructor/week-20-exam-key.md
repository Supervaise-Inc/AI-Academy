# Week 20 — Final Exam: instructor answer key and marking scheme

**Confidential: for instructors only.** The student notebook (notebooks/week-20.ipynb) is a public download and contains no answers; its self-checks compare against hashed fingerprints (label w20-b1). Part A runs in the app (exam-mode quiz id week-20-final, 40 questions, 1 point each). Part B (60 points) is marked by the instructor from the submitted notebook. The capstone Demo Day is marked separately (see week-20-demo-sheet.md).

---

# Instructor key: Part A (written exam), Week 20 Final

Not for students. 40 questions, 1 point each, 60 minutes, pass mark 70 percent of Part A as a quiz (28 of 40). The Final is cumulative: 6 questions on the Prelim (Weeks 1–5), 8 on the Midterm (Weeks 6–10), 8 on the Pre-final (Weeks 11–15) and 18 on the Final term (Weeks 16–19). Calculation answers were recomputed with Python by build_final.py, which asserts every correct value, that the four options of each question differ, the topic counts and the answer positions.

| # | Topic | Answer | Type | Level |
|---|---|---|---|---|
| 1 | Prelim: learning and CNNs (Weeks 1–5) | D | concept | recall |
| 2 | Prelim: learning and CNNs (Weeks 1–5) | C | calculation | apply |
| 3 | Prelim: learning and CNNs (Weeks 1–5) | B | calculation | apply |
| 4 | Prelim: learning and CNNs (Weeks 1–5) | A | calculation | apply |
| 5 | Prelim: learning and CNNs (Weeks 1–5) | D | concept | analyse |
| 6 | Prelim: learning and CNNs (Weeks 1–5) | A | concept | recall |
| 7 | Midterm: language models and Transformers (Weeks 6–10) | C | calculation | apply |
| 8 | Midterm: language models and Transformers (Weeks 6–10) | C | calculation | apply |
| 9 | Midterm: language models and Transformers (Weeks 6–10) | D | calculation | apply |
| 10 | Midterm: language models and Transformers (Weeks 6–10) | A | concept | recall |
| 11 | Midterm: language models and Transformers (Weeks 6–10) | B | concept | recall |
| 12 | Midterm: language models and Transformers (Weeks 6–10) | D | concept | recall |
| 13 | Midterm: language models and Transformers (Weeks 6–10) | B | calculation | apply |
| 14 | Midterm: language models and Transformers (Weeks 6–10) | A | calculation | analyse |
| 15 | Pre-final: images, generation and fine-tuning (Weeks 11–15) | D | calculation | apply |
| 16 | Pre-final: images, generation and fine-tuning (Weeks 11–15) | A | calculation | apply |
| 17 | Pre-final: images, generation and fine-tuning (Weeks 11–15) | B | concept | recall |
| 18 | Pre-final: images, generation and fine-tuning (Weeks 11–15) | D | calculation | apply |
| 19 | Pre-final: images, generation and fine-tuning (Weeks 11–15) | D | calculation | apply |
| 20 | Pre-final: images, generation and fine-tuning (Weeks 11–15) | A | concept | analyse |
| 21 | Pre-final: images, generation and fine-tuning (Weeks 11–15) | D | calculation | apply |
| 22 | Pre-final: images, generation and fine-tuning (Weeks 11–15) | B | concept | analyse |
| 23 | Final: RAG, agents, trust and deployment (Weeks 16–19) | A | calculation | recall |
| 24 | Final: RAG, agents, trust and deployment (Weeks 16–19) | B | concept | analyse |
| 25 | Final: RAG, agents, trust and deployment (Weeks 16–19) | C | calculation | apply |
| 26 | Final: RAG, agents, trust and deployment (Weeks 16–19) | C | concept | apply |
| 27 | Final: RAG, agents, trust and deployment (Weeks 16–19) | B | concept | recall |
| 28 | Final: RAG, agents, trust and deployment (Weeks 16–19) | A | calculation | apply |
| 29 | Final: RAG, agents, trust and deployment (Weeks 16–19) | A | calculation | apply |
| 30 | Final: RAG, agents, trust and deployment (Weeks 16–19) | C | calculation | apply |
| 31 | Final: RAG, agents, trust and deployment (Weeks 16–19) | B | concept | analyse |
| 32 | Final: RAG, agents, trust and deployment (Weeks 16–19) | A | concept | recall |
| 33 | Final: RAG, agents, trust and deployment (Weeks 16–19) | B | calculation | recall |
| 34 | Final: RAG, agents, trust and deployment (Weeks 16–19) | C | calculation | recall |
| 35 | Final: RAG, agents, trust and deployment (Weeks 16–19) | C | calculation | recall |
| 36 | Final: RAG, agents, trust and deployment (Weeks 16–19) | D | calculation | apply |
| 37 | Final: RAG, agents, trust and deployment (Weeks 16–19) | C | calculation | apply |
| 38 | Final: RAG, agents, trust and deployment (Weeks 16–19) | C | concept | analyse |
| 39 | Final: RAG, agents, trust and deployment (Weeks 16–19) | B | calculation | apply |
| 40 | Final: RAG, agents, trust and deployment (Weeks 16–19) | D | concept | analyse |

Topic counts: Prelim 6, Midterm 8, Pre-final 8, Final term 18

Level counts: recall 12, apply 20, analyse 8

Answer position counts: A 10, B 10, C 10, D 10

Calculation questions: 24

## Working for calculation questions

**Q2** (Prelim: learning and CNNs (Weeks 1–5)), correct C: 2.170

- Working: softmax(dog)=e/(e^3+e+1)=0.1142; -ln=2.1698. Distractors: the probability 0.114; -ln of the largest probability 0.170; -log10 0.942.
- Options: A 0.114; B 0.170; C 2.170; D 0.942

**Q3** (Prelim: learning and CNNs (Weeks 1–5)), correct B: 0.74

- Working: yhat=1.30; dL/dw=2(yhat-t)x=-0.80; w=0.740. Distractors: sign error 0.66; missing factor 2 0.72; error only 0.71.
- Options: A 0.66; B 0.74; C 0.72; D 0.71

**Q4** (Prelim: learning and CNNs (Weeks 1–5)), correct A: 10

- Working: conv = floor((42+2-5)/2)+1 = 20; pool = floor((20-2)/2)+1 = 10. Distractors: no padding 9; conv only 20; pooled twice 5.
- Options: A 10; B 9; C 20; D 5

**Q7** (Midterm: language models and Transformers (Weeks 6–10)), correct C: 0.20

- Working: 0.4*0.5=0.2. Distractors: 3/10 (wrong denominator) 0.30; sum 0.90; 0.4*0.3=0.12.
- Options: A 0.30; B 0.90; C 0.20; D 0.12

**Q8** (Midterm: language models and Transformers (Weeks 6–10)), correct C: 0.952

- Working: dot=20; |u|=3; |v|=7; cos=0.9524. Distractors: 20/(9+49)=0.345; 20/(3+7)=2.000; 20/42=0.476.
- Options: A 0.345; B 2.000; C 0.952; D 0.476

**Q9** (Midterm: language models and Transformers (Weeks 6–10)), correct D: 11.98

- Working: weights=[0.8438, 0.1142, 0.042]; out=11.9822. Distractors: unscaled 10.00; equal weights 20.00; divided by d_k instead of sqrt 17.46.
- Options: A 10.00; B 20.00; C 17.46; D 11.98

**Q13** (Midterm: language models and Transformers (Weeks 6–10)), correct B: 7 billion

- Working: active=3+2*2=7; total=35. Distractors: total 35; experts only 4; one expert 5.
- Options: A 35 billion; B 7 billion; C 4 billion; D 5 billion

**Q14** (Midterm: language models and Transformers (Weeks 6–10)), correct A: Lab A uses the optimal 20 tokens per parameter; Lab B uses 200 per parameter and ten times the compute, for a model smaller than optimal.

- Working: Lab A: D/N=20; Lab B: D/N=200; C_A=6ND=5.88e+21; C_B=5.88e+22; ratio 10.
- Options: A Lab A uses the optimal 20 tokens per parameter; Lab B uses 200 per parameter and ten times the compute, for a model smaller than optimal.; B Both labs are compute-optimal, because the Chinchilla rule fixes only the size of the model and says nothing about the number of tokens.; C Lab A is under-trained, because the Chinchilla rule asks for 200 tokens per parameter, a ratio that only Lab B reaches in this comparison.; D Lab B uses ten times the compute of Lab A, so the scaling law says that it must be ten times better on every one of the tasks.

**Q15** (Pre-final: images, generation and fine-tuning (Weeks 11–15)), correct D: 25

- Working: (180/30)*(120/30)+1=25. Distractors: 24 (no class token); 180*120/30=720; 6+4+1=11.
- Options: A 24; B 720; C 11; D 25

**Q16** (Pre-final: images, generation and fine-tuning (Weeks 11–15)), correct A: 0.80

- Working: p=0.8047. Distractors: scale 10 -> 0.61; no scale -> 0.36; plain share 0.28/0.60 -> 0.47.
- Options: A 0.80; B 0.61; C 0.36; D 0.47

**Q18** (Pre-final: images, generation and fine-tuning (Weeks 11–15)), correct D: 18

- Working: (6-1)*3-0+3=18. Distractors: 15 (forgot k); 21 (n*s+k); 12 (doubling).
- Options: A 15; B 21; C 12; D 18

**Q19** (Pre-final: images, generation and fine-tuning (Weeks 11–15)), correct D: 0.80

- Working: abar=0.9*0.8*0.5=0.36; noise weight=0.80. Distractors: abar 0.36; signal weight 0.60; 1-abar 0.64.
- Options: A 0.36; B 0.60; C 0.64; D 0.80

**Q21** (Pre-final: images, generation and fine-tuning (Weeks 11–15)), correct D: 3,145,728

- Working: 8*4096=32768; *4*24=3145728. Distractors: r*d only 1,572,864; full matrices 402,653,184; one matrix per layer 786,432.
- Options: A 1,572,864; B 402,653,184; C 786,432; D 3,145,728

**Q23** (Final: RAG, agents, trust and deployment (Weeks 16–19)), correct A: 10

- Working: ceil(2340/240)=ceil(9.75)=10. Distractors: 2400/300=8; floor 9; 2400/240+2=12.
- Options: A 10; B 8; C 9; D 12

**Q25** (Final: RAG, agents, trust and deployment (Weeks 16–19)), correct C: 4.608 GB

- Working: 1500000*768*4=4,608,000,000 bytes = 4.608 GB. Distractors: 2 bytes 2.304; 1 byte 1.152; 8 bytes 9.216.
- Options: A 2.304 GB; B 1.152 GB; C 4.608 GB; D 9.216 GB

**Q28** (Final: RAG, agents, trust and deployment (Weeks 16–19)), correct A: 9,000

- Working: n*C0 + s*n(n-1)/2 = 4800 + 150*28 = 9000. Distractors: linear growth 6,000; final context 1,800; n(n+1)/2 10,200.
- Options: A 9,000; B 6,000; C 1,800; D 10,200

**Q29** (Final: RAG, agents, trust and deployment (Weeks 16–19)), correct A: 26.9%

- Working: TP=180; FP=490; precision=0.2687. Distractors: recall 90.0%; FP taken from attacks only 94.7%; TP/FP 36.7%.
- Options: A 26.9%; B 90.0%; C 94.7%; D 36.7%

**Q30** (Final: RAG, agents, trust and deployment (Weeks 16–19)), correct C: 4,800 pesos

- Working: 0.003*2000*800=4800. Distractors: 0.3 read as 30% 480,000; 3% 48,000; 0.1% 1,600.
- Options: A 480,000 pesos; B 48,000 pesos; C 4,800 pesos; D 1,600 pesos

**Q33** (Final: RAG, agents, trust and deployment (Weeks 16–19)), correct B: 80 GB

- Working: 5e9*16=80 GB. Distractors: FP16 weights 10 GB; FP32 weights 20 GB; 8 bytes 40 GB.
- Options: A 10 GB; B 80 GB; C 20 GB; D 40 GB

**Q34** (Final: RAG, agents, trust and deployment (Weeks 16–19)), correct C: 4.5 GB

- Working: 2*15/16*2.4=4.50. Distractors: 2.4; 2*2.4=4.8; 16*2.4=38.4.
- Options: A 2.4 GB; B 4.8 GB; C 4.5 GB; D 38.4 GB

**Q35** (Final: RAG, agents, trust and deployment (Weeks 16–19)), correct C: 0.333

- Working: 5/15=0.3333. Distractors: 5/10=0.500; 6/16=0.375; 1/6=0.167.
- Options: A 0.500; B 0.375; C 0.333; D 0.167

**Q36** (Final: RAG, agents, trust and deployment (Weeks 16–19)), correct D: 0.003

- Working: scale=0.01; q=44; back=0.44; err=0.0030. Distractors: truncation to 43 gives 0.007; the scale itself 0.010; the weight 0.437.
- Options: A 0.007; B 0.010; C 0.437; D 0.003

**Q37** (Final: RAG, agents, trust and deployment (Weeks 16–19)), correct C: 100 tokens/s

- Working: 650e9/6.5e9=100. Distractors: FP16 25; INT8 50; FP32 12.5.
- Options: A 25 tokens/s; B 50 tokens/s; C 100 tokens/s; D 12.5 tokens/s

**Q39** (Final: RAG, agents, trust and deployment (Weeks 16–19)), correct B: 0.783

- Working: P=0.8182; R=0.7500; F1=0.7826. Distractors: precision 0.818; recall 0.750; accuracy 0.875.
- Options: A 0.818; B 0.783; C 0.750; D 0.875

---

# Instructor key: Week 20 Final practical exam (Part B), Tasks 1 and 2 (24 points)

Not for students. All numbers were computed by running the reference solution (seed 42, CPU; Tasks 1 and 2 together run in about 2 seconds, the same in quick mode). Reference notebook: `w20/p1/ref-P1.py`; every self-check prints a tick for it, and the unmodified starter runs without errors (all crosses). The self-checks accept the value rounded to the stated decimals (plus or minus 0.4 of the last digit) and never show it. Marks are in whole or half points (quarter points appear where a sub-question is worth 0.25). Rule for all code questions: correct method but wrong number earns half of the numeric points; a hard-coded number with no working earns at most half. No points are deducted for a wrong answer. Where a student's earlier answer is wrong but later work is right for that wrong input, give the later marks (own-error carried forward).

## Task 1. RAG: chunking, retrieval and its metrics (12 points)

The data (fictional Harbour Technical School handbook): 8 passages, 226 words when joined (`HANDBOOK`), 5 questions.

### B1.1 Chunking (2.5)
- Model answer: `s` runs over `range(0, len(w), size - overlap)`; take `w[s:s+size]`; stop after the chunk with `s + size >= len(w)`. L = 226. For the settings (40, 8), (60, 10), (30, 0), (50, 25): `n_chunks_list` = **[7, 5, 8, 9]**. Check: ⌈(226 − 8)/32⌉ = ⌈6.81⌉ = 7; ⌈216/50⌉ = ⌈4.32⌉ = 5; ⌈226/30⌉ = ⌈7.53⌉ = 8; ⌈201/25⌉ = ⌈8.04⌉ = 9. `last_len` for (40, 8): the 7th chunk starts at word 192 and has 226 − 192 = **34** words. `formula_ok = True`.
- Tolerance: list exact; `last_len` exact; Boolean True.
- Marking: `chunk_words` code 1; `n_chunks_list` 0.5; `last_len` 0.5; `formula_ok` 0.5. Typical errors: not stopping after the chunk that reaches the end, so an extra chunk made only of overlap words is appended (counts [8, 6, 8, 10], last_len 8): list 0, function 0.5, `formula_ok` False is then consistent (give 0.5 for it if the student's own comparison shows the mismatch honestly); stride `size` instead of `size − overlap` (counts 6, 4, 8, 5): 0; floor instead of ceiling in the formula (`formula_ok` False): 0.

### B1.2 A TF-IDF index (3.5)
- Model answer: `tokenize` = `[t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]`; `VOCAB` = sorted set of tokens of the 8 passages: `vocab_size` = **119**. `idf[w] = ln((1+8)/(1+df)) + 1`: library (df 2) = ln(3) + 1 = **2.0986**, school (df 3) = ln(9/4) + 1 = **1.8109**, fee (df 1) = ln(4.5) + 1 = **2.5041**. `vectorize` puts `tf × idf` in the slot of each word of VOCAB; `search` ranks by cosine, with ties kept in id order (`sorted` is stable). For the first question (tokens: much, late, fee, library, book) the scores are P0 0.446, P7 0.236, P3 0.101 and zero for the rest: `top3` = **[0, 7, 3]**, `top_score` = **0.446** (0.4462).
- Tolerance: `vocab_size` exact; `idf_vals` 3 decimals (the self-check mixes all three entries); `top3` exact list in order; `top_score` 3 decimals.
- Marking: code of the four pieces 1; `vocab_size` 0.5; `idf_vals` 0.75; `top3` 0.75; `top_score` 0.5. Typical errors: idf without the smoothing or with `ln(N/df)` (library 1.386, school 0.981, fee 2.079): 0 for `idf_vals`, half for the method if the formula is visible, and the later values then differ slightly (the ranking `top3` may still be right: give it); not removing stop words or splitting `wi-fi` differently (vocab 119 changes): 0 for `vocab_size` only if the student's tokenizer deviates from the stated rule; building `VOCAB` from the passages and the queries (a larger vocabulary): 0; scores without normalising by the vector length (a dot product): ranking can change, `top_score` 0; counting `df` by occurrences instead of passages: 0 for `idf_vals`.

### B1.3 Measuring the retriever (3)
- Model answer: the full rankings (ids best first) for the five questions: q0 [0, 7, 3, 1, 2, 4, 5, 6]; q1 [0, 1, 2, 3, 4, 5, 6, 7] (all scores are 0: "robot builders gather" shares no token with the robotics passage, so the ties keep the id order and the right passage P5 is at position 6); q2 [3, 4, 6, 0, 1, 2, 5, 7]; q3 [0, 1, 2, 3, 4, 5, 6, 7] (again all zeros: fever, exam, sit do not occur; the right passage P2 is at position 3 by luck of the id order); q4 [7, 0, 4, 2, 1, 3, 5, 6]. With k = 3: `P3_list` = **[0.3333, 0.0, 0.6667, 0.3333, 0.3333]**; `R3_list` = **[1, 0, 1, 1, 1]**; reciprocal ranks 1, 1/6, 1, 1/3, 1, so `mrr` = 3.5 / 5 = **0.700**. Mean recall@k for k = 1 to 6: 0.5, 0.6, 0.8, 0.8, 0.8, 1.0; the smallest k with at least 0.9 is `best_k` = **6**.
- Tolerance: `P3_list` and `R3_list` 3 decimals (arrays of five); `mrr` 3 decimals; `best_k` exact.
- Marking: three functions 0.5; `P3_list` 0.75; `R3_list` 0.75; `mrr` 0.5; `best_k` 0.5. Typical errors: precision divided by the number of relevant ids instead of k (swapping the two): 0 for both lists, 0.25 each if the definitions are visibly swapped; reciprocal rank counted from 0 (division by zero or 1/(pos)): 0 for `mrr`; `best_k` with mean precision instead of recall or ≥ 1 instead of ≥ 0.9 (6 either way, accept with a note that the threshold was 0.9 and the answer is the same); reading `best_k` as the first k where a question gets recall 1 (3): 0. Note for the marker: a ranking with ties depends on the rule "ties keep the order of the ids"; a student who sorts the other way around gets different rankings for q1 and q3: follow their rule through (own-error carried forward) if it is stated in the code.

### B1.4 The context budget (1.5)
- Model answer: `c_tokens` = 60 × 1.35 = **81.0**. `k_max` = ⌊(2048 − 150 − 300) / 81⌋ = ⌊1598 / 81⌋ = ⌊19.73⌋ = **19**. `tokens_k5` = 150 + 5 × 81 = **555**. `share_k5` = 555 / 2048 = **0.271**.
- Tolerance: `c_tokens` and `tokens_k5` 2 decimals; `k_max` exact; `share_k5` 3 decimals (27.1 % accepted if written as a share).
- Marking: 0.25, 0.5, 0.5, 0.25. Typical errors: rounding `k_max` to 20 instead of rounding down (a 20th chunk would not fit): 0; forgetting the reply (1,898 / 81 = 23): 0; including the reply in `tokens_k5` (855): 0, and `share_k5` 0.417 is then own-error carried forward (give 0.25).

### B1.5 Where retrieval fails (1.5)
- Model answer: `worst_q` = **1** (the robot builders question; its relevant passage P5 has reciprocal rank 1/6). Written answer, as an example of a full-mark answer (the wording will differ): the question says robot builders, which afternoon, the passage says robotics, club, meets, Tuesdays and 15:30. They share no token (TF-IDF has no stemming: robot is not robotics; and afternoon does not match 15:30), so every score is 0 and the order is the order of the ids. One change: dense embeddings or a hybrid (TF-IDF plus embeddings), stemming or query rewriting; measure recall@3 and MRR on the same five questions before and after (here 0.8 and 0.700). A person must still check that the retrieved passage really answers the question and that the answer repeats it faithfully (also the unanswerable case).
- Tolerance: `worst_q` exact. Written: see marking.
- Marking: `worst_q` 0.5. Written 1: (0.4) names the shared or missing words correctly and links the low rank to exact-word matching, (0.3) a sensible change and a measurement (recall@k, MRR, precision) named, (0.3) a human check that is more than a generic "verify". `worst_q` of 3 (the fever question, RR 1/3) with a correct account of why it ranks low: 0 for `worst_q`, written marks as usual (the explanation is also correct for q3). Typical errors: answering "more chunks fixes it" without a metric; blaming the model for a retrieval failure.

Task 1 total: 2.5 + 3.5 + 3 + 1.5 + 1.5 = 12.

## Task 2. Agents and guardrails (12 points)

The data: the toy model is a rule-based function that obeys an e-mail instruction found in the goal (first step) or in the last tool result; the tools are `calculator`, `lookup`, `send_email`; `ALLOWED_DOMAIN = "@harbourtech.example"`; the suite has 10 attacks (numbers 0 to 9) and 10 legitimate goals.

### B2.1 The agent loop (3)
- Model answer: loop `for step in 1..max_steps`: call the model; on `final` return the answer with `steps = step` and `stopped = "final"`; otherwise validate (if given), look up the tool (`"error: unknown tool"` if missing), run it, append `(reply, observation)`; after the loop return `"stopped: step limit"`, `steps = max_steps`, `stopped = "limit"`. The four demo goals use `steps_list` = **[2, 2, 6, 2]**: the calculator goal takes a call for the tool and a call for the final answer; the lookup goal the same; the "keep searching" goal never finishes and is cut at 6; the hallucinated `delete_all` gets an error observation and the model then gives up in a second call. `n_limit` = **1**. `calc_value` = 12 × 5 + 3 = **63.0**.
- Tolerance: `steps_list` exact; `n_limit` exact; `calc_value` 2 decimals (63).
- Marking: code 1 (the loop with a step limit, one model call per step, validate before the tool runs, unknown tool handled); `steps_list` 1; `n_limit` 0.5; `calc_value` 0.5. Typical errors: counting tool calls instead of model calls ([1, 1, 6, 1]): 0 for the list, half if the definition is visible; an `off-by-one` loop (`range(max_steps)` with `steps = step`: [1, 1, 5, 1]): same; no step limit (an endless loop: the cell never ends: code 0 and the later parts cannot run: give the numbers they can compute by hand); returning the observation instead of parsing the number (`calc_value` fails): 0.

### B2.2 A validator for tool calls (2.5)
- Model answer: order of the checks: tool in `TOOLS` → all values strings → tool-specific checks (calculator: allowed characters, then at most 40 characters; send_email: address ends with the allowed domain, then text at most 200 characters; lookup: topic at most 30 characters). On `CALLS`: 1 accepted, 2 `chars` (letters and underscores), 3 accepted, 4 `domain`, 5 `length` (300 characters), 6 `tool`, 7 accepted, 8 `type` (42 is not a string). `valid_flags` = **[1, 0, 1, 0, 0, 0, 1, 0]**, `n_rejected` = **5**, `reason_4` = **"domain"**, `reason_8` = **"type"**.
- Tolerance: list exact; `n_rejected` exact; the reasons are the exact words (case-insensitive).
- Marking: `validate_call` code 1; `valid_flags` 0.5; `n_rejected` 0.25; `reason_4` 0.25; `reason_8` 0.5. Typical errors: checking the length before the domain (call 5 is the only one affected: still `length`); a block-list instead of the stated allow-list of characters (call 2 may pass: 0 for the list entry; explain the lesson: allow-lists); returning `"not a string"` or other free text instead of the exact words: `reason_8` 0 but the flags are right; a validator that crashes on call 8 (calls `len` on 42): code 0.5 at most.

### B2.3 Rails and the attack suite (3)
- Model answer: `input_rail` = any of the four patterns in the lower-cased text; `output_rail` replaces `sk-` keys and 11-digit numbers starting with 09 by `[hidden]`; `guarded_agent` = input rail → `run_agent(goal, validate=validate_call)` → output rail on the answer. Results: unguarded `open_stopped` = **0** of 10 attacks (every attack succeeds against the gullible model). Guarded `guard_stopped` = **9** of 10: attacks 0, 1, 5 and 9 are blocked by the input rail; 2, 3 and 4 are stopped by the validator (external address); 6 and 7 are redacted by the output rail; attack 8 passes every layer (the helpdesk phone number is written with dashes, `0917-555-0142`, and the output rail's pattern only knows eleven digits in a row), so `leaked_ids` = **[8]**. `guard_served` = **8** of 10 legitimate goals: the goal with "ignore all my earlier typos" is blocked by the input rail (false positive) and the e-mail to a gmail address is refused by the validator (a legitimate goal that the allow-list does not allow).
- Tolerance: all exact; `leaked_ids` is a list (the self-check accepts only [8]).
- Marking: the three functions 1 (input rail case-insensitive, output rail substitution with both patterns, guarded agent in the right order); `open_stopped` 0.5; `guard_stopped` 0.5; `guard_served` 0.5; `leaked_ids` 0.5. Typical errors: a case-sensitive input rail (attacks 0, 1, 5, 9 are written with a capital: `guard_stopped` falls to 5 and the validator and output rail still stop 2 to 4, 6, 7, so 5: 0 for the entry, half for the idea); validator not passed to `run_agent` (the indirect attacks 3 and 4 and the direct 2 succeed: `guard_stopped` 6): 0; output rail applied to the goal instead of the answer: leaks 6 to 8 remain; applying `evaluate` to `guarded_agent` outputs without the `"answer"` key (a crash): code 0.5.

### B2.4 Layers, loops and cost (2)
- Model answer: (a) pass-through = 0.4 × 0.3 × 0.5 = **0.060** (6 of 100 attacks get through; the product is only right if the layers fail independently). (b) 0.5^n < 0.01 needs n = **7** (0.5^6 = 0.0156, 0.5^7 = 0.0078). (c) tokens read by 8 loops: 8 × 600 + 150 × 8 × 7 / 2 = 4,800 + 4,200 = **9,000**; for 16 loops: 9,600 + 150 × 120 = 27,600, so `ratio_16_8` = 27,600 / 9,000 = **3.07** (more than 2 because of the n(n − 1)/2 term). (d) 0.9^6 = **0.531**.
- Tolerance: `pass_through` 3 decimals; `layers_needed` exact; `read_8` exact; `ratio_16_8` 2 decimals; `p_six` 3 decimals.
- Marking: 0.5, 0.5, 0.5, 0.25, 0.25. Typical errors: adding the catch rates or averaging them; using 0.6 × 0.7 × 0.5 = 0.21 (the share stopped by all, not the pass-through): 0; `layers_needed` = 6 (rounding down, 0.0156 is not below 0.01): 0; using n·C0 + S·n (forgetting n − 1 and the 1/2): 0; ratio 2 (assuming linear growth): 0.

### B2.5 What your defence still lets through (1.5)
- Model answer (example of full marks, the wording will differ): attack 8 still succeeds because the number is written with dashes, the output rail's pattern needs eleven digits in a row, the input rail sees no injection words and the validator sees a harmless lookup; two legitimate goals were stopped (the sentence with "ignore all" and the e-mail to the gmail address), so 2 of 10 users are not served, which costs trust and staff time; a fix: also hide numbers with dashes (a wider pattern or a classifier) or let the validator send unknown domains to a person for approval instead of refusing; a person must decide which tools may send e-mail and approve irreversible actions (in a capstone: the licensed engineer makes every pass or fail decision).
- Marking: 0.5 names the surviving attack and the reason each layer missed it, using the student's own measurement (`leaked_ids`); 0.25 names the false positives and a cost; 0.5 a concrete fix tied to one of the two problems (a fix that hides everything or blocks more without discussing false alarms earns 0.25); 0.25 a human decision that is specific (not just "a human checks"). If the student's own `leaked_ids` differs from [8], mark the explanation for what their numbers show.
- The self-check only counts words (at least 40).

Task 2 total: 3 + 2.5 + 3 + 2 + 1.5 = 12. Tasks 1 and 2 together: 24 of the 60 points of Part B.

## Timing and names (for the integrator)
- The reference for Tasks 1 and 2 runs in about 2 seconds in total. Student time: 20 minutes each is realistic for a prepared student; Task 2 B2.3 is the longest piece.
- Names defined by the cells of Tasks 1 and 2 (Task 3 to 5 cells should avoid them): see the report of Agent P1.

---

# Week 20 Final Exam: instructor key, Part B Tasks 3 to 5 (36 of the 60 points of Part B)

**Confidential: for instructors only.** All numbers below were computed by running the reference solution (the same code that produced the fingerprints in the public notebook, label `w20-b1`). Self-checks only print a tick or a cross; the mark is given from the code, the numbers and the written answers. Tasks 3 to 5 are worth 12 + 16 + 8 = 36 of the 60 points of Part B (Tasks 1 and 2, 12 + 12, are keyed separately). Where a student's earlier answer is wrong but later work is right for the wrong input, give the later marks (own-error carried forward). Every sub-question of Tasks 3 and 4 has its own data, so an early mistake does not spoil a later part. Marks are given in steps of 0.25 where the notebook shows 0.25-point items (halve them if you prefer 0.5 steps). Checks of student-trained results (B4.2, B4.5) are thresholds, not exact values: a retrained model differs a little between machines.

---

## Task 3: Scaling and deploying (12 points)

### B3.1 Memory arithmetic (2 points)
Model answer. `weights_gb(n, b) = n * b` (billions times bytes gives GB). `mem_fp16` = [14, 26, 68, 140] GB; `train_gb_13` = 13 x 16 = 208 GB; `n_gpu_70` = ceil(140 / 48) = 3 (2.92); `biggest_int4_24` = 34 (INT4 weights 17 GB; the 70 B model needs 35 GB). Test values of the check: `weights_gb(3, 2)` = 6, `weights_gb(10, 0.5)` = 5.
- 0.5 the function; 0.5 `mem_fp16`; 0.5 `train_gb_13`; 0.25 `n_gpu_70` (2 earns 0.0: 96 GB cannot hold 140 GB); 0.25 `biggest_int4_24`.
Typical errors: 2 bytes for INT8 or 4 for FP16; 70 / 48 = 1.46 rounded to 1 or 2; 16 x 13 = 208 forgotten as "GB x 10^9"; choosing 70 (ignoring that it needs 35 GB) or 13.

### B3.2 The price of data parallelism (2 points)
Model answer. `ring_traffic(S, n) = 2(n - 1)/n * S`. For 8 GPUs: 2 x 7/8 x 6 = 10.5 GB; `comm_s` = 10.5 / 50 = 0.21 s; `eff` = 0.9 / (0.9 + 0.21) = 0.811; with 16 GPUs traffic is 11.25 GB, 0.225 s, `eff_16` = 0.800 (more GPUs hardly change it: the traffic per GPU is almost the same); `link_needed`: efficiency 0.9 needs comm <= 0.9 x (1/0.9 - 1) = 0.1 s, so 10.5 / 0.1 = 105 GB/s. Check test values: `ring_traffic(4, 4)` = 6, `ring_traffic(1, 2)` = 1.
- 0.5 function; 0.25 `traffic`, 0.25 `comm_s`; 0.25 `eff`; 0.25 `eff_16`; 0.5 `link_needed` (0.25 for the right method with a wrong earlier value carried forward).
Typical errors: `2 * n * S` or `S` alone (no 2(n-1)/n factor: 6 GB, 0.12 s, efficiency 0.882); `n * S`; efficiency = comm / (comm + compute) (0.189); link needed 10.5 / 0.9 = 11.7 (comparing comm with compute, not with a tenth of it).

### B3.3 The pipeline bubble (1.5 points)
Model answer. A GPipe schedule has `2(m + p - 1)` steps: `pipe_steps(p, m) = 2 * (m + p - 1)`; useful cells are 2mp of the p x steps cells, so `bubble = 1 - 2mp / (p * steps) = (p - 1)/(m + p - 1)`. Check values: `pipe_steps(4, 4)` = 14, `bubble(4, 4)` = 0.4286; `pipe_steps(3, 6)` = 16, `bubble(3, 6)` = 0.25. `bubble_4_8` = 3/11 = 0.2727; `bubble_4_32` = 3/35 = 0.0857; `m_min` for p = 6: 5/(m + 5) <= 0.10 gives m >= 45, so 45 (bubble 0.1000 exactly).
- 0.5 for the two functions together; 0.5 for the two bubbles (0.25 each); 0.5 for `m_min` (0.25 for 44 or 46, off by one).
Typical errors: forward only (steps m + p - 1, same bubble: accept if the bubble is right and `pipe_steps` is explained, 0.25 off); p x m in the denominator; bubble = 1/m; `m_min` = 50 (rounded up to a tidy number) or 5 / 0.1 = 50.

### B3.4 Tensor parallelism: split a matrix product (1.5 points)
Model answer. Column split: `np.concatenate([X @ W[:, :4], X @ W[:, 4:]], axis=1)`; row split: `X[:, :3] @ W[:3] + X[:, 3:] @ W[3:]`. Both equal `Y_full` (the matrices hold small whole numbers, so they are equal exactly); `equal_full` = True; `needs_sum` = "row" (the partial results of the row split must be added across the GPUs, an all-reduce; the column split only joins the pieces).
- 0.5 `Y_col`; 0.5 `Y_row` (each must be built from halves; a bare `X @ W` earns at most 0.25 each); 0.25 `equal_full`; 0.25 `needs_sum`.
Typical errors: splitting `X` by columns without splitting the rows of `W` to match; concatenating the row-split pieces instead of adding them; saying "column" for needs_sum.

### B3.5 Quantization from scratch (2 points)
Model answer. `scale = max|w| / (2^(bits-1) - 1)`, `q = round(w / scale)`, `restored = q * scale`. Check: `quantize_sym([0.5, -1.0, 0.25], 4)` gives scale 0.142857 and q = [4, -7, 2]. For `w_q` (the outlier 0.8 sets the maximum): `scale8` = 0.006299, `scale4` = 0.114286; `mae8` = 0.001624, `mae4` = 0.027198; with four groups of 32, each with its own scale, `mae4_grouped` = 0.008533 (about 3 times smaller, because only the group with the outlier has a large scale); `bits_grouped` = 4 + 16/32 = 4.5 bits per weight.
- 0.75 `quantize_sym` (and a working `dequantize`); 0.25 the two scales; 0.5 the two errors (0.25 each); 0.25 `mae4_grouped`; 0.25 `bits_grouped`.
Typical errors: dividing by 2^bits (or 2^(bits-1)) instead of 2^(bits-1) - 1; no rounding; using `abs(w).max()` of the wrong group; forgetting the scale storage (4 bits); 4 + 16 = 20 (the scale shared by all 32 weights).

### B3.6 Serving plan: decoding speed, KV cache and batching (2 points)
Model answer. Weights: 16 GB in FP16, 4 GB in INT4. `tps_fp16` = 273 / 16 = 17.06 tokens/s; `tps_int4` = 273 / 4 = 68.25 tokens/s (upper bounds). `kv_gb = 2 * layers * kv_heads * head_dim * bytes * tokens * batch / 1e9`: one 8,192-token conversation `kv_one` = 2 x 32 x 8 x 128 x 2 x 8,192 = 1.0737 GB; free memory 24 - 4 = 20 GB, so `n_conv` = floor(20 / 1.0737) = 18. Check values: `kv_gb(2, 4, 64, 100)` = 0.0002048, `kv_gb(1, 1, 8, 10, 4, 3)` = 0.00000192. Batching: `lat_16` = 0.060 + 0.004 x 16 = 0.124 s; throughput 16 / 0.124 = 129.0 requests/s against 1 / 0.064 = 15.6, so `gain` = 8.26.
- 0.5 the two bounds; 0.5 `kv_gb`; 0.25 `kv_one`; 0.25 `n_conv`; 0.5 `lat_16` and `gain` (0.25 each).
Typical errors: forgetting the factor 2 (keys and values), using the number of attention heads (32) instead of the 8 key-value heads (cache 4 GB, `n_conv` 4); FP16 weights (16 GB) subtracted for `n_conv` (3); bandwidth / parameters (not bytes); `gain` = 16 (ignoring the fixed T0).

### B3.7 Choose a plan (1 point, written)
Model answer. 13 B in FP16 = 26 GB, which does not fit on a 24 GB card; INT4 = 6.5 GB; per token the 8 B model stores about 0.131 MB, so 16 users x 4,096 tokens = about 8.6 GB (a 13 B model needs somewhat more), and 6.5 + 8.6 = 15.1 GB fits; quantize to INT4 and limit the conversation length (or batch with care); test the quality of the quantized model on the clinic's own questions.
- 0.25 says that FP16 does not fit, with 26 GB; 0.25 the INT4 size and the KV-cache number; 0.25 a sensible change (INT4/INT8, shorter context, fewer users, more GPUs) that fits the numbers; 0.25 a person tests quality (or safety) before relying on it.
Typical errors: saying FP16 fits "with offloading" without numbers; ignoring the KV cache; claiming that quantization has no effect on quality.

**Task 3 total: 2 + 2 + 1.5 + 1.5 + 2 + 2 + 1 = 12.**

---

## Task 4: Build and evaluate a small classifier (16 points)

### B4.1 Data and a baseline (2 points)
Model answer. `counts_train` = [715, 303, 182]; `majority_label` = 0 (clean); `majority_acc` = 369 / 600 = 0.615 (the test set has 369 clean, 149 stain and 82 crack tiles); `crack_recall_base` = 0.0 (a model that always says "clean" finds no crack at all); `pixel_mean` = 0.3195. The lesson is that a 61.5 % accuracy looks like something while the model is useless for the one class that matters.
- 0.5 `counts_train`; 0.25 `majority_label`; 0.5 `majority_acc`; 0.25 `crack_recall_base`; 0.5 `pixel_mean`.
Typical errors: baseline accuracy taken on the training set (0.595); `np.bincount` on the 64-column array; pixel mean of `Xte`; crack recall 0.15 (the prevalence) instead of 0.

### B4.2 Train the classifier (2.5 points)
Model answer. `torch.manual_seed(42)`; `nn.Sequential(nn.Linear(64, 24), nn.ReLU(), nn.Linear(24, 3))`; Adam lr 0.01; 300 full-batch steps of `F.cross_entropy`, with `loss.item()` appended each step; the loss falls from about 1.09 to about 0.10. Reference results: test accuracy 0.9567, crack recall 69 / 82 = 0.8415. The checks accept any run with accuracy >= 0.90, crack recall >= 0.70, a loss at the end below 30 % of the first, `probs_test` of shape (600, 3) with rows summing to 1, and values that agree with the stored probabilities.
- 1 the function (all four steps in the right order: `zero_grad`, forward and loss, `backward`, `step`); 0.5 `probs_test` (softmax applied; `torch.no_grad()` is welcome); 0.5 `acc_test`; 0.5 `crack_recall`.
Typical errors: forgetting `opt.zero_grad()` (loss still falls but is unstable); softmax inside the training loss (`F.cross_entropy` expects logits); a float label tensor; recall computed as precision; accuracy computed on `Xtr`.

### B4.3 The confusion matrix and per-class scores by hand (3 points)
Model answer. `confusion` counts `cm[true, pred]` (row = truth). Check: `confusion([0,1,1,2,2,2], [0,1,2,2,2,0], 3)` = [[1,0,0],[0,1,1],[1,0,2]]. On the given 40 tiles `cm_g` = [[21, 1, 1], [2, 3, 1], [0, 0, 11]]; `recall_g` = [0.913, 0.500, 1.000]; `precision_g` = [0.913, 0.750, 0.846]; F1 = [0.913, 0.600, 0.917] so `macro_f1` = 0.810; `acc_g` = 35 / 40 = 0.875. (For reference, the model of B4.2 gives [[364, 1, 4], [6, 141, 2], [5, 8, 69]], recalls 0.986, 0.946, 0.841.) Reading: the accuracy 0.875 hides that half of the stains are missed; the crack recall is perfect here but the precision is 0.846.
- 0.75 `confusion` (orientation matters: transposed matrix earns 0.25); 0.5 `cm_g`; 0.5 `recall_g`; 0.5 `precision_g`; 0.5 `macro_f1`; 0.25 `acc_g`. `class_scores` is marked together with the three lists (zero denominators handled).
Typical errors: transposed matrix (then recall and precision swap: recall [0.913, 0.750, 0.846]); using `sklearn` (no marks for those lines); macro F1 taken as the F1 of the macro precision and recall (0.817 differs slightly; accept only 0.810); a division by zero when a class is never predicted.

### B4.4 Choose a threshold by cost (3 points)
Model answer. `triage_cost` adds, for each tile, 5 if P(clean) < t, otherwise the cost of auto-passing the true class (0, 20 or 100). Check values: tiles [0.9, ...] true crack and [0.4, ...] true stain at t = 0.5 cost 100 + 5 = 105; one tile with P(clean) = 0.6 at t = 0.7 costs 5. For `THRESH` = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95] the costs are [285, 85, 100, 110, 140, 150]; `best_t` = 0.6; at 0.6, 17 of 30 tiles go to an engineer (`review_share` = 0.567). At 0.5 two cracks (P(clean) = 0.50 and 0.55) are auto-passed, which is why 0.5 costs 285; from 0.7 up, the extra reviews cost 5 each without saving another miss. Everything to the engineer would cost 150.
- 1 the function; 1 `costs` (0.5 if one or two values are wrong; the data have no tile exactly on a threshold except P(clean) = 0.5 at t = 0.5, which `>=` auto-passes); 0.5 `best_t`; 0.5 `review_share`.
Typical errors: comparing `P[:, 2]` (crack) instead of `P[:, 0]`; charging the review cost for every tile; reversing the rule (`<=`); `review_share` counted for the wrong threshold; choosing the largest threshold "to be safe" (150, the cost of reviewing everything).

### B4.5 A brightness shift (2 points)
Model answer. The first model sees every pixel 0.35 brighter and calls almost every tile a stain or a crack: `acc_bright` = 0.238 (the check accepts anything in (0, 0.7)). `normalise(X)` subtracts each row's mean (check: [[1, 2, 6], [4, 4, 7]] gives [[-2, -1, 3], [-1, -1, 2]]); a new model trained on normalised tiles reaches `acc_norm_bright` = 0.977 on the normalised bright test tiles (accepted: >= 0.93); `repaired` = True. The lesson: the shift is a covariate shift; a simple input normalisation that removes the nuisance makes the model robust to it, and the shifted test set must be part of the evaluation.
- 0.5 `acc_bright`; 0.5 `normalise` (`axis=1` and `keepdims`; subtracting the global mean of the batch earns 0.25); 0.5 `acc_norm_bright` (the new model must be trained on normalised data and tested on normalised bright data); 0.5 `repaired`.
Typical errors: normalising only the test set (the model was trained on raw pixels); `X.mean()` over the whole array: this also removes a constant shift of the whole test batch (accuracy about 0.975 in the reference run) but uses the statistics of the test batch and fails the check of `normalise` on two tiles with different means; give 0.25 and say why per-tile is the safer rule; training on the bright set; calling the result "robust" after one shift.

### B4.6 A gradient-sign step on a linear scorer (2 points)
Model answer. z0 = 1.2 x 0.9 - 0.8 x 0.1 + 0.5 x 0.4 + 2.0 x 0.6 - 0.3 = 2.10; p0 = sigma(2.10) = 0.891. `x_adv = x - 0.3 sign(w)` = [0.6, 0.4, 0.1, 0.3] (the second weight is negative, so that pixel goes up). z_adv = 2.10 - 0.3 x ||w||_1 = 2.10 - 0.3 x 4.5 = 0.75; p_adv = 0.679. `eps_flip` = z0 / ||w||_1 = 2.10 / 4.5 = 0.4667: every unit of eps lowers z by the sum of |w_i|, so the decision flips for any eps above 0.467 although each number changes by less than half a unit.
- 0.5 `z0` and `p0`; 0.5 `x_adv`; 0.5 `z_adv` and `p_adv`; 0.5 `eps_flip`.
Typical errors: `x + eps * sign(w)` (raises the score: z_adv = 3.45); `sign(x)` or `w` itself instead of `sign(w)`; forgetting the bias; eps_flip = z0 / max|w| = 1.05; sigmoid with the wrong sign.

### B4.7 What would you ship? (1.5 points, written)
Model answer (one possible, 137 words). "On my test set the classifier reaches 95.7% accuracy, but the better question is crack recall: it finds 69 of 82 cracks, 84.1%, so about one crack in six is still missed. I would not ship it as a decision maker. I would use it only to auto-pass tiles when P(clean) is at least 0.6, which was the cheapest rule in B4.4 and sends about 57% of the tiles to an engineer, and I would let a person see everything else. A brightness shift of 0.35 dropped the first model to 23.8% accuracy, while per-tile normalisation brought a new model back to 97.7%, so before the camera is used in the field I would add a test on shifted photos with normalisation in the pipeline. A licensed engineer makes the final pass or fail decision."
Students' numbers vary a little with their own run; mark against their own numbers.
- 0.5 a decision (ship, ship with limits, or do not ship) that uses the per-class number (crack recall), not only accuracy; 0.25 the auto-pass rule with a threshold and the share or cost from B4.4; 0.5 a test for the shift (from B4.5) or for other data change before field use; 0.25 a person (licensed engineer) decides, the model never signs off.
Typical errors: "ship because accuracy is 95%"; no numbers from the task; a threshold chosen without the cost; the assistant approves the structure (0 for the last point and a note to the instructor).

**Task 4 total: 2 + 2.5 + 3 + 3 + 2 + 2 + 1.5 = 16.**

---

## Task 5: Cumulative synthesis (8 points)

Marking: B5.1 is checked by the self-checks and the code; B5.2 to B5.4 are read and marked from the lines below. The Structural-Inspection rule applies to every written answer: a licensed engineer (a person) makes every pass or fail decision; an answer that lets the assistant sign off on safety loses the point for it.

### B5.1 Five computations (2 points, 0.4 each)
Model answer. (a) softmax of [2.0, 1.0, 0.1] gives p = [0.659, 0.242, 0.099]; `ce` = -ln 0.242 = 1.417. (b) scores / sqrt(16) = [2, 1, 0.5]; `att_w` = [0.6285, 0.2312, 0.1402]. (c) `block_params` = 4d^2 + 8d^2 = 12 x 768^2 = 7,077,888. (d) `abar_3` = 0.9 x 0.8 x 0.7 = 0.504. (e) C = 6 x 3e9 x 60e9 = 1.08e21 operations; at 2e14 per second that is 5.4e6 s = `train_days` = 62.5 days.
- 0.4 each; tolerance 3 decimals for (a), (b), (d); exact for (c); one decimal for (e).
Typical errors: log10 or the probability instead of -ln (0.242); no scaling in (b) (softmax of [8, 4, 2] = [0.98, ...]); 8d^2 only or 12d (c); sum of the betas, 1 - 0.6 = 0.4, instead of the product (d); forgetting the 6 or counting hours (e).

### B5.2 Context budget (Weeks 8 and 16), 2 points
Model answer. (a) 4,096 - 300 - 60 - 450 = 3,286 tokens for chunks; 3,286 / 240 = 13.7, so 13 chunks (3,120 tokens; the prompt uses 3,930 of 4,096). (b) 20 chunks need 4,800 tokens, more than the 3,286 available, so the prompt is cut, the call fails, or the answer is squeezed; more text also dilutes the relevant chunk (the model attends worse to the middle) and costs more (the attention cost grows with the square of the length). Try fewer chunks (top-5), re-ranking, smaller chunks, or compressing. (c) Retrieval fails when the right passage is not in the index, the chunking splits the answer, or the query and document use different words; a longer context only holds what was retrieved.
- 0.5 (a) 13 (0.25 for 14 from rounding, or for a correct method with one missing term); 0.5 (b) the overflow with numbers (4,800 > 3,286) and the effect; 0.5 two valid remedies; 0.5 (c) one valid retrieval failure that a longer context cannot repair.
Typical errors: 4,096 / 240 = 17 (forgetting the reserved tokens); counting the question but not the system prompt; "a longer window fixes it"; "use a bigger model".

### B5.3 The refund tool, memory and LoRA (Weeks 14, 17 and 18), 2 points
Model answer. (a) Two layers, for example: an argument validator (the amount and order number must be valid and within a limit) before the tool runs; a human approval or output rail for refunds above a limit; a step limit and a log. Each stops something named (an invalid or injected argument, a large unchecked refund, a runaway loop). (b) 7 B x 2 bytes = 14 GB in FP16: the weights fit on a 24 GB card but leave 10 GB for the KV cache, activations and the framework; INT4 would need 3.5 GB. (c) One adapter has r x (d_in + d_out) = 8 x 8,192 = 65,536 numbers, 65,536 / 16,777,216 = 0.39 % of the matrix.
- 0.75 (a) two layers, each with what it stops (0.25 for a layer without its purpose); 0.75 (b) "fits", 14 GB, and an INT8/INT4 size (7 or 3.5 GB) if more room is needed; 0.5 (c) 65,536 and 0.39 %.
Typical errors: "the model will be told to be careful" as a layer; 7 GB or 28 GB in (b); r x d x d = 524,288 (counting a full matrix); 8 x 4,096 = 32,768 (one side only); LoRA share quoted as a share of the model.

### B5.4 The launch note (Weeks 1 to 19), 2 points
Model answer (one possible, 117 words). "What it is: a 7 B language model (Weeks 6 to 9) with retrieval over our help pages (Week 16), a refund tool inside a guarded agent loop (Week 17), served in INT4 on one GPU (Week 18). Number: recall@5 of the retrieval, measured on 200 real, unseen customer questions, with the target agreed in advance. Risk: a prompt injection could make the agent refund money; the argument validator, the refund limit and a human approval stop it. Shift test: run the same questions after the help pages are updated and with a different wording style, and compare the number. Who decides: a named person at the company, not the assistant, decides whether the pilot goes on."
- 0.4 what it is, with at least three course weeks that match the parts; 0.4 one number with the data it is measured on (real, unseen); 0.4 one specific risk with its stop; 0.4 a test for a data shift; 0.4 a person decides.
Typical errors: weeks that do not match (diffusion for retrieval); a number measured on training or synthetic data (0 for that line); a risk that is generic ("it might be wrong"); fewer than 70 or more than 140 words is not penalised if all five lines are there.

**Task 5 total: 2 + 2 + 2 + 2 = 8.**

---

## Totals and pass mark
Task 3: 2 + 2 + 1.5 + 1.5 + 2 + 2 + 1 = 12. Task 4: 2 + 2.5 + 3 + 3 + 2 + 2 + 1.5 = 16. Task 5: 2 + 2 + 2 + 2 = 8. Together with Tasks 1 and 2 (12 and 12 points) Part B is 60; with Part A (40) the Final is out of 100 and the pass mark is 70. Time boxes: Task 3 20 minutes, Task 4 40 minutes, Task 5 20 minutes.

## Timing evidence
The reference solution runs the whole of Tasks 3 to 5 in about 3.3 seconds of CPU compute on this machine (the two network trainings of B4.2 and B4.5 take about 1.0 second each; no other cell takes more than 0.2 seconds); in quick test mode (`NB_TEST=1`) it takes the same time, because nothing in these tasks is shortened, and every self-check is still passed (64 of 64). The slowest cell takes about 1.4 seconds, far below the limit of 90 seconds. The unmodified starter runs without error and every one of the 64 checks of Tasks 3 to 5 prints a cross. Writing time is the limit: the 40 minutes of Task 4 assume that the data generator and the network are given, as they are; the three written answers of Task 5 need about 4 minutes each. Learners' trained models differ slightly from the reference between machines: the checks of B4.2 and B4.5 use thresholds for that reason.
