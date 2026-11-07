# Week 20 — Demo Day: instructor marking sheet

> **Confidential: for instructors only. Not served to students.** This sheet supports the capstone Demo Day. It uses only the criteria and weights fixed in the course outline. **You decide the marking scale, the dates, the order of teams, the slot length, the format and how the capstone is submitted and counted**; nothing here announces them. The capstone is 20% of the final grade according to the demo schedule.

## 1. Criteria, weights and level descriptors

Weights: Problem and proposal 15 % · Technical implementation 40 % · Responsible AI 15 % · Presentation 20 % · Documentation 10 % (total 100 %). The descriptors are plain-language guides at three levels; they carry no point values. Choose your own scale and write the score in the notes table.

| Criterion (weight) | Strong | Adequate | Weak |
|---|---|---|---|
| Problem and proposal (15 %) | A named user and problem, a clear scope, a stated list of what the assistant will not do; the audience can repeat the problem in one sentence; the demo matches the proposal | The problem is understandable but general; scope is partly stated or has drifted from the proposal without comment | Vague goal; no scope; the demo solves a different problem from the proposal |
| Technical implementation (40 %) | Pipeline works end to end on a photo chosen on the spot (or a clearly shown recording); tests shown and explained: confusion matrix, per-class recall, latency, a threshold chosen by cost; the team explains its own design and can answer why | Pipeline works on prepared examples; some tests but not per class or not explained; some answers rely on one member | Works only on one hidden example or not at all; no tests or only accuracy; the team cannot explain the parts |
| Responsible AI (15 %) | The AI-use rule built in and stated aloud; a real engineer step with an audit log; rails with tests; one honest failure with its fix; limits and data written in the model card; privacy handled (no real people's photos or data) | Rule stated and an engineer step shown, but rails or tests are thin; failures are mentioned but not shown; limits partly written | Rule absent or broken (the assistant prints a verdict or claims safety); no failures admitted; real personal data shown |
| Presentation (20 %) | A clear story in the given time; all three roles speak; numbers explained in plain words; calm, evidence-based answers to questions; the five-part structure is easy to follow | Understandable and mostly on time, but unevenly shared or too technical in places; questions answered but without evidence | Hard to follow, reading from the screen, far over or under time; one person presents; defensive or evasive answers |
| Documentation (10 %) | README, model card, failure log and data statement are ready to open; a teammate could run it from the README; numbers agree across slides and documents; AI assistance disclosed | Files exist and are mostly current; small mismatches between slides and documents; disclosure brief | Files missing, empty or out of date; numbers disagree; no disclosure of AI assistance |

## 2. Observation checklist: the AI-use rule and responsible-AI evidence

Tick during the demo. Missing items are not automatically a penalty; they are prompts for your feedback and for your question time.

- [ ] The team states the AI-use rule aloud: a licensed engineer makes every pass or fail decision; the assistant never signs off on safety.
- [ ] The human step is visible on screen (Decide, audit log, name of the engineer or librarian role).
- [ ] No output claims that a structure is safe or sound; if one appears, the team logs it as a failure.
- [ ] The DRAFT label and "review required" wording are visible on the output.
- [ ] At least one real failure is shown honestly, with cause, fix and a test that keeps it fixed.
- [ ] Rails are shown working, not only described (input rail, output rail, fixed reply to a verdict request).
- [ ] Results are reported per class and by source or condition, not only as accuracy; the small test set is acknowledged.
- [ ] A threshold is justified by cost or risk.
- [ ] Model card and failure log are opened when asked and match what was said.
- [ ] No real people's photos or personal data are on screen; synthetic, public or consented images only; secrets are not visible.
- [ ] Recordings and synthetic items are labelled as such.
- [ ] The team discloses any AI assistance used in the project, as the course rules require.
- [ ] Limits and next steps are specific and prioritised.

## 3. Notes per team

Use one row per team. Write the scale you have chosen in the header.

| Team / project | Date and slot | Problem and proposal (15 %) | Technical implementation (40 %) | Responsible AI (15 %) | Presentation (20 %) | Documentation (10 %) | AI-use rule: said aloud / shown? | Failure shown? | Notes and questions asked |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

## 4. How to give feedback

- Give feedback in the **same five criteria** the students have seen, so they know where the marks came from.
- Start with one **specific strength** ("the live run used a photo the audience chose"), then the **one or two changes that matter most**, then a **question** to think about. Quote the moment, not the person.
- Point to **evidence** the team showed or did not show (a confusion matrix, a failure log entry), not to impressions.
- For Responsible AI, name the rule or rail you saw; if a rule was broken (for example a verdict printed by the assistant), say plainly why that matters, even if the rest was strong.
- Where a member spoke little or answered nothing, say so neutrally and ask how the team shared the work, before deciding anything.
- Keep the comments short enough to be read: about five sentences per team, delivered in the form and at the time you announce.
- Peer feedback, if you collect it, uses the student form (what worked, one question, one suggestion, the responsible-AI choice noticed). Decide whether it is collected, anonymous or counted; it is not required by the course.

## 5. Reminder: what you decide

The course outline fixes only the five criteria, their weights, the three team roles, the AI-use rule and that the capstone is 20 % of the final grade according to the demo schedule. **You decide and announce**: the date, order, slot length, format (live, recorded, slides), room, whether questions sit inside the slot, how the capstone is submitted, the marking scale, how late or missing teams are handled and how marks are returned. The Demo Day section of the Week 20 page deliberately uses an example 7-minute slot only to show arithmetic: replace it with your real rules when you announce them.
