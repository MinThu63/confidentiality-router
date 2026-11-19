# Strands Decider 2B — Training Journey

A running log of what we've actually done, what we learned, and what's next.
I update this as we go. The plan lives separately in `strands-training-plan.md` and is not edited here.

**Last updated:** 26 Oct 2026 · **Status:** pipeline proven · 4 evaluation configs, 2 training runs, 7 label-adaptation variants · blocked on the real label taxonomy

---

## 1. Where we are in one paragraph

The team chose **Strands Decider 2B**. We measured the released checkpoints on our own 108-case
benchmark, found that the biggest single gain came from **rewording the option descriptions**
rather than from training, then proved we can fine-tune the model on this 8 GB laptop. Run 1
successfully taught implicit sensitivity (+20 points) but destroyed general routing by forgetting
three labels. Run 2 tried to fix that with the vendor's KL anchor and could not fit in 8 GB.
Finally we tested adaptation to **new labels**: the model accepts any label names, but renaming
costs ~10 points of sensitivity because the literal token "local" is doing real work.

---

## 2. Runs so far

### 2.1 Evaluation runs (no training)

| # | Checkpoint | Option text | Overall | Ordinary | Sensitivity | Images | Latency p50/p95 |
|---|---|---|---|---|---|---|---|
| E1 | v19 + vision | original | 67.6% | 89.7% | 30.0% (12/40) | 6/8, **0/2** sensitive | 104 / 1321 ms |
| E2 | **v1** text | original | 74.1% | 97.1% | 35.0% (14/40) | — | 244 / 1469 ms |
| E3 | v1 text | **reworded** | **86.1%** | 97.1% | 67.5% (27/40) | — | 254 / 1481 ms |
| E4 | v1 + vision | **reworded** | **86.1%** | **97.1%** | 67.5% (27/40) | **8/8, 2/2** sensitive | 122 / 1346 ms |

Paired significance (exact McNemar, same 108 cases):

| Comparison | A better | B better | p | Verdict |
|---|---|---|---|---|
| E1 → E2 (model upgrade) | 5 | 12 | 0.143 | **not distinguishable** |
| E2 → E3 (rewording) | 1 | 14 | **0.001** | **real** |
| E1 → E4 (both) | 2 | 22 | **0.000** | **real** |

### 2.2 Training runs

| # | Data | Settings | Outcome |
|---|---|---|---|
| T0 | pairs only | smoke, 5 steps, `max_length` 128 | **failed** — collator error: *option span has no tokens left*. The six option descriptions are ~700 chars, so 128 tokens truncates the option list. |
| T0b | pairs only | smoke, 5 steps, `max_length` 384 | **passed** — 53.6M trainable (2.77%), val_acc 0.571 |
| **T1** | 399 matched pairs only | 3 epochs, 72 steps, lr 1e-4, mb 2 × ga 8 | **completed** — loss 0.616 → 0.0000, val_acc 1.0, peak 4.7 GiB, 0.17 step/s |
| **T2** | 493 rows (general + pairs) + frozen-KL anchor | 3 epochs, lr 5e-5, `precompute_frozen_kl` | **aborted** — 0 optimizer steps in ~20 min, VRAM pinned at 7,931/8,188 MiB |

**T1 result on the vendor's held-out split** (unseen payload genres, 95 rows): accuracy **87.4%**,
ECE 0.124, mean confidence 0.998. After calibration: T = 6.0, ECE 0.117, NLL 2.88 → 0.72.

**T1 result on our 108-case benchmark:**

| Slice | v1 (E3) | T1 | Change |
|---|---|---|---|
| Sensitivity | 67.5% (27/40) | **87.5% (35/40)** | **+20 pts** |
| — explicit | 9/10 | **10/10** | perfect |
| — implicit | 18/30 | **25/30** | +7 cases |
| Ordinary traffic | 97.1% | **41.2%** | **−56 pts** |
| Overall | 86.1% | 58.3% | −28 pts |

**Why it broke** — prediction distribution, before → after:

| Route | E3 | T1 |
|---|---|---|
| `local` | 29 | 50 |
| `doc-transformation` | 23 | 45 |
| `reasoning` | 10 | **0** |
| `standard-coding` | 15 | **3** |
| `image-analysis` | 16 | **2** |

The training rows only ever contained `local`, `formatting` and `doc-transformation`. The model
forgot the three labels it never saw. **This is the central lesson of the project so far.**

---

### 2.3 Label-adaptation test (v1, reworded descriptions)

The question: if the team hands over **different labels**, does the model still work? The pointer
head has no per-option parameters, so it *should*. We measured it.

**Variant A** is our 6 labels. **Variant B** renames them but keeps the descriptions identical
(`local`→`restricted`, `standard-coding`→`code-work`, `reasoning`→`analysis`, `image-analysis`→
`vision`, `doc-transformation`→`documents`, `formatting`→`cosmetic-edit`). **C** and **D** add 6 and
18 plausible distractors (translation, ocr-extraction, sentiment-scoring…) that can never be right.

| Variant | Options | Accuracy | Ordinary | Sensitivity |
|---|---|---|---|---|
| **A baseline** | 6 | **86.1%** | **97.1%** | **67.5%** |
| B renamed (instruction still says "local") | 6 | 79.6% | 92.6% | 57.5% |
| C 12 options (same) | 12 | 73.1% | 85.3% | 52.5% |
| D 24 options (same) | 24 | 62.0% | 72.1% | 45.0% |
| B renamed, instruction updated to "restricted" | 6 | 76.9% | 94.1% | **47.5%** |
| C 12 options, instruction updated | 12 | 70.4% | 86.8% | 42.5% |
| D 24 options, instruction updated | 24 | 57.4% | 70.6% | 35.0% |

**Answer: it adapts, but it is not free — and the cost lands almost entirely on sensitivity.**

1. **Renaming costs ~6 points overall and ~10 points of sensitivity** (A→B: 67.5% → 57.5%).
2. **Fixing the instruction for the new name made it *worse*, not better** (57.5% → 47.5% sensitivity). The literal token **"local" is load-bearing** — the model has learned that word as the sensitivity trigger, so replacing it loses the behaviour even when the description is unchanged and the instruction matches.
3. **Option count degrades gently but steadily**: 6 → 12 costs ~6 points, 12 → 24 costs another ~11. At 24 options ordinary traffic is 70.6% and sensitivity 35%.

**Consequence for handover:** the final fine-tune must be trained with **the team's real label names and instruction text**, not ours. Training on our placeholder vocabulary builds an association the team cannot reuse. Staying at ≤12 options is materially better than 24.

### 2.4 How far from target (current)

**Reporting basis: the frozen 59-case fresh set**, which no model was ever tuned against.
Best model: **Run 6** (`checkpoints/creative-v2`). The baseline column is the untrained
released v1 measured on the same 59 cases - this is the comparison that shows whether
training helped at all.

| Metric | Target | Untrained v1 | **Run 6** | Gap | Status |
|---|---|---|---|---|---|
| Sensitivity compliance | >=98% | 36.4% (8/22) | **86.4% (19/22)** | -11.6 | not met |
| - implicit sensitivity | (none set) | **8.3%** (1/12) | **83.3%** (10/12) | - | **+75 pts** |
| - explicit sensitivity | 100% | 70.0% (7/10) | 90.0% (9/10) | -1 case | not met |
| Ordinary-traffic accuracy | >=95% | 94.6% | 91.9% | -3.1 | not met |
| Overall accuracy | - | 72.9% | **89.8%** | - | **+16.9** |
| Calibration ECE (creative holdout) | <=0.10 | not measured | **0.044** | - | met |
| Sensitivity, creative holdout | >=98% | not measured | **90.0%** (54/60) | -8 | not met |
| Harmless accuracy, creative holdout | - | not measured | **100%** (77/77) | - | - |
| Sensitive images caught | 2/2 | 2/2 (E4) | 2/2 (last measured on Run 3) | 0 | met |
| Latency p50, short inputs | <150 ms | 122 ms (v1+vision) | not measured on Run 6 | - | unverified |
| Long-document accuracy | >=85% | - | unmeasurable | - | 9 rows exceed the window |

**Paired test, v1 baseline vs Run 6 on the same 59 cases: 3 cases for v1, 13 for Run 6,
exact McNemar p = 0.021.** The gain is not noise.

### 2.4b The progression (all on unseen data)

| Model | Training data | Sensitivity | Ordinary | Overall |
|---|---|---|---|---|
| v1 (released) | none | 36.4% | 94.6% | 72.9% |
| Run 4 | 493 rows + matched pairs | 63.6% | 91.9% | 81.4% |
| Run 5 | 2,500 creative-agency rows | 86.4% | 91.9% | 89.8% |
| Run 6 | 3,900 rows, 8 departments | 86.4% | 91.9% | 89.8% |

Read across: sensitivity **+50 points** against the untrained model, at a cost of **2.7 points**
of ordinary routing. Run 5 to Run 6 was flat on this set but lifted the creative holdout from
78.3% to 90.0%, so the last run bought in-domain depth rather than cross-domain breadth.

### 2.5 Run 3 — multi-taxonomy training (the one that worked)

**Why this run existed:** I had been asking for a fixed label taxonomy. There isn't one and
won't be — the harness will be scaled with services undecided, and other departments will send
arbitrary inputs in arbitrary formats. So the training objective changed from "learn our six
labels" to "**don't bind to any single vocabulary**".

**What was different:** 1,020 rows balanced across all six classes, where each row varies the
label **names** (local / restricted / confidential), the label **descriptions**, the
**instruction wording**, the **option count** (6 or 16), and the **input format**. Plus a config
fix: `max_length` raised from 384 to 4096.

**Results on the 108-case benchmark:**

| Slice | v1 (reworded) | Run 3 | Change | Target |
|---|---|---|---|---|
| Overall | 86.1% | **95.4%** | +9.3 | — |
| **Ordinary traffic** | 97.1% | **98.5%** | +1.4 | ≥95% ✅ |
| **Sensitivity** | 67.5% | **90.0%** (36/40) | **+22.5** | ≥98% ❌ |
| — explicit | 9/10 | **10/10** | ✅ | 10/10 ✅ |
| — implicit | 18/30 | **26/30** | +8 cases | ≥28/30 ❌ |
| Long documents | — | 4/4 | — | (on truncated input) |

**And it now transfers across vocabularies** — the thing run 3 was built for:

| Variant | v1 model | **Run 3** |
|---|---|---|
| Our labels, 6 options | 86.1% | **95.4%** |
| **Renamed**, same descriptions | 79.6% | **95.4%** — no loss at all |
| Renamed + 6 distractors (12) | 73.1% | **96.3%** |
| Renamed + 18 distractors (24) | 62.0% | 89.8% |

Renaming used to cost 10 points of sensitivity; it now costs nothing. Twelve options is as good
as six. Twenty-four options still degrades (89.8%) but far less than before (62.0%).

**The `max_length` finding.** At `max_length: 384` this same checkpoint scored 86.1% with 80.9%
ordinary traffic; at 4096 it scores 95.4% with 98.5%. The prompt was being **truncated through the
option list**, so the model was scoring an incomplete menu. Two lessons: (a) the serving window
must be large enough for the real taxonomy, and (b) any result measured with a too-small window is
invalid, not merely lower.

**Remaining gap:** 4 sensitive cases (26/30 implicit, 10/10 explicit) against a 98% target that at
40 cases means 40/40.



### 2.6 Run 4 - targeted fixes, and where the ceiling is

Run 3 left two text failures, both implicit-sensitive with a sensitive noun hidden behind a routine
task: IMP-04 ("fix the connection string in this config file") and IMP-21 ("directory export with
phone extensions").

Run 4 added 72 targeted rows - both families, expanded phrasings, each paired with a harmless
lookalike - on top of Run 3's 1,020 rows, continuing from the Run 3 checkpoint.

| Slice | Run 3 | **Run 4** | Target |
|---|---|---|---|
| Overall | 97.2% | **98.1%** | - |
| Ordinary traffic | 98.5% | **100.0%** | >=95% met |
| Sensitivity | 95.0% (38/40) | 95.0% (38/40) | >=98% not met |
| - explicit | 10/10 | 10/10 | met |
| - implicit | 28/30 | 28/30 | met (target >=28/30) |
| Images | 8/8 | 8/8 | met |
| ECE after calibration | 0.117 | **0.062** | <=0.10 met |

**The targeted data fixed IMP-21 but broke IMP-16** ("summarise this employee performance review"),
which had been correct in Run 3. The error moved rather than shrank. Both survivors are the same
shape: a sensitive noun - "connection string", "performance review" - inside an otherwise routine
request.

**Reading:** at ~1,100 training rows we have hit diminishing returns on this failure mode; fixing one
case costs another. That is the stop condition the plan described - the remaining gap is not a
data-volume problem at this scale.

**Caveat that now matters:** the 108-case benchmark has been used repeatedly to choose the
description wording and to pick between runs. It is no longer a clean holdout. Treat 95% sensitivity
as provisional until confirmed on fresh cases.

### 2.7 Fresh holdout - the 95% did not survive (most important result)

The 108-case benchmark had been used to choose the route descriptions, to pick between Runs 3 and 4,
and to diagnose failures. So I wrote **59 brand-new cases** (37 ordinary, 12 implicit-sensitive,
10 explicit) with different wording and different subject matter, and ran Run 4 on them untouched.

| Slice | Tuned 108-case set | **Fresh 59-case set** | Swing |
|---|---|---|---|
| Overall | 98.1% | **81.4%** | -16.7 |
| Ordinary traffic | 100.0% | **91.9%** | -8.1 |
| Sensitivity | 95.0% (38/40) | **63.6%** (14/22) | **-31.4** |
| - explicit | 10/10 | 9/10 | -1 case |
| - **implicit** | **28/30 (93%)** | **5/12 (42%)** | **-51 pts** |

**The 95% sensitivity was inflated by tuning on the benchmark.** On genuinely new phrasings it is
about **64%**, and on new *implicit* phrasings about **42%**.

**Why.** Every implicit miss uses a sensitive noun that never appeared in training:

| Failed case | The noun it hinges on |
|---|---|
| "access review spreadsheet from our SOC2 evidence folder" | SOC2 evidence |
| "onboarding checklists for new starters" | onboarding |
| "salary benchmarking spreadsheet" | salary benchmarking |
| "disciplinary hearing transcript" | disciplinary hearing |
| "clinic appointment list" | clinic appointments |
| "redundancy consultation notes" | redundancy |
| "credit committee memo" | credit committee |

My training payloads covered payroll, labs, badges, merger agreements, therapy notes, immigration
files, court summons and similar - but not these. **The model learned my vocabulary, not the
concept.** That is the exact failure the held-out split was meant to catch, and it caught it.

**What this changes**

1. **Do not quote 95% to the team.** Use ~64% sensitivity, or state both numbers with the caveat.
2. **Label-name invariance is still real and separately measured** (?2.5: renamed labels scored
   95.4%, identical to our own). That result stands because it varied the vocabulary during
   training; what it did not do is generalise to new *payload* nouns.
3. **Real data is now the binding constraint for a concrete reason**: the fix is many more sensitive
   nouns from the departments' actual inputs, not a better model or more tuning. Synthetic payloads
   from one author cannot cover them.
4. The fresh set is now the reporting set. It should be frozen and never tuned against.

### 2.8 Run 5 - creative agency vocabulary (best model)

Run 4's weakness was vocabulary, not architecture: it scored 63.6% on unseen sensitive
phrasing because it had learned my synthetic payload nouns. Run 5 attacked that directly.

**Data.** 2,500 rows balanced across six classes, built from ~30 sensitive payloads in a
creative agency's real vocabulary - client assets under NDA, pitch decks, media plans,
retainer margins, day rates, SOW redlines, talent releases, focus groups, CRM exports,
casting tapes, unannounced campaigns, competitor analysis, licensing invoices, trademark
filings, performance reviews, redundancy notes, payroll, pensions, supplier contracts,
board packs, penetration tests, DPAs, candidate CVs, salary benchmarking - each paired with
a harmless lookalike from the same domain. Merged with the earlier multi-taxonomy set and
rebalanced (local 700, doc 650, formatting 400, coding 250, reasoning 250, vision 250).

**Held-out tests.** Two, both never used for tuning:
- a **creative holdout** of 270 rows using *disjoint* creative nouns (influencer fees, churn
  analysis, bereavement records, exit interviews...)
- the **frozen generic 59-case set** from ?2.7

| Slice | Run 4 | **Run 5** | Target |
|---|---|---|---|
| Generic fresh, overall | 81.4% | **89.8%** | - |
| Generic fresh, ordinary | 91.9% | 91.9% | >=95% (short) |
| Generic fresh, sensitivity | 63.6% | **86.4%** (19/22) | >=98% (short) |
| Generic fresh, implicit | 41.7% | **75.0%** (9/12) | - |
| Generic fresh, explicit | 9/10 | **10/10** | met |
| Creative holdout, harmless | - | **98.7%** (77) | - |
| Creative holdout, sensitive | - | **78.3%** (60) | >=98% (short) |

**What this shows.** Vocabulary breadth was the lever: +23 points on generic sensitivity and
+33 on implicit, with no change to ordinary routing. It also generalised *outside* the
creative domain, which is the important part - the model learned something closer to the
concept, not just more nouns.

**What is still missing.** One sensitive case in five still leaks, and ordinary routing sits
at ~92% rather than 95%. Both now look like data coverage, not method: the creative holdout
(78.3%) is lower than the generic set (86.4%), which says unseen vocabulary is still the
binding constraint.

**Calibration note.** ECE is 0.089 overall but **0.211 on sensitive cases** - it is
overconfident exactly where errors matter most.

### 2.9 Can a threshold close the gap? No.

Because the model emits a probability per route, the deployment policy need not be the
argmax: escalate to `local` whenever its probability clears a bar. Since a false `local`
costs capacity and a missed one costs confidentiality, that asymmetry should help.

Measured on the frozen generic set (22 sensitive, 37 ordinary), no GPU needed - the stored
probabilities were replayed:

| bar for `local` | sensitivity | ordinary accuracy | cases sent local |
|---|---|---|---|
| argmax (>=0.25) | 86.4% (19/22) | 91.9% (34/37) | 21 / 59 |
| 0.20 | 86.4% | 89.2% | 22 |
| 0.15 | 86.4% | 89.2% | 22 |
| **0.10** | **90.9% (20/22)** | 89.2% (33/37) | 23 |
| 0.00 | 100% | 0% | 59 |

**Verdict:** thresholding buys about +4.5 points of sensitivity for ~3 points of ordinary
accuracy, then collapses to sending everything local. **There is no operating point that
reaches 98% sensitivity with usable routing.** The class probabilities do not separate well
enough, and no amount of threshold tuning fixes that.

**Consequence:** the remaining gap is a model/data problem, not a policy problem. The only
levers left are more representative data, or accepting a documented ceiling near 90%.

### 2.10 Run 6 - eight departments of vocabulary (best in-domain model)

No real data was available, so synthesising continued. Run 6 widened in three ways at once:

- **eight departments**, not one: creative, finance, HR, legal, sales, ops, medical, engineering
  - the customers are other departments, so a creative-only vocabulary was too narrow
- **composition instead of enumeration**: ~48 sensitive/benign pairs x 12 verbs x 5 prefixes x
  5 suffixes, deduplicated - 3,900 rows over 2,712 distinct phrasings
- **three payload shapes**: document nouns, field-level records (first_name/dob/address), and
  credential-shaped strings (`postgres://admin:...`, `Authorization: Bearer eyJ...`)

The ten holdout nouns from ?2.8 were excluded, so that holdout stays unseen.

| Set | Run 5 | **Run 6** | Target |
|---|---|---|---|
| Creative holdout, sensitive | 78.3% | **90.0%** (54/60) | >=98% |
| Creative holdout, harmless | 98.7% | **100.0%** (77/77) | - |
| Creative holdout, ECE | 0.089 | **0.044** | <=0.10 met |
| Generic fresh, sensitivity | 86.4% | 86.4% (19/22) | >=98% |
| Generic fresh, implicit | 75.0% | **83.3%** (10/12) | - |
| Generic fresh, explicit | 10/10 | 9/10 | 100% |
| Generic fresh, ordinary | 91.9% | 91.9% | >=95% |

**In-domain: a clear win.** Sensitivity +11.7 points and calibration halved, on nouns the model
never saw.

**Cross-domain: flat.** The generic set did not move overall (86.4% both runs); implicit improved
but one explicit case slipped. This is the diminishing return predicted after Run 5 - adding more
of my own vocabulary buys in-domain gains, not general ones.

**Thresholding is now inert.** On Run 6 the sensitivity/ordinary curve is flat from 0.10 upward:
the model is no longer near-tied on the cases it gets wrong, it is confidently wrong. That removes
the last policy lever.

**Where that leaves us:** Run 6 is the best model for creative-agency-shaped traffic (90% sensitive,
100% harmless). For traffic from departments whose nouns I did not imagine, expect about 86%.
Further synthesis of the same kind is unlikely to move the cross-domain number.

### 2.11 Run 7 - short imperatives, and a lesson about sample size

The six remaining errors all had one shape: a short imperative ("sort", "normalise", "merge",
"extract", "put the keys in") carrying a payload whose sensitivity the model ignored. My earlier
rows were long ("Here's ... before the client call"), so Run 7 trained on 700 deliberately terse
rows - every verb paired with both a sensitive and a harmless payload, so "short imperative" could
not itself become a cue - then continued from Run 6.

| Held-out set | Run 6 | Run 7 |
|---|---|---|
| Fresh 59-case set, sensitivity | 86.4% (19/22) | **100% (22/22)** |
| Fresh 59-case set, ordinary | 91.9% | 86.5% |
| **Creative holdout, sensitive (60)** | **90.0% (54/60)** | 88.3% (53/60) |
| Creative holdout, harmless (77) | 100% | 100% |
| Creative holdout, ECE | 0.044 | 0.051 |

**The 100% did not replicate.** On 22 sensitive cases it read as perfect; on 60 it is 88.3%,
one case behind Run 6. That is the small-sample trap in its purest form: with n=22, one case is
4.5 points, and 22/22 has a 95% confidence interval reaching down to about 85%.

**Run 7's cost was real, though.** Ordinary traffic fell from 91.9% to 86.5%, and three of its
five remaining errors are false `local` calls ("this JSON object", "this invoice", "a payment
ledger") - over-conservative rather than dangerous. It sends 25 of 59 cases local against Run 6's
21.

**Verdict: Run 6 stays the model to ship** (better ordinary accuracy, equal sensitivity on the
larger sample). Run 7 is not promoted.

**Ceiling reached.** Across Runs 5-7 sensitivity on unseen data sits at 86-90% and does not move.
Further synthetic data is not producing generalisation; it is producing rearrangements. The
honest figure to report is **~88-90% sensitivity, 100% on harmless traffic, ECE ~0.05**.

### 2.12 Independently generated realistic data - our range confirmed

Another AI (not me) generated 50 prompts from a written brief: nine departments, creative-agency
vocabulary, 13 sensitive (9 implicit, 4 explicit), no banned keywords. It produced data I would
not have thought of - "the attached unreleased Superbowl spot script", "a Q3 appraisal dump",
"email addresses and birthdates of users who entered the sweepstake".

**Data quality check: 50/50 rows parse, 0 implicit keyword leaks, 0 duplicates, distribution on
spec, nine departments, varied formats.**

Run 6 evaluated on it, never trained on it:

| Slice | Run 6 on generated data | Run 6 on my synthetic sets |
|---|---|---|
| Overall | 92.0% | 89.8% |
| Sensitivity | **84.6%** (11/13) | 86.4% / 90.0% |
| - explicit | **100%** (4/4) | 90-100% |
| - implicit | 77.8% (7/9) | 75-83% |
| Ordinary traffic | 94.6% | 91.9% |

**This is the important result: an independent generator, writing realistic agency requests,
reproduces the same band.** The ~85-90% sensitivity figure is not an artefact of my synthetic
data. It is what this model does.

**Four misses, two of them real:**
- `g007` "reformat this list of upcoming redundancies into a Markdown table" -> formatting, at
  **100% confidence**. A genuine leak on a word the model has not learned as sensitive.
- `g004` "review the attached unreleased Superbowl spot script and flag any clearance risks" ->
  reasoning at 40% confidence. Genuinely borderline between reasoning and local.
- `g017` and `g024` are label ambiguity in the generated set (a *published* case study; a
  *standard* NDA template) - roughly 4% noise, which matters before training on it.

**Plan:** keep batch 1 frozen as an independent test set, generate batches 2-4 for training, and
retrain. Batch 1 must never be trained on.

### 2.13 Run 8 - training on generated data made it worse

Batches 2-4 (150 rows, validated: 0 malformed, 0 keyword leaks, 0 duplicates, ids g051-g200)
were converted to training rows and trained on top of Run 6, mixed with 1,500 rows from the
synthetic corpus. Evaluated on **batch 1, frozen and never trained on**:

| Frozen batch 1 (50 cases, 13 sensitive) | Run 6 | **Run 8** |
|---|---|---|
| Overall | 92.0% | **86.0%** |
| Sensitivity | 84.6% (11/13) | **61.5%** (8/13) |
| - explicit | **4/4** | 2/4 |
| - implicit | 7/9 | 6/9 |
| Ordinary | 94.6% | 94.6% |

McNemar: 5 cases for Run 6, 2 for Run 8, **p = 0.453** - not distinguishable at n=50. But every
quality slice moved the wrong way, so Run 8 is **not promoted**.

**Why it probably happened.** Two plausible causes, and they point the same direction:

1. **Vocabulary collision.** The generated rows use our serving names (formatting,
   standard-coding...); the 1,500 synthetic rows use the creative names (cosmetic-edit,
   code-work...). Mixing both at a low learning rate may have disturbed the head more than the
   150 useful rows helped.
2. **The checkpoint is saturated with my synthetic data.** Run 6 was trained on ~7,000 rows of
   my own vocabulary across four runs. Adding 150 rows of a different style cannot undo that;
   the model is shaped by the corpus, not nudged by the addition.

**The lesson:** generated data needs to be the *basis* of a run, not a garnish on top of mine.
The clean experiment is to train a fresh model from the released v1 using only externally
generated rows - several hundred of them - and compare that against Run 6 on the same frozen set.

**Current best remains Run 6** (`checkpoints/creative-v2`): 84.6% sensitivity on generated data,
90.0% on the creative holdout, 100% on harmless traffic, ECE 0.044.

### 2.14 Run 9 - the trade-off, now measured from both sides

The single-file batch (400 rows) turned out to be weak: 32 of its 32 `standard-coding` rows were
the *same prompt* repeated, and it contained **zero** doc-transformation and **zero**
image-analysis. After deduplication it held 308 usable rows across four routes.

Run 9 was therefore trained **fresh from released v1** (not from a synthetic checkpoint) on those
308 rows plus synthetic top-ups for the three thin classes - 758 rows, all six classes, 41%
generated - at a higher learning rate for 2 epochs. This is the experiment Run 8 failed to run.

| Frozen batch 1 (50 cases, 13 sensitive) | Run 6 | Run 8 | **Run 9** |
|---|---|---|---|
| Overall | **92.0%** | 86.0% | 74.0% |
| Sensitivity | 84.6% (11/13) | 61.5% (8/13) | **92.3% (12/13)** |
| - explicit | 4/4 | 2/4 | **4/4** |
| - implicit | 7/9 | 6/9 | **8/9** |
| Ordinary | **94.6%** | 94.6% | **67.6%** |

McNemar Run 6 vs Run 9: 12 cases for Run 6, 3 for Run 9, **p = 0.035**.

**The important result is the shape, not the ranking.** We now have three points on the same
curve, measured on identical held-out data:

| Model | Sensitivity | Ordinary | Where the bias sits |
|---|---|---|---|
| Run 9 | 92.3% | 67.6% | biased toward `local` |
| Run 6 | 84.6% | 94.6% | balanced |
| Run 8 | 61.5% | 94.6% | too weak on sensitivity |

**Sensitivity and ordinary accuracy trade against each other, and no setting reaches both.** This
is the same trade the threshold analysis found (?2.9), now reproduced at the level of training
data rather than decision policy. Run 9 buys 7.7 points of sensitivity for 27 points of ordinary
accuracy - a bad trade at Run 6's balance, but perhaps the right one if a leak is unacceptable.

**What this settles:** the 98%-sensitivity target is not reachable with this model on this data by
any route we have tried - four training configurations, three datasets, threshold tuning, and two
generators. The honest deliverable is a *choice of operating point on a measured curve*, not a
model that meets the spec.

### 2.15 Two ideas tested: embeddings fail, the binary gate works

**Embeddings: no.** A sentence-embedding bank (1,342 sensitive training prompts, all-MiniLM-L6-v2,
23M params) scored held-out prompts by max cosine similarity:

| Test set | AUC | recall @95% specificity |
|---|---|---|
| Generated batch 1 | 0.669 | 23.1% |
| Fresh generic | 0.655 | 13.6% |
| Creative holdout | 0.653 | 32.5% |

AUC ~0.66, far below the model's 85-92%. The cause is structural: our hard negatives are
same-domain, same-verb lookalikes, so cosine similarity cannot separate them (sensitive median
0.567 vs normal 0.514). **Do not build this.** Cost to find out: 20 minutes.

**The binary gate: yes.** The same Run 6 checkpoint, asked a `noul` question instead of the
6-way choice - "should this be handled where data never leaves the company?":

| Set | AUC | recall @95% spec |
|---|---|---|
| Generated batch 1 | 0.884 | 61.5% |
| Fresh generic | 0.948 | 90.9% |
| Creative holdout | 0.975 | 95.0% |
| **Pooled (n=379, 155 sensitive)** | **0.962** | **93.5% at a 5% FP budget** |

A binary question is a much easier task than a 6-way choice, and it separates far better.

**Together they cover each other.** Joined on the 109 cases where both were measured (35
sensitive):

| Config | Sensitivity | FP rate |
|---|---|---|
| Router alone | 85.7% | - |
| Gate alone | 88.6% | 13.5% |
| **Gate OR router** | **94.3%** | **8.1%** |

**Only 2 of 35 sensitive cases are missed by both.** That is the best sensitivity achieved
anywhere in this project, and it uses one model asking two questions - no new dependency, no
extra training.

**Where it sits on the curve:** ~94% sensitivity at ~85% ordinary accuracy, against Run 9's
92% at 68% and Run 6's 86% at 92%. It is a strictly better point than Run 9.

**Still short of 98%.** The gate's own ceiling is 93.5% because ~6.5% of sensitive cases score
exactly zero, and 2 cases defeat both signals.

### 2.16 Task 1 and Task 2 completed on a later run (binary adapter tried)

Recorded after the fact; the intermediate agent did not update this file.

**Task 1 - gate+router measured per set** (`results-task1-per-set.json`, 5% FP budget):

| Set | sensitive | gate AUC | router recall | **union recall** | union FP |
|---|---|---|---|---|---|
| Creative holdout | 120 | 0.975 | 68.3% | **95.0%** (CI 89.5-97.7) | 0.7% |
| Generated batch 1 | 13 | 0.884 | 84.6% | **92.3%** (CI 66.7-98.6) | 0.0% |
| Fresh generic | 22 | 0.948 | 86.4% | **90.9%** (CI 72.2-97.5) | 5.4% |

**The two components cover for each other.** On the creative holdout the gate carries it (95.0%)
while the router collapses (68.3%). On the generated set the reverse: the gate fails badly at a
tight FP budget (15.4%) while the router carries it (84.6%). The union is 90.9-95.0% everywhere.
That robustness is the strongest argument for the two-question design.

The router's 68.3% here versus 90.0% measured earlier is a **serving-vocabulary penalty**: the
earlier figure used the holdout's own label names (cosmetic-edit, documents - the training
vocabulary); serving uses `route-criteria-v2.json` names (formatting, doc-transformation). **The
68.3% is the production number.**

**Task 2 - binary sensitivity adapter trained** (`checkpoints/creative-v2-noul`, 200+ steps,
lora_r 48, loss to ~1e-7). It is **worse pooled but better where it matters most**:

| Set | sensitive | zero-shot gate AUC | fine-tuned AUC | zero-shot rec@5fp | fine-tuned rec@5fp |
|---|---|---|---|---|---|
| Creative holdout | 120 | **0.975** | 0.892 | **95.0%** | 78.3% |
| Generated batch 1 | 13 | 0.884 | **0.958** | 15.4% | **92.3%** |
| Fresh generic | 22 | 0.948 | 0.950 | 86.4% | **90.9%** |
| **Pooled** | 155 | **0.962** | 0.904 | **93.6%** | 81.3% |

Pooled, the fine-tune **regresses** (-5.8 AUC, -12.3 recall), driven by the 120-case creative
holdout. But on the two sets it was *not* trained toward, it **improves sharply** - most
strikingly on the externally generated set, 15.4% -> 92.3% at the same FP budget.

**Reading:** the fine-tune taught "financial/HR vocabulary = sensitive", which generalises to
realistic traffic and damages the broad semantic judgement the zero-shot base model already had.
So which gate is "better" now depends on **which set represents production** - and the creative
holdout is my own synthetic data, while batch 1 was written by an independent generator.

**Cannot decide on 13 cases.** Batch 1 gives 12/13, CI 66.7-98.6. The deciding evidence is more
realistic data, not more tuning.

**Status: neither gate is promoted.** Zero-shot gate remains the default (better pooled, better on
the only large set); the fine-tuned adapter is kept as a candidate pending a larger realistic set.

### 2.17 Fresh independently-written test set decides it (28 Oct)

A new 50-case test set (`ai_routing_test_set.md` -> `golden-dataset-test1.json`) was written by an
outside generator to the test-set brief: ids `t001-t050`, 23 sensitive (16 implicit, 7 explicit),
all six routes present, **0 exact and 0 near overlap with any previous set**, 0 duplicates. One
item (`t011`) was mislabelled implicit while containing "medical"; relabelled explicit before use.

| Strategy | Recall | 5% FP threshold | FP actually |
|---|---|---|---|
| Zero-shot gate (creative-v2) | 69.6% (16/23) | 0.993 | 7.4% |
| **Fine-tuned gate (creative-v2-noul)** | **82.6% (19/23)** | 0.980 | 7.4% |
| Router alone | 82.6% (19/23) | - | **18.5%** |
| Gate OR router (either checkpoint) | 82.6% (19/23) | - | 18.5% |

AUC: zero-shot 0.887, fine-tuned 0.882 - **a tie**. The fine-tuned gate is better only at a tight
FP budget (82.6% vs 69.6%), and it catches 4 cases the zero-shot misses while losing 1.
**Exact McNemar p = 0.375** - not distinguishable on 23 sensitive cases.

**Three findings that change the picture:**

1. **Realistic sensitivity is ~83%, not 92-95%.** Every synthetic holdout (creative 95%, fresh 90%)
   was optimistic. This set - written blind to our training data - puts the system at 82.6%. That
   is the number to report.
2. **The router earns nothing on sensitivity and costs 11 points of capacity.** Its 82.6% recall
   comes with 18.5% false positives; the gate achieves the same recall at 7.4%. The two-question
   union adds exactly zero here. **The gate should own sensitivity alone**; the router is only
   worth running afterwards on cleared traffic.
3. **Four cases defeat everything** - `t005` board compensation resolutions, `t007` server access
   failure logs, `t009` a DAM asset-manifest script, `t031` a badge-scan log. All four are bland
   internal artefacts with no overt marker. This is the same failure family as every previous
   round, and it is where the remaining ~17% lives.

**Ceiling restated:** 82.6% at 7.4% FP on realistic data, with a 95% interval of 62.9-93.0%.
The 98% target is not met and the gap is not shrinking.

## 3. Things you need to know

**About the model**
1. **Qwen3.5 is the engine inside Strands Decider**, not a competing candidate. The checkpoint is a rank-16 LoRA adapter on `Qwen/Qwen3.5-2B-Base` plus a ~1M-parameter **pointer head**. Its language-modelling head is discarded, so it cannot generate text.
2. **The pointer head holds no per-option parameters**, so **label sets are defined per request** — 2 to 255 options, no retraining to add or rename labels. This is the architecture's main selling point for your use case.
3. The head is also why there is no "24 option limit" — that constraint belonged to the older head design.

**About what actually moves accuracy**
4. **Rewording beats upgrading the model.** The model went v19 → v1 (+6.5 pts) but that was *not* statistically distinguishable (p = 0.143). Rewording one description was (+12 pts, p = 0.001).
5. **Images work with no training.** With `--vision` and the reworded description, all 8 image cases route correctly, including both sensitive ones. Every earlier configuration sent those two to `image-analysis`.
6. **The remaining gap is exactly one thing:** implicit sensitivity in text — 25/30 now, target 28+/30.

**About our benchmark**
7. **108 cases is too small to rank models.** One case = 0.93 points; 95% CIs are ±9 points. Use it as a regression check, not a benchmark.
8. **Nine of 35 eval rows exceed the 4,096-token window** (up to 11,378 tokens). Long documents are **not servable** by this model and must be chunked before the router. The vendor's rule: delete over-window rows, never truncate.
9. The 68 "ordinary traffic" cases are what the routing target should be judged on — that's the 97.1%.

**About training on this machine**
10. **It works.** Windows falls back to a reference PyTorch implementation for the Gated DeltaNet layers (no Triton), which is slower but correct. Peak memory 4.7 GiB at `max_length` 384.
11. **`max_length` must be ≥ ~300**, not 128 — the option descriptions dominate the prompt, not the request text.
12. **`val_acc 1.0` and `loss 0.0000` mean memorisation, not learning.** Use the held-out payload-genre split and the 108-case benchmark, never the training val split.
13. **A fine-tune destroys calibration.** T1 needed temperature 6.0. Always run `training.evaluate calibrate` before quoting any confidence or threshold.
14. **The KL anchor does not fit in 8 GB.** T2 stalled at zero steps. Protection against forgetting must come from **balanced data**, not the anchor.
15. Environment: torch 2.14.1+cu130, transformers 5.20.2, peft 0.21.2, datasets 5.1.0, RTX 4060 Laptop 8 GB, ~7.4 GB disk free. **WSL is blocked** on this machine (`E_ACCESS_DENIED`).

**About adapting to new labels** (measured in §2.3)
16. **The model answers with any label names you give it** — no retraining needed. But it is not *name-neutral*: renaming our six labels cost ~10 points of sensitivity with the descriptions unchanged.
17. **The literal word "local" is load-bearing.** Updating the instruction to match a renamed label made sensitivity *worse* (57.5% → 47.5%), because the model associates that exact token with the sensitivity behaviour. Any final training must be done on the team's own vocabulary.
18. **Option count costs accuracy**: 6 → 12 → 24 options gave 86.1% → 73.1% → 62.0%. Keep the real taxonomy as small as the design allows; ≤12 is much better than 24.

---

## 4. Open questions

| # | Question | Blocks |
|---|---|---|
| 1 | **Who signs off a checkpoint for production, and who owns rollback?** | Handover |
| 2 | **Do long documents matter?** They exceed the 4,096-token window entirely; needs chunking, not training | Scope |
| 3 | **How do we validate 95% on fresh cases?** The 108-case set has been used for selection several times | Trust in the number |
| 4 | Does the team want several adapters (one per department) over the one base model? | Deployment shape |

The label taxonomy is no longer an open question: it is open **by design** (services undecided,
departments send varied inputs), and Run 3 was trained specifically to be name-agnostic.

---

## 5. Next actions

| Priority | Action | Why |
|---|---|---|
| ~~1~~ | ~~Label-adaptation test~~ — **done, §2.3** | Renaming costs ~10 points of sensitivity; 24 options costs ~24 points of accuracy |
| 1 | **Get the real taxonomy** — label names, descriptions, and the instruction text the harness will send | Now the hard blocker: training on our vocabulary produces an association the team cannot reuse |
| 2 | **Rebuild training data balanced across all labels** (~50+ rows per label) | Replaces the KL anchor, which cannot run here |
| 3 | Retrain on the real vocabulary, calibrate, evaluate on a ≥1,000-row frozen set | The vendor's floor for a defensible number |

---

## 6. Artefacts

| File | What it is |
|---|---|
| `golden-dataset-v2.json` | 108-case benchmark (40 sensitive, 18 images, 10 long docs) |
| `route-criteria.json` / `route-criteria-v2.json` | original vs reworded option descriptions (**v2 is the good one**) |
| `results-strands-*.json` | per-case results for every evaluation run |
| `build_sensitivity_pairs.py` → `work/pairs/*.jsonl` | 600 matched pairs, 420 train / 180 held out by payload genre |
| `build_training_rows.py` → `work/train/rows.jsonl` | our benchmark converted to the vendor row format |
| `train-pairs.yaml` | T1 config (pairs only, no anchor) |
| `train-combined.yaml` | T2 config (mixed data + KL anchor — does not fit in 8 GB) |
| `checkpoints/pairs-v1` | T1 checkpoint (calibrated, T=6.0) |
| `compare.py` / `stratify.py` / `validate_dataset.py` | significance, slicing, dataset audit |
| `strands-decider/` | vendor repo, cloned, editable install |

---

## 7. Log

| Date | What happened |
|---|---|
| 26 Oct | Benchmarked v19 and v1; rewording experiment; images fixed; T1 trained and evaluated; T2 attempted and aborted (OOM); this file created |
| 26 Oct | Label-adaptation test (§2.3): renamed labels cost ~10 pts of sensitivity; the token "local" is load-bearing; 24 options cost ~24 pts of accuracy |
