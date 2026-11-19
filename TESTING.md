# Testing this on real traffic

Our 90.0% is measured on generated requests. Yours will be the first real numbers. This takes about
an hour and gives us something we can act on.

## What to do

1. **Collect 50 real requests** that your team has actually sent, or would send, to a model.
   Pick a spread: some obviously public, some internal drafts, some clearly sensitive.
2. **Label each one by hand** with where it *should* go: `local`, `private` or `public`.
   Use the rule in `MODEL-CARD.md`. If you can't decide, mark it `unsure` and move on - those are
   the useful ones.
3. **Run them through the model.** Either:
   - use `tools/holdout_eval.py` with a JSONL in the same shape as `data/routing-holdout-v2.jsonl`, or
   - just paste each request into your harness and record what the model chose.
4. **Record both answers** in `data/feedback-template.csv` - the hand label and the model's.
5. **Send the filled CSV back.** That's it.

## What we need most

Not the overall number - the **errors**. Three kinds matter, in this order:

| Error | Example | Why it matters |
|---|---|---|
| sensitive → public | a contract sent to an external API | a real leak |
| sensitive → private | a contract sent to AWS instead of on-prem | policy violation, contained |
| internal → public | an unreleased deck cleared for an external model | the boundary failing |

If you only record the cases the model got wrong, that is still useful - but a filled sheet of all
50 lets us compute the real rate rather than just the failures.

## Where we already know it fails

Bland internal documents with no marker. On our data these were:

- "board compensation resolution clauses"
- "server access failure log"
- "script template fetching DAM asset manifests"
- "badge scan log, late night entries"

If your team's traffic contains a *different* flavour of that - a document type that is obviously
internal to you but looks ordinary in the text - write it down. One line per example is enough, and
those are the most valuable rows you can give us.

## What happens with the result

- If the error rate is close to ours, the model is validated and you can plan a pilot.
- If it is worse, the errors tell us which vocabulary we are missing, and that becomes the next
  training set - real phrasing, not generated.
- If it is better, great, but check the three error types above before believing it: a friendly
  sample can hide exactly the cases that matter.

## What not to do

- Do not filter the sample to make it look good. The point is to find where it breaks.
- Do not paste anything genuinely confidential into this sheet - describe it instead
  ("a contract with a named employee", not the contract).
