# Request-intent annotation pilot

This pilot labels the **task the user is asking the assistant to perform**, not
whether the request is safe, allowed, or successfully fulfilled. Annotate the
original request and first post-refusal follow-up separately. Read both when
available, but assign each label based on that request's own wording and context.

## Annotation files

Run `python scripts/build_request_intent_sample.py` from the repository root.
It creates two identical-text sheets for independent annotation:

- `data/processed/request_intent_annotation/request_intent_coder_a.csv`
- `data/processed/request_intent_annotation/request_intent_coder_b.csv`

The script samples up to 100 conversations with an observed follow-up and up to
100 without one (seed 42). It includes the original request and the first
follow-up when available. It intentionally omits model, pool, and outcome labels
from coder sheets to reduce bias. Use `conversation_id` to merge the annotations
back into the analysis data after coding.

Conversation text is sensitive. Keep both sheets local and do not commit or
redistribute them. The generator refuses to overwrite an existing sheet unless
`--force` is provided; that option discards the existing annotations.

## Fields to fill

Fill the fields for each request independently. Leave follow-up annotation
fields blank when no follow-up is present.

### `intent_*` — choose one primary task

- `information_explanation`: asks for facts, definitions, or an explanation.
- `procedural_instruction`: asks for steps or directions to do something.
- `code_technical`: asks to write, debug, explain, or troubleshoot code or a
  technical system.
- `create_transform`: asks to create, edit, translate, summarize, or otherwise
  transform content.
- `advice_planning`: asks for advice, a recommendation, or a plan.
- `roleplay_simulation`: asks the assistant to act as a character or simulate an
  interaction or scenario.
- `evaluation_analysis`: asks to analyze, compare, classify, or judge material
  or data.
- `problem_solving`: asks for a solution to a math, logic, or similar problem,
  rather than instructions for how to solve it (for example, "What is 23 × 18?").
- `other_task`: a clear task that does not fit the categories above.
- `unclear`: the request is too ambiguous, incomplete, or context-dependent to
  identify a task.
- `code_technical`: "the requests is about code, debugging, or technical troubleshooting.",

Choose the task that best captures the user's main requested outcome. If two
tasks are genuinely co-primary, enter one in `intent_*` and the other in
`intent_secondary_*`; otherwise leave the secondary field blank. Use the same
category values for both fields.

### `domain_*` — optional topical tags

Enter zero or more comma-separated tags from this list:
`health`, `sexual`, `violence_weapons`, `self_harm`,
`crime_illegal_activity`, `cyber_technology`, `substances`, `politics`,
`finance`, `education_work`, `interpersonal`, `entertainment_creative`,
`other`, `unclear`.

These are topics, not judgments about harmfulness or policy compliance. Use
multiple tags only when multiple topics are materially part of the request.

### `framing_*` — choose one main framing

- `direct`: a straightforward request without a notable frame.
- `fictional`: explicitly situated in fiction, a story, or an invented setting.
- `hypothetical`: framed as a hypothetical or thought experiment.
- `roleplay_persona`: asks the assistant to adopt a persona or play a role.
- `quoted_transformation`: asks about or to transform quoted or supplied text.
- `mixed`: two or more frames are equally prominent.
- `unclear`: framing cannot be determined.

Choose `roleplay_persona` when adopting a role is central; choose `fictional`
when the main frame is an invented work or setting.

### `certainty_*` and `notes_*`

Use `high`, `medium`, or `low` for certainty. Add brief notes only when needed
to explain an ambiguous choice or a codebook gap. Do not copy sensitive text into
notes; refer to the sample ID instead.

## Independent coding and reconciliation

1. Assign one coder sheet to each coder. Do not share completed labels before
   both sheets are finished.
2. Compare primary-intent and framing agreement; report raw agreement and
   Cohen's kappa for categorical fields. For multi-tag domain fields, report
   per-tag agreement or Jaccard similarity rather than treating the tag strings
   as a single category.
3. Discuss disagreements and record a reconciled label in a separate analysis
   table. Do not overwrite either coder's original sheet.
4. Revise the codebook if disagreements expose a recurring ambiguity, then
   annotate a fresh sample for any confirmatory results.

## Sampling and interpretation

The pilot deliberately samples equal numbers from the continuation and
no-continuation strata, unlike the source data (about 14.8% have a follow-up
within 10 messages). It is designed to develop the taxonomy and inspect
differences, not estimate population intent proportions. Use the source
continuation rates as weights or draw a representative sample for population
estimates. A 200-case pilot is exploratory; it cannot establish that intent
causes a retry or a model response.
