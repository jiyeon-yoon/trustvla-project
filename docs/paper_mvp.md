# Paper MVP

## Working Title

TrustVLA-Guard: Paired Instruction Stress Testing and Runtime Verification for
Vision-Language-Action Models

## Claim

Standard VLA evaluation hides important failures because it mostly reports task success.
TrustVLA evaluates paired instruction edits under the same scene and measures whether
the policy changes, preserves, or stops behavior as required. A lightweight runtime
verifier can reduce wrong-target and unsafe-success failures without retraining the VLA.

## Minimum Experiment

- Benchmark: 20-30 LIBERO tasks.
- Variants: base, paraphrase, target swap, negation, spatial swap, safety constraint,
  impossible object, ambiguous reference.
- Initial states: 5-10 per task.
- Models: start with one real VLA, then add a second if the pilot works.
- Main comparison: VLA vs. VLA + RuntimeVerifier.

## Hypotheses

H1. Base-task success overestimates reliability under instruction edits.

H2. Target swap, negation, impossible-object, and ambiguous-reference edits reveal
visual-prior behavior: the policy continues manipulating a plausible object even when
the instruction changed.

H3. Safety-constraint edits reveal unsafe success: the policy completes the nominal
task while violating a process-level safety constraint.

H4. A runtime verifier reduces wrong-target, constraint-violation, and unsafe-success
rates. It may reduce nominal success because it blocks uncertain actions.

## Required Tables

Table 1: Benchmark composition by variant type.

Table 2: Model performance under base vs. paired variants.

Table 3: Before/after runtime verifier comparison.

Table 4: Failure taxonomy with qualitative rollout examples.

## Go/No-Go Criteria After Pilot

Continue if at least one real VLA shows:

- base success is meaningfully higher than edited-instruction success, or
- wrong-target / unsafe-success failures appear in at least two variant families, and
- the verifier reduces those failures without collapsing all actions into no-op.

Stop or pivot if:

- real VLA cannot be run within budget,
- edited tasks cannot be scored reliably,
- or the verifier only works by blocking nearly every episode.

