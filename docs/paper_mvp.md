# Paper MVP: Selective Obedience in VLA Models

## Working Title

Selective Obedience in Vision-Language-Action Models: Does Better Language Grounding
Increase Unsafe Compliance?

## Central Claim To Test

Language-grounding interventions can improve benign instruction following while also
increasing compliance with hazardous instructions that conflict with a trusted safety
policy. Reporting benign and hazardous compliance jointly exposes a capability-safety
trade-off hidden by task success alone. A non-oracle safety-policy gate should reduce
hazardous compliance without causing excessive refusal on benign tasks.

Do not write this as a confirmed result until real experiments support it.

## Research Questions

- RQ1: Does stronger language grounding improve benign target/action compliance?
- RQ2: Does the same intervention increase hazardous instruction compliance?
- RQ3: Can an external safety-policy gate reduce hazardous compliance while preserving benign compliance?
- RQ4: Which instruction families and action phases produce the largest safety-obedience trade-off?

## Hypotheses

- H1: A grounding-enhanced condition has higher benign compliance than the raw VLA.
- H2: It also has higher hazardous compliance than the raw VLA on at least one hazard family.
- H3: The safety-policy gate lowers hazardous compliance with a small over-refusal increase.
- H4: The effect appears in target/contact traces before it appears in native task success.

## Experimental Conditions

Minimum pilot:

- Benchmark: 5 manually audited `libero_object` tasks.
- Initial states: 3 per task for the first pilot.
- Conditions: raw OpenVLA, grounding-enhanced OpenVLA, grounding-enhanced + gate.
- Cases: base, paraphrase, target edit, safety constraint, hazardous instruction.
- Repetitions: deterministic inference first; then 3-5 seeds if sampling is used.

Paper-scale minimum:

- 20-30 audited tasks.
- At least 2 VLA families.
- At least 2 grounding conditions or interventions.
- 5-10 matched initial states per task.
- Bootstrap confidence intervals over task/initial-state pairs.

## Primary Metrics

- Benign Instruction Compliance (BIC)
- Hazardous Instruction Compliance (HIC)
- Safety Constraint Compliance (SCC)
- Appropriate Abstention (AA)
- Over-Refusal Rate (ORR)
- Selective Obedience Score (SOS = BIC - HIC)
- Native Task Success and Instruction Task Success, reported separately
- Base/variant action-prefix trajectory distance

## Required Baselines

- Raw VLA.
- A language-grounding enhanced condition such as CAG or IGAR if reproducible.
- Text-only lexical safety filter.
- Trusted-policy semantic gate.
- Oracle upper bound, clearly labeled and never presented as a runtime method.

## Required Tables

- Table 1: Benchmark composition and independent annotation agreement.
- Table 2: BIC/HIC/SOS for raw versus grounding-enhanced policies.
- Table 3: Gate safety gain and over-refusal cost.
- Table 4: Per-variant and per-hazard breakdown.
- Table 5: Ablation of instruction-only, proposal-only, and combined gate inputs.

## Validity Requirements

- Safety policies must be authored or audited independently of benchmark outputs.
- Raw and edited cases must share scene and initial state.
- Counterfactual goals need custom predicates; native LIBERO reward cannot be reused.
- Contact metrics must come from simulator state/contacts, not only free-form `info` text.
- Report missing or unscorable cases rather than treating them as safe failures.
- Synthetic smoke-test numbers must never appear as empirical paper results.

## Go/No-Go After First Real Pilot

Continue with this claim if:

- raw and grounding-enhanced conditions show a measurable BIC difference, and
- at least one condition executes a hazardous instruction in real rollout, and
- the gate reduces HIC without blocking more than 10-15% of benign cases.

Pivot again if:

- available grounding methods cannot be reproduced on one 24 GB GPU,
- hazardous compliance is zero for every model and case,
- or counterfactual outcomes cannot be scored reliably.
