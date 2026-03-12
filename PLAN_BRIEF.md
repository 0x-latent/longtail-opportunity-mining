# Longtail Opportunity Mining — Plan Brief

## Goal
Build a first-version opportunity discovery pipeline for cold/cough social-media text mining.

The objective is NOT generic trend analysis. The objective is to discover niche but actionable opportunity signals that are usually buried under mainstream/high-frequency discussion, and translate them into:
- marketing angles
- segment opportunities
- potential new product directions

## Scope for V1
Domain: cold / cough
Data size: ~100k posts to start (can expand later)

## Known labels available
Assume existing label taxonomy includes at least these dimensions:
- scenario
- symptom
- solution / product
- audience / people
- emotion
- efficacy
- dosage form
- concerns / barriers

Implementation should make label dimensions configurable because the full final taxonomy may be adjusted later.

## Core approach
Use soft filtering first, not hard deletion.

### Soft filtering rule
When text matches known labels, replace matched spans with masks rather than deleting the whole context.
Examples:
- scenario -> [SCENE]
- symptom -> [SYMPTOM]
- product/solution -> [PRODUCT]
- audience -> [AUDIENCE]
- concern -> [CONCERN]

Goal: preserve sentence structure and unknown context so residual semantics remain clusterable.

## Desired outputs
Two-layer output design:

### 1. Insight cards (primary output)
Each card should explain:
- topic / opportunity name
- why it matters
- representative posts
- relevant known-label distribution
- candidate business interpretation
- whether it is worth follow-up

### 2. Opportunity ranking (secondary output)
Rank opportunities by a score composed from factors like:
- novelty
- concentration / cluster coherence
- business relevance
- support size

## Output shape preferences
Primary output should support both:
- topic groups
- structured combinations such as scenario × symptom × need / product feature / solution preference

For V1, prioritize discovering:
- scenario × symptom × need
Then optionally extend to:
- scenario × symptom × product/solution

## Proposed pipeline to design
1. Data ingestion and cleaning
2. Dedup / normalization
3. Label matching and mask-based soft filtering
4. Residual text embedding
5. Residual clustering
6. c-TF-IDF or equivalent cluster-level term amplification
7. Insight-card generation
8. Opportunity scoring and ranking
9. Evaluation / review workflow

## Key design questions to answer in the plan
Please produce a concrete design/spec for:

1. Recommended project structure
2. Data schema
3. Label configuration format
4. Masking strategy details
5. Residual clustering design
6. c-TF-IDF scoring design
7. How to derive both:
   - topic-group output
   - structured combination output
8. Evaluation framework
9. MVP experiment plan
10. What should be implemented first vs later

## Required comparison experiments
Design V1 so it can compare:
1. soft filtering vs hard filtering
2. no clustering vs clustering + c-TF-IDF
3. unigram vs phrase-level extraction
4. topic-group output vs structured-combination output

## Success criteria for V1
Not purely offline metrics. Focus on business usefulness.
A good V1 should enable manual review where:
- top 20 results contain at least several clearly meaningful niche opportunities
- at least 2-3 outputs can be translated into concrete marketing or product hypotheses
- outputs are more novel/actionable than simple high-frequency summaries

## Execution preference
Use Claude Code as the primary executor for this project.
Use Codex only as auxiliary support when helpful.

## Current instruction for this run
This run is PLAN MODE ONLY.
Do not implement the pipeline yet.

Deliverables for this run:
- a concrete technical/design plan
- recommended folder structure
- implementation phases
- risks / assumptions
- what should be confirmed before coding

If there are optional branches, propose them clearly and recommend one default path.

When finished, summarize in a concise human-readable way.

When completely finished, run this command to notify me:
openclaw system event --text "Done: longtail opportunity mining plan drafted" --mode now
