# MATRIX research-fit retrieval foundation

## Status

`research-fit-v1` is the request contract introduced before a retrieval-index
change. It preserves the user's full question, a reviewable interpretation,
and future-facing filters while the current SPECTER2/FAISS author-chunk index
continues to serve results.

This is not a claim that the current index can yet apply every field. In
particular, `affiliation_filters` are preserved for the planned source-backed
work-level corpus and are visibly marked as not yet applied in the search
plan.

## Current compatibility path

1. The chat creates a plan with topic, method, population, setting,
   evidence-stage, needed-capability, and constraint fields.
2. The user sees the plan and can refine it before confirming search.
3. The current backend builds one deterministic compatibility query from the
   positive retrieval fields and sends it to the existing SPECTER2/FAISS path.
4. The full original question, rather than only the shortened retrieval query,
   is supplied to the bounded evidence-note step.

The plan does not select, reorder, or invent candidates. Existing selected-team
context and exclusions remain authoritative.

## Future corpus gate

Before applying affiliations, article topics/subtopics, evidence-stage filters,
or work recency as ranking/filtering signals, document for each signal:

| Signal | Source/snapshot | Coverage | Identity confidence | User-visible behavior | Ranking/filter use |
| --- | --- | --- | --- | --- | --- |
| Work title/abstract | pending | pending | pending | evidence link | retrieval |
| Topic/subtopic | pending | pending | pending | interpreted field/evidence | retrieval + coverage |
| Affiliation | pending | pending | pending | explicit filter | filter only |
| Evidence stage | pending | pending | pending | supporting work field | retrieval + coverage |

No new signal should affect ranking until its source, freshness, coverage,
identity resolution, terms, and failure behavior have been reviewed.

## Accuracy baseline required before rank tuning

Create a small, domain-reviewed fixture set with guest, mentor, collaborator,
and selected-team examples. Each case needs expected supporting works and hard
negatives for a wrong method, population, evidence stage, or author identity.

Compare the frozen current path against later changes using at least:

- evidence-supported candidate Recall@k and nDCG@k;
- requested-field coverage and constraint violations;
- team-gap coverage and redundancy;
- evidence precision (whether cited works support the stated field);
- no-result, partial-coverage, and latency behavior.

The later retrieval migration should be work-first: dense SPECTER2 retrieval,
BM25 exact-term retrieval, reciprocal-rank fusion, then documented author
evidence aggregation. It must not retrieve an author first and search for a
plausible paper afterwards.

## Non-claims

Publication evidence does not establish willingness, availability, formal
mentoring quality, funding fit, clinical qualifications, collaboration, or
impact. Graph proximity is not recorded collaboration evidence.
