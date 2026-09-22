# MATRIX persona readiness — 2026-09-15

## Implemented locally

- Changing Mentor/Collaborator starts a fresh conversation. Signed-in conversations are saved before navigation; a save failure keeps the current conversation visible. Previous lists remain filtered by persona. Switching is disabled during active requests; Stop releases it.
- Saved people are a shared collection. Include/exclude is conversation state; it never deletes the person. Mentor selections are a shortlist to discuss. Collaborator selections are the working team and retain the existing team-aware retrieval/exclusion behavior.
- Mentor chat can compare included people using their stored publication evidence. Selection does not silently rerank current results. A new search follows the stated goal; a request to find similar people can explicitly refine that goal.
- Junior investigators can supply up to ten representative titles. The next mentor search calls the existing author-preview endpoint; an empty field retains goal-based search. Paper matching exposes the existing transformed similarity score under Reviewer details, not a mentor-suitability probability. This currently searches Bridge2AI members.
- Results contain a short explanation and supporting paper, expandable evidence, and a save action. Names and order come from retrieval. Timeout/failure uses publication context instead of generated notes.
- Conversation reset clears selections, paper input, results and messages. Historical sessions restore their own paper input and selections. Cancelled search/chat responses cannot repopulate a reset conversation.

## Case-study boundaries

Case 1 now has a representative-title path and reviewer-visible scores. The learner's own titles are distinct from saved prospective mentors. A publication-topic match is not evidence of mentoring experience, availability or willingness. Domain reviewers still need to accept shortlist quality and the desired search population (Bridge2AI-only versus wider researchers).

Case 2's junior-PI/team-formation flow exists. The program-officer lens is **not implemented as a funding or impact product**. Current candidate responses have identity, publication and retrieval data, not verified award/program alignment. Chat must not infer NIH/NSF/DOD support or impact from these records.

To implement that lens, first supply a versioned source linking canonical researcher IDs to awards, agency, program, award dates/status and source URLs. Agree whether alignment means active funding, historical funding, eligibility or topical relevance. Then add deterministic filters and sourced award evidence alongside publications. Coverage views need an explicit program/cohort denominator, time window, deduplication rules and provenance. Citation reach, adoption and impact must have separate definitions. Acceptance requires reviewers to verify person-award identity, funding status and a sample of program coverage counts.

## Verification and remaining gates

- Focused compilation and mocked endpoint checks; no backend import/model loading.
- `scripts/persona-smoke.cjs` mocks every application API, using one Chromium process against local Next on 3100. Covers include/exclude, representative titles, fresh mode switches both ways, previous mentor restoration, four viewport widths (1440/1024/768/390), visible composer, horizontal overflow and browser errors.
- Browser dependencies were extracted only to `/tmp/matrix-browser-libs-wpMAg0`; no system package installation. Run with `NODE_PATH=/home/shaked/src/bridge2aikg/node_modules LD_LIBRARY_PATH=/tmp/matrix-browser-libs-wpMAg0/extracted/usr/lib/aarch64-linux-gnu node scripts/persona-smoke.cjs` from frontend. Temporary paths may expire.
- Screenshots: `/tmp/matrix-persona-desktop.png`, `/tmp/matrix-persona-mobile.png`. Synthetic people, mocked responses.
- Still required: real provider explanations, authenticated persistence/Graph iframe integration, author-preview internal-token configuration, real shortlist review, keyboard/screen-reader review beyond focus styling, and program-officer data integration. Mock tests do not establish these gates.
- No commit, push or production deployment in this work.
