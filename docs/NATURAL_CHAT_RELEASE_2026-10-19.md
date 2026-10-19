# Natural chat release candidate

October 19, 2026. Branch: `fix/conference-natural-questions`, based on deployed
MATRIX main `483864f`. The user selected **natural question + focused explanation**
in `/home/shaked/work/conference-review-20261016/index.html`.

## Included behavior

- Six editable scientific questions; personal context travels in structured fields.
- Short reasons describe supported work and connect it to the specific request.
  Methods/results need source text; material scientific qualifications remain.
- Mentor guidance focuses on the requested learning skill and avoids unsolicited
  bibliometrics. Team evidence reading preserves assembled members and role order.
- A named, quoted publication already known to the agent must cite that publication.
- Added researchers remain separate from the viewer in reasons and follow-ups.

No index, scoring algorithm, database schema, deployment configuration or data
snapshot changes. Existing retrieval order constraints remain. Mentor query guidance
can change the search requirements; model wording and shortlist composition can vary.
The Graph touch candidate and archived broader agent/retrieval experiments are excluded.

## Fresh checks

Evidence: `/home/shaked/work/natural-chat-release-20261019/`.

| Check | Result |
| --- | --- |
| Backend regression suite | 71 tests passed: concurrency, attachment admission, exact-title citation, recovery, context, sessions, person lookup, paper discovery. |
| Isolated development history retention | 1 test passed against PostgreSQL on port 55432. |
| Research contract and frontend focused checks | Passed contract smoke; context choices/starters, follow-ups, match notes, Graph account proxy and saved people checks. |
| Frontend production build | Passed with existing dependencies. |
| Six exact visible starters, browser and live model | Passed routing, scope/count, role rendering, evidence-bearing cards, guest save boundary, profile action and no page exceptions. Explore used an explicit URL identity, not OAuth. |
| Five additional live-model cases | Passed guest selected-person identity; follow-up keeps cards and cites the displayed paper; explicit empty paper scope; selected-paper audience; attached-proposal audience. |
| Catalog evidence | All 40 displayed cards and 67 author–paper pairs passed ownership/eligibility. Every linked and guest Explore paper was checked for both coauthors. |
| Scientific review | Read the fresh card reasons against their source abstracts. Methods/settings stayed supported; adjacent matches retained their qualifications. Citation existence was checked separately. |
| Browser regression with test identity and model fixtures | Passed actual local API/DB pagination, cross-page selection, Retry, whole-person context, topic snapshots, chat hiding/late writes, Stop, account save/remove/reload and mobile fit. |

The development and production manifests reference identical `papers.sqlite` and
paper-vector checksums. Development keeps its own snapshot pointer, Graph metadata
and review state; this is scientific-catalog parity, not full runtime/data parity.

## Observations and remaining acceptance

- Team implementation returned a cardiac-surgery decision-support analogue, not a
  demonstrated multicenter sepsis deployment. Its reason and framing preserved that
  distinction. Stronger role coverage remains a later retrieval-quality review.
- Two mentor cards cite the FAIR principles paper. One collaborator's catalog display
  name is only “Kwak.” These are diversity/name-quality review items, not runtime
  failures; do not guess or merge identities to improve presentation.
- The six starters and five context cases are fresh single-run acceptance, not an
  exhaustive review of every follow-up, attachment format or model outcome. Niche
  counts cite external OpenAlex searches and were not independently recounted here.
- Real Google/ORCID callbacks and real-account saved/history behavior need a human
  acceptance pass. Browser test identities do not establish provider acceptance.
- Several real accounts should rehearse together on shared Wi-Fi. No attendee-capacity
  benchmark or fairness guarantee has been established.
- Candidate is locally validated. Main and production remain at `483864f`; public
  mounted-route smoke belongs to the eventual controlled release.

See [CHAT_EXAMPLES_AND_REVIEW_RULES.md](CHAT_EXAMPLES_AND_REVIEW_RULES.md) for the
full scenario inventory and `/home/shaked/docs/CONFERENCE_REVIEW_ITEMS_2026-10-19.md`
for the concise conference checklist.
