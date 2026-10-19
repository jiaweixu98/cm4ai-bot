# Conference chat release candidate

`fix/conference-release-20261019` starts from the pushed integration `e9742fd`.
The added fixes are `3dd20fe` (history routes in FastAPI's thread pool, bounded PDF
model admission and Word extraction off the event loop) and `8403bb0` (standalone
saved-person handlers with origin validation, cookies and correct rewrite precedence).
The existing endpoint contracts and core model instructions are retained.

The local `review/conference-ai-20261019` branch at `bf105bb` preserves the natural
examples, scientific explanations, mentor/team guidance, title/citation check and
comparison scripts. It is excluded from this release pending user review and is
not pushed. Full comparison: `/home/shaked/work/conference-review-20261016/index.html`.

The paired release record, manual acceptance checklist, nginx proposal and rollout
order are in Graph's `docs/CONFERENCE_RELEASE_2026-10-19.md`. Merge is gated on the
local walkthrough and real-provider acceptance. Production deployment requires a
separate explicit request; no production state is changed by preparing this branch.

Paired draft PRs: [MATRIX #21](https://github.com/jiaweixu98/cm4ai-bot/pull/21) and
[Graph #30](https://github.com/bridge2ai/bridge2aikg/pull/30).

Fresh verification passed: Next production build; 69 focused Python tests;
research-contract smoke; one isolated Postgres history-retention test; JavaScript
context-choice, starter, follow-up, match-note, Graph-proxy and saved-person checks.
The browser suite passed against that production build with actual development APIs
and Postgres: pagination/Retry and cross-page selection, whole-person context,
restored interest snapshots, history removal/late-save rejection, SSE busy Retry,
Stop/late-response exclusion, authenticated standalone Save/Remove, Graph agreement,
reload and mobile sizing. The local backend and production-built frontend serve the
selected core source on `8100` and `3100`; production services were not restarted.

Local logs use `/tmp/conference-release-20261019-*`. Browser replies are fixtures,
accounts and persistence use isolated development Postgres at `127.0.0.1:55432`;
real Google/ORCID and live-model acceptance are separate. Existing Python vector
file-resource and JavaScript module-type warnings remain. No new live-model or
load test was run. Both selected release diffs pass `git diff --check`.
