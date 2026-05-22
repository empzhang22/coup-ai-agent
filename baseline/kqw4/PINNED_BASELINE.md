# kqw4 statistical baseline (pinned)

Source repository: `/Users/eunalee/Documents/GitHub/coup`

| Field | Value |
|-------|--------|
| Commit | `996bde8` (core agent) + contest `statistical-ai.js` from `coup` repo (adds `decideChallengeBlock`) |
| Message | improved statistical agent; block-challenge API synced from contest |
| Author | Kilian Weinberger (kqw4@cornell.edu) |
| Date | 2026-01-25 |

Files extracted from that commit:

- `statistical-ai.js` — `ParticleBeliefTracker` + `StatisticalAI`
- `ai-engine.js` — base class

Browser copies at repo root (`statistical-ai.js`, `ai-engine.js`) match this pin.

Headless training uses `src/kqw4-statistical.js` (CommonJS wrapper around the same source).
