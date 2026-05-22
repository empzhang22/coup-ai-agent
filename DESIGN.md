# Coup PPO Agent — Design Document

This document explains what Proximal Policy Optimization (PPO) reinforcement learning is, how this project uses it, how the pieces fit together, and why we chose PPO over alternatives such as REINFORCE.

---

## 1. Problem statement

**Goal:** Build an agent for the Cornell Coup AI contest that learns **only from self-play and fixed baselines** (no human labels), and **beats the kqw4 Statistical agent** in 3–5 player games.

**Constraints:**

- Rules and flow must match `aicontest.html` in the `coup` repo (block challenges, reveal timing, 2-player coin rule, etc.).
- The agent may be assigned to **any seat** (AI 1–5), not a fixed player index.
- Training runs headless in Node; the final policy is deployed in the browser via `ppo-ai.js` and `models/ppo-browser-model.js`.

---

## 2. What is reinforcement learning here?

In Coup, each decision (main action, challenge, block, challenge block) is a **partially observable** choice with delayed reward (win/loss at game end). We model this as a **Markov decision process**:

| Component | In this project |
|-----------|-----------------|
| **State / observation** | Fixed-length vector: decision type, seats, coins, influence, revealed cards, own hidden roles, global reveal counts, turn index (`src/encoder.js`, mirrored in `ppo-ai.js`) |
| **Action** | One of 25 discrete actions (income, tax, targeted coup/steal/assassinate, pass, challenge, blocks), restricted by a **legal mask** |
| **Policy** | Neural network → softmax over **legal** actions only |
| **Reward** | Sparse: +1 win, 0 loss (and per-step 0 with terminal credit assignment via GAE) |

The policy is **learned by trial and error**, not by imitating logged games.

---

## 3. What is PPO?

**PPO (Proximal Policy Optimization)** is an on-policy actor–critic algorithm. It improves a stochastic policy while limiting how much the policy can change per update (the **trust region**), which stabilizes learning in noisy multi-agent games.

### 3.1 Actor–critic

- **Actor (policy network):** Given observation \(s\), outputs probabilities over legal actions \(a\).
- **Critic (value network):** Estimates \(V(s)\) — expected return from this state. Used to compute **advantages** \(A_t = \text{return}_t - V(s_t)\) so the actor knows whether an action was better or worse than average.

Both are small fully connected nets (96 hidden units, `tanh`) in `src/neural-ppo-agent.js`.

### 3.2 Generalized Advantage Estimation (GAE)

After each game, we walk backward over the learner’s transitions with discount \(\gamma = 0.99\) and GAE \(\lambda = 0.94\), then normalize advantages across the batch. This reduces variance compared to raw Monte Carlo returns.

### 3.3 Clipped surrogate objective

For each stored transition we have old log-probability \(\log \pi_{\text{old}}(a|s)\). After an update pass, we compute ratio \(r = \exp(\log \pi_{\text{new}} - \log \pi_{\text{old}})\). PPO maximizes:

\[
\min\left(r \cdot A,\; \text{clip}(r,\, 1-\epsilon,\, 1+\epsilon) \cdot A\right)
\]

with \(\epsilon = 0.2\). Large policy jumps are clipped, which prevents collapse when opponents or the policy shift quickly.

### 3.4 Entropy bonus

We subtract \(\beta \cdot H(\pi)\) (\(\beta = 0.01\)) from the policy loss to keep exploration early in training.

### 3.5 Legal action masking

Illegal actions get logit \(-\infty\) before softmax. The policy never assigns probability to illegal moves in training or at inference.

---

## 4. Architecture in this repository

```text
┌─────────────────────────────────────────────────────────────────┐
│  Training (Node)                                                │
│  scripts/train-ppo.js                                           │
│    → CoupEnv (src/coup-env.js)                                  │
│    → encodeObservation / encodeLegalMask (src/encoder.js)       │
│    → NeuralPPOAgent.act / .update (src/neural-ppo-agent.js)     │
│    → opponents: Random, Statistical, frozen self-play snapshots│
│    → checkpoint: models/ppo-agent-v2-best-statistical.json      │
└────────────────────────────┬────────────────────────────────────┘
                             │ export-browser-model.js
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Deployment (browser)                                           │
│  models/ppo-browser-model.js  →  window.COUP_PPO_MODEL          │
│  ppo-ai.js (PPOAI)          →  aicontest.html contest           │
└─────────────────────────────────────────────────────────────────┘
```

### 4.1 Headless environment (`src/coup-env.js`)

Runs the same phased flow as the contest: main action → challenges → blocks → block challenges → resolution. Observations and masks are produced for the **current deciding player**.

### 4.2 Encoder (`src/encoder.js`)

**Observation size:** 75 floats (4 decision types + 5 current seat + 5 self seat + 5×10 per-seat features + 5 own-role counts + 5 global reveal counts + turn).

**Actions:** 25 indices in `src/constants.js` (shared with browser `PPO_ACTIONS`).

### 4.3 Training loop (`scripts/train-ppo.js`)

Each iteration:

1. Play `games-per-iteration` games (default 192) with **3 players**.
2. **Learner seat:** uniform random `0 .. playerCount-1` each game (any contest slot).
3. **Opponent curriculum** (by iteration progress):
   - Early: 50% random / 30% self-play snapshot / 20% statistical
   - Late: 10% / 25% / **65% statistical**
4. Collect transitions only for the learner; terminal win/loss applied via GAE.
5. PPO update: 4 epochs, batch size 128.
6. Every `eval-every` iterations: `evaluateVsStatistical` with **rotating PPO seat**; save best checkpoint as `*-best-statistical.json`.

Self-play uses **frozen snapshots** of past policies (not the live weights), so the learner does not train against identical copies of itself.

### 4.4 Browser inference (`ppo-ai.js`)

`PPOAI` extends `AIEngine` and implements contest callbacks: `chooseAction`, `decideChallengeAction`, `decideChallengeBlock`, `decideBlockAction`, `decideBlockClaim`.

- Loads weights from `window.COUP_PPO_MODEL` (exported JSON).
- Uses **greedy** argmax over legal actions (matches eval `greedy: true`).
- Observation encoding matches `src/encoder.js` so train and deploy stay aligned.

---

## 5. Why PPO instead of REINFORCE?

| Aspect | REINFORCE (Monte Carlo policy gradient) | PPO (this project) |
|--------|----------------------------------------|---------------------|
| Updates | Full trajectory, high variance | Batched with baseline (critic) + GAE |
| Step size | Sensitive; one bad batch can ruin policy | Clipped ratio limits per-update change |
| Sample efficiency | Poor; often discards data after one pass | Reuses rollouts for multiple epochs |
| Non-stationarity | Opponents change (self-play, curriculum) | Clipping + value net helps stability |
| Implementation | Simpler | Moderate; we already share nets with masked softmax |

Coup games are **long, sparse-reward, multi-agent, and non-stationary** (curriculum + snapshots). REINFORCE can work in theory but typically needs more samples and careful learning-rate tuning. PPO is the standard choice for stable policy-gradient training in similar environments.

We also ship `src/rl-agent.js` (linear policy) for smoke tests — not competitive with Statistical.

---

## 6. Why PPO instead of the contest Neural AI?

The `coup` repo includes a **supervised Neural AI** (`neural-ai.js`) trained from game logs / online learning in the browser. That is a different paradigm:

| | Contest Neural AI | Our PPO agent |
|---|-------------------|---------------|
| Learning signal | Imitation / online labels from play | Win/loss + policy gradient |
| Opponents during training | Whatever appears in logs | Controlled: random, statistical, snapshots |
| Objective | Match recorded actions | Maximize win rate vs baselines |
| Deployment | localStorage weights, trains during sim | Fixed checkpoint from `train:full` |
| MEng story | “We copied their net” | “We trained RL only via self-play + statistical curriculum” |

We chose **PPO RL** to meet the project requirement of **learning without human demonstrations** and to directly optimize performance against the **Statistical** baseline, which is the course’s reference opponent.

---

## 7. Design choices and known limitations

### 7.1 Seat encoding

Observations include a one-hot **self seat** (0–4). Training **rotates** the learner seat each game so the policy works on any AI slot in the contest. All-PPO lobbies (three copies of the same bot) are **not** a contest format and are not a training target — identical policies with different seat inputs can show skewed win rates.

### 7.2 Player count

Primary training uses **3 players** (matches main benchmark). The policy generalizes reasonably to 4–5 players; **2-player** duels are weaker (different meta, under-trained).

### 7.3 Deterministic contest play

Browser inference is **greedy** (max probability). Training samples stochastically. Eval uses greedy. This is standard for deployment and reduces variance in reported win rates.

### 7.4 Contest vs headless

Headless win rate vs statistical can exceed browser rate if files are stale or encodings diverge. **Browser results** with fresh `ppo-ai.js` + `models/ppo-browser-model.js` are the authoritative contest metric.

### 7.5 Checkpoints

- `models/ppo-agent-v2-best-statistical.json` — best vs statistical during training (use for export/reporting).
- `models/ppo-browser-model.js` — browser bundle (`COUP_PPO_MODEL`).
- Observation size **75** after `challenge_block` was added; older 74-dim checkpoints are incompatible.

---

## 8. Evaluation protocol (recommended)

1. **Primary:** 3 players, one **PPO**, others **Statistical**, 5000 games, PPO on each seat (1–5) if reporting seat invariance.
2. **Secondary:** Mixed tables (Challenger, 4–5 players) as generalization.
3. **Avoid:** All PPO or multiple PPOs in one game as headline metrics.
4. **Headless sanity:** `npm run evaluate:statistical -- --model <best> --games 500 --players 3`

---

## 9. File reference

| File | Role |
|------|------|
| `src/coup-env.js` | Game engine |
| `src/encoder.js` | Observations and masks |
| `src/neural-ppo-agent.js` | PPO learner |
| `scripts/train-ppo.js` | Training |
| `scripts/export-browser-model.js` | JSON → `COUP_PPO_MODEL` |
| `ppo-ai.js` | Browser `PPOAI` |
| `models/ppo-browser-model.js` | Exported weights |
| `src/statistical-agent.js` | kqw4 statistical opponent |

---

## 10. References

- Schulman et al., *Proximal Policy Optimization Algorithms* (2017)
- Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (2016)
- Course Coup contest: `coup/aicontest.html`, kqw4 Statistical baseline
