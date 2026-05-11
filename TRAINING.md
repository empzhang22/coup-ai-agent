# Coup RL Training

This project now has a headless Node environment for self-play reinforcement learning. It keeps the simplified Coup rules from `aicontest.html`, but moves them into code that can run many games without the browser.

## Files

- `src/coup-env.js`: game rules, turn phases, challenges, blocks, influence loss, and terminal rewards.
- `src/encoder.js`: converts imperfect-information observations into fixed-size numeric vectors and legal-action masks.
- `src/neural-ppo-agent.js`: neural actor-critic PPO agent with masked actions.
- `src/rl-agent.js`: lightweight linear learner kept for smoke tests and comparison.
- `scripts/train-ppo.js`: serious self-play training loop with frozen snapshots and baseline opponents.
- `scripts/train-rl.js`: simple linear training loop.
- `scripts/export-browser-model.js`: converts a trained PPO JSON checkpoint into `models/ppo-browser-model.js` for `aicontest.html`.
- `rl-ai.js`: browser `AIEngine` wrapper that lets the trained model compete in the existing contest page.
- `scripts/evaluate-rl.js`: evaluates a saved model against random and heuristic baselines.
- `scripts/smoke-random-games.js`: fast simulation sanity check.

## Commands

Run tests:

```bash
npm test
```

Smoke-test the environment:

```bash
npm run smoke
```

Train a quick PPO smoke model:

```bash
npm run train -- --iterations 5 --games-per-iteration 16 --players 3 --out models/ppo-agent.json
```

Evaluate it:

```bash
npm run evaluate -- --model models/ppo-agent.json --games 1000 --players 3
```

Run a serious local training job:

```bash
npm run train -- --iterations 300 --games-per-iteration 128 --players 3 --epochs 4 --batch-size 256 --out models/ppo-agent.json
```

Resume training:

```bash
npm run train -- --resume models/ppo-agent.json --iterations 300 --games-per-iteration 128 --players 3 --out models/ppo-agent.json
```

If `npm` is not on your PATH, run the same scripts directly:

```bash
node scripts/train-ppo.js --iterations 300 --games-per-iteration 128 --players 3 --out models/ppo-agent.json
node scripts/evaluate-rl.js --model models/ppo-agent.json --games 1000 --players 3
```

## What Counts As "Real" Training

The PPO trainer uses a neural actor-critic policy, legal action masking, generalized advantage estimation, clipped PPO updates, entropy regularization, gradient clipping, frozen self-play snapshots, and random/heuristic baseline opponents.

For a serious MEng run, train multiple seeds and compare them:

```bash
node scripts/train-ppo.js --seed 1 --iterations 500 --games-per-iteration 192 --out models/ppo-seed-1.json
node scripts/train-ppo.js --seed 2 --iterations 500 --games-per-iteration 192 --out models/ppo-seed-2.json
node scripts/train-ppo.js --seed 3 --iterations 500 --games-per-iteration 192 --out models/ppo-seed-3.json
```

Then evaluate each with at least 5,000 games. Pick the checkpoint with the best win rate against the fixed baseline mix, not the one with the best training log.

## Use In The Browser Contest

After training, export the model for `aicontest.html`:

```bash
node scripts/export-browser-model.js --model models/ppo-agent.json --out models/ppo-browser-model.js
```

Then open `aicontest.html` and choose `RL PPO AI (Trained Model)` for one or more players.

The repository includes a tiny smoke-trained browser model so the dropdown works immediately. It is not competitive. Replace it by exporting a long-trained checkpoint before using the agent in your MEng comparisons.
