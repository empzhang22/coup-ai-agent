"use strict";

const fs = require("fs");
const path = require("path");
const { CoupEnv } = require("../src/coup-env");
const { encodeLegalMask, encodeObservation } = require("../src/encoder");
const { LinearPolicyAgent } = require("../src/rl-agent");

const args = parseArgs(process.argv.slice(2));
const games = Number(args.games || 2000);
const playerCount = Number(args.players || 3);
const saveEvery = Number(args["save-every"] || 250);
const seed = Number(args.seed || 1);
const modelPath = path.resolve(args.out || "models/rl-agent.json");

fs.mkdirSync(path.dirname(modelPath), { recursive: true });

const agent = args.resume && fs.existsSync(args.resume)
  ? LinearPolicyAgent.load(path.resolve(args.resume))
  : new LinearPolicyAgent({ seed });

const stats = {
  wins: Array(playerCount).fill(0),
  totalTurns: 0,
  totalTransitions: 0
};

for (let game = 1; game <= games; game++) {
  const result = playTrainingGame(agent, { playerCount, seed: seed + game });
  stats.totalTurns += result.turns;
  stats.totalTransitions += result.transitions.length;
  if (result.winner !== null) stats.wins[result.winner]++;

  const update = agent.update(result.transitions);

  if (game % Math.max(1, Math.floor(games / 20)) === 0 || game === 1) {
    const winRates = stats.wins.map(w => `${(100 * w / game).toFixed(1)}%`).join(" / ");
    console.log(
      `game=${game} turns=${(stats.totalTurns / game).toFixed(1)} ` +
      `transitions=${Math.round(stats.totalTransitions / game)} ` +
      `wins=${winRates} policyLoss=${update.policyLoss.toFixed(4)} valueLoss=${update.valueLoss.toFixed(4)}`
    );
  }

  if (game % saveEvery === 0) {
    const snapshotPath = modelPath.replace(/\.json$/, `-${game}.json`);
    agent.save(snapshotPath);
  }
}

agent.save(modelPath);
console.log(`saved ${modelPath}`);

function playTrainingGame(agent, { playerCount, seed }) {
  const env = new CoupEnv({ playerCount, seed });
  const transitions = [];
  const latestTransitionByPlayer = Array(playerCount).fill(null);

  let safety = 10000;
  while (!env.gameOver && safety-- > 0) {
    const playerId = env.currentPlayerId();
    const observation = env.observe(playerId);
    const vector = encodeObservation(observation);
    const legalMask = encodeLegalMask(env, playerId);
    const { actionIndex } = agent.act(vector, legalMask);

    const transition = {
      playerId,
      observation: vector,
      legalMask,
      actionIndex,
      reward: 0
    };
    transitions.push(transition);
    latestTransitionByPlayer[playerId] = transition;

    const result = env.step(actionIndex);
    for (let i = 0; i < result.rewards.length; i++) {
      if (latestTransitionByPlayer[i]) latestTransitionByPlayer[i].reward += result.rewards[i];
    }
  }

  if (safety <= 0) throw new Error("Training game exceeded safety limit");
  return { transitions, winner: env.winnerId(), turns: env.turnCount };
}

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (!arg.startsWith("--")) continue;
    const key = arg.slice(2);
    const next = argv[i + 1];
    if (!next || next.startsWith("--")) out[key] = true;
    else out[key] = argv[++i];
  }
  return out;
}
