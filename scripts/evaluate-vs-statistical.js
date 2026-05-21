"use strict";

const path = require("path");
const { evaluateVsStatistical, parsePlayerCounts } = require("../src/eval-vs-statistical");
const { LinearPolicyAgent } = require("../src/rl-agent");
const { NeuralPPOAgent } = require("../src/neural-ppo-agent");
const { BASELINE_COMMIT } = require("../src/kqw4-statistical");
const { OBSERVATION_SIZE, ACTION_COUNT } = require("../src/encoder");

const args = parseArgs(process.argv.slice(2));
const games = Number(args.games || 5000);
const seed = Number(args.seed || 7000);
const modelPath = args.model ? path.resolve(args.model) : null;
const playerCounts = parsePlayerCounts(args.players, 3);

if (!modelPath) {
  console.error("Usage: node scripts/evaluate-vs-statistical.js --model <path> [--games 5000] [--players 3|all|2,3,4,5]");
  process.exit(1);
}

const rlAgent = loadAgent(modelPath);
console.log(`baseline commit: ${BASELINE_COMMIT}`);
console.log(`model: ${modelPath}`);
console.log(`games per player count: ${games}`);
console.log("");

let totalRlWins = 0;
let totalGames = 0;

for (const playerCount of playerCounts) {
  const stats = evaluateVsStatistical(rlAgent, { games, playerCount, seed: seed + playerCount * 100000 });
  totalRlWins += stats.rlWins;
  totalGames += stats.games;

  console.log(`--- ${playerCount} players ---`);
  console.log(`RL win rate: ${stats.rlWinRate.toFixed(1)}% (${stats.rlWins}/${stats.games})`);
  console.log(`Statistical win rate: ${stats.statisticalWinRate.toFixed(1)}%`);
  if (stats.draws > 0) {
    console.log(`Draws (max turns): ${stats.draws} — decided-only RL rate: ${stats.rlWinRateDecided.toFixed(1)}%`);
  }
  console.log(`Average turns: ${stats.averageTurns.toFixed(1)}`);
  console.log("");
}

if (playerCounts.length > 1) {
  console.log(`--- overall (${playerCounts.join(",")} player counts) ---`);
  console.log(`RL win rate: ${(100 * totalRlWins / totalGames).toFixed(1)}% (${totalRlWins}/${totalGames})`);
}

function loadAgent(modelPath) {
  const data = require(modelPath);
  if (data.type === "NeuralPPOAgent") {
    if (data.observationSize !== OBSERVATION_SIZE || data.actionCount !== ACTION_COUNT) {
      throw new Error(
        `Model incompatible: obs ${data.observationSize} vs ${OBSERVATION_SIZE}, ` +
        `actions ${data.actionCount} vs ${ACTION_COUNT}. Retrain after reveal-parity change.`
      );
    }
    return NeuralPPOAgent.load(modelPath);
  }
  if (data.type === "LinearPolicyAgent") return LinearPolicyAgent.load(modelPath);
  throw new Error(`Unsupported model type: ${data.type}`);
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
