"use strict";

const path = require("path");
const { CoupEnv } = require("../src/coup-env");
const { StatisticalMaskedAgent } = require("../src/statistical-agent");
const { encodeLegalMask, encodeObservation } = require("../src/encoder");
const { LinearPolicyAgent } = require("../src/rl-agent");
const { NeuralPPOAgent } = require("../src/neural-ppo-agent");
const { BASELINE_COMMIT } = require("../src/kqw4-statistical");

const args = parseArgs(process.argv.slice(2));
const games = Number(args.games || 1000);
const playerCount = Number(args.players || 3);
const seed = Number(args.seed || 7000);
const modelPath = args.model ? path.resolve(args.model) : null;

const rlAgent = modelPath ? loadAgent(modelPath) : null;
let rlWins = 0;
let statisticalWins = 0;
let totalTurns = 0;

for (let game = 0; game < games; game++) {
  const env = new CoupEnv({ playerCount, seed: seed + game });
  const agents = Array.from({ length: playerCount }, (_, id) => {
    if (id === 0 && rlAgent) return rlAgent;
    return new StatisticalMaskedAgent({ playerId: id, seed: seed + game * 17 + id });
  });

  let safety = 20000;
  while (!env.gameOver && safety-- > 0) {
    const playerId = env.currentPlayerId();
    const legalMask = encodeLegalMask(env, playerId);
    const agent = agents[playerId];

    let action;
    if (agent instanceof StatisticalMaskedAgent) {
      action = agent.act(null, legalMask, { env });
    } else {
      const observation = env.observe(playerId);
      action = agent.act(encodeObservation(observation), legalMask, {
        greedy: playerId === 0,
        observation
      });
    }
    env.step(action.actionIndex);
  }
  if (safety <= 0) throw new Error(`Game ${game} exceeded safety limit`);

  const winner = env.winnerId();
  if (winner === 0) rlWins++;
  else statisticalWins++;
  totalTurns += env.turnCount;
}

console.log(`baseline commit: ${BASELINE_COMMIT}`);
console.log(`games: ${games}, players: ${playerCount}`);
if (modelPath) {
  console.log(`model: ${modelPath}`);
  console.log(`RL win rate vs statistical: ${(100 * rlWins / games).toFixed(1)}%`);
} else {
  console.log("model: (none) — all seats statistical; use --model for RL eval");
}
console.log(`Statistical win rate: ${(100 * statisticalWins / games).toFixed(1)}%`);
console.log(`Average turns: ${(totalTurns / games).toFixed(1)}`);

function loadAgent(modelPath) {
  const data = require(modelPath);
  if (data.type === "NeuralPPOAgent") return NeuralPPOAgent.load(modelPath);
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
