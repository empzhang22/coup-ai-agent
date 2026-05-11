"use strict";

const path = require("path");
const { CoupEnv } = require("../src/coup-env");
const { HeuristicAgent, RandomMaskedAgent } = require("../src/baseline-agents");
const { encodeLegalMask, encodeObservation } = require("../src/encoder");
const { LinearPolicyAgent } = require("../src/rl-agent");
const { NeuralPPOAgent } = require("../src/neural-ppo-agent");

const args = parseArgs(process.argv.slice(2));
const games = Number(args.games || 500);
const playerCount = Number(args.players || 3);
const modelPath = path.resolve(args.model || "models/ppo-agent.json");
const seed = Number(args.seed || 5000);

const rlAgent = loadAgent(modelPath);
const results = {
  rlWins: 0,
  randomWins: 0,
  heuristicWins: 0,
  totalTurns: 0
};

for (let game = 1; game <= games; game++) {
  const env = new CoupEnv({ playerCount, seed: seed + game });
  const agents = Array.from({ length: playerCount }, (_, id) => {
    if (id === 0) return rlAgent;
    if (id % 2 === 0) return new HeuristicAgent({ seed: seed + game * 17 + id });
    return new RandomMaskedAgent({ seed: seed + game * 31 + id });
  });

  let safety = 10000;
  while (!env.gameOver && safety-- > 0) {
    const playerId = env.currentPlayerId();
    const observation = env.observe(playerId);
    const vector = encodeObservation(observation);
    const legalMask = encodeLegalMask(env, playerId);
    const agent = agents[playerId];
    const { actionIndex } = agent.act(vector, legalMask, {
      greedy: playerId === 0,
      observation
    });
    env.step(actionIndex);
  }
  if (safety <= 0) throw new Error("Evaluation game exceeded safety limit");

  const winner = env.winnerId();
  if (winner === 0) results.rlWins++;
  else if (winner % 2 === 0) results.heuristicWins++;
  else results.randomWins++;
  results.totalTurns += env.turnCount;
}

console.log(`model: ${modelPath}`);
console.log(`games: ${games}, players: ${playerCount}`);
console.log(`RL win rate: ${(100 * results.rlWins / games).toFixed(1)}%`);
console.log(`Random baseline win rate: ${(100 * results.randomWins / games).toFixed(1)}%`);
console.log(`Heuristic baseline win rate: ${(100 * results.heuristicWins / games).toFixed(1)}%`);
console.log(`Average turns: ${(results.totalTurns / games).toFixed(1)}`);

function loadAgent(modelPath) {
  const data = require(modelPath);
  if (data.type === "NeuralPPOAgent") return NeuralPPOAgent.load(modelPath);
  if (data.type === "LinearPolicyAgent") return LinearPolicyAgent.load(modelPath);
  throw new Error(`Unsupported model type in ${modelPath}: ${data.type}`);
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
