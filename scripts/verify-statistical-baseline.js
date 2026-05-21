"use strict";

const { CoupEnv } = require("../src/coup-env");
const { RandomMaskedAgent } = require("../src/baseline-agents");
const { StatisticalMaskedAgent } = require("../src/statistical-agent");
const { BASELINE_COMMIT } = require("../src/kqw4-statistical");

const args = parseArgs(process.argv.slice(2));
const games = Number(args.games || 2000);
const playerCount = Number(args.players || 3);
const seed = Number(args.seed || 9001);

let statisticalWins = 0;
let randomWins = 0;
let totalTurns = 0;

for (let game = 0; game < games; game++) {
  const env = new CoupEnv({ playerCount, seed: seed + game });
  const agents = Array.from({ length: playerCount }, (_, id) => {
    if (id === 0) return new StatisticalMaskedAgent({ playerId: id, seed: seed + id });
    return new RandomMaskedAgent({ seed: seed + game * 31 + id });
  });

  let safety = 20000;
  while (!env.gameOver && safety-- > 0) {
    const playerId = env.currentPlayerId();
    const action = agents[playerId].act(null, env.legalActionMask(playerId), { env });
    env.step(action.actionIndex);
  }
  if (safety <= 0) throw new Error(`Game ${game} exceeded safety limit`);

  const winner = env.winnerId();
  if (winner === 0) statisticalWins++;
  else randomWins++;
  totalTurns += env.turnCount;
}

console.log(`baseline commit: ${BASELINE_COMMIT}`);
console.log(`games: ${games}, players: ${playerCount}`);
console.log(`Statistical (seat 0) win rate: ${(100 * statisticalWins / games).toFixed(1)}%`);
console.log(`Random opponents win rate: ${(100 * randomWins / games).toFixed(1)}%`);
console.log(`Average turns: ${(totalTurns / games).toFixed(1)}`);
console.log("");
console.log("Expected: statistical >> random. Run 5000+ games in browser contest to cross-check.");

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
