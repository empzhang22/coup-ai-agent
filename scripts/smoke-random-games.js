"use strict";

const { CoupEnv } = require("../src/coup-env");
const { RandomMaskedAgent } = require("../src/baseline-agents");
const { encodeLegalMask, encodeObservation } = require("../src/encoder");

const games = Number(process.argv[2] || 100);
const playerCount = Number(process.argv[3] || 3);
const agents = Array.from({ length: playerCount }, (_, id) => new RandomMaskedAgent({ seed: 100 + id }));
let turns = 0;
const wins = Array(playerCount).fill(0);

for (let game = 0; game < games; game++) {
  const env = new CoupEnv({ playerCount, seed: 1000 + game });
  let safety = 10000;
  while (!env.gameOver && safety-- > 0) {
    const playerId = env.currentPlayerId();
    const observation = env.observe(playerId);
    const action = agents[playerId].act(encodeObservation(observation), encodeLegalMask(env, playerId));
    env.step(action.actionIndex);
  }
  if (safety <= 0) throw new Error(`Game ${game} exceeded safety limit`);
  wins[env.winnerId()]++;
  turns += env.turnCount;
}

console.log(`simulated ${games} games with ${playerCount} players`);
console.log(`wins: ${wins.join(" / ")}`);
console.log(`average turns: ${(turns / games).toFixed(1)}`);
