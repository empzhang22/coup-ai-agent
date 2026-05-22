"use strict";

const { CoupEnv } = require("./coup-env");
const { StatisticalMaskedAgent } = require("./statistical-agent");
const { encodeLegalMask, encodeObservation } = require("./encoder");

/**
 * Play games with PPO (greedy) vs StatisticalMaskedAgent on all other seats.
 * By default the PPO seat rotates each game so eval is not tied to AI 1 / seat 0.
 */
function evaluateVsStatistical(rlAgent, { games = 1000, playerCount = 3, seed = 7000, rotateSeat = true } = {}) {
  let rlWins = 0;
  let statisticalWins = 0;
  let draws = 0;
  let totalTurns = 0;

  for (let game = 0; game < games; game++) {
    const rlSeat = rotateSeat ? (seed + game) % playerCount : 0;
    const env = new CoupEnv({ playerCount, seed: seed + game });
    const agents = Array.from({ length: playerCount }, (_, id) => {
      if (id === rlSeat && rlAgent) return rlAgent;
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
          greedy: true,
          observation
        });
      }
      env.step(action.actionIndex);
    }
    if (safety <= 0) throw new Error(`evaluateVsStatistical game ${game} exceeded safety limit`);

    const winner = env.winnerId();
    if (winner === rlSeat) rlWins++;
    else if (winner === null) draws++;
    else statisticalWins++;
    totalTurns += env.turnCount;
  }

  const decided = games - draws;
  return {
    games,
    playerCount,
    rlWins,
    statisticalWins,
    draws,
    rlWinRate: 100 * rlWins / games,
    statisticalWinRate: 100 * statisticalWins / games,
    rlWinRateDecided: decided > 0 ? 100 * rlWins / decided : 0,
    averageTurns: totalTurns / games
  };
}

function parsePlayerCounts(playersArg, defaultCount = 3) {
  if (playersArg === "all" || playersArg === undefined && defaultCount === "all") {
    return [2, 3, 4, 5];
  }
  if (typeof playersArg === "string" && playersArg.includes(",")) {
    return playersArg.split(",").map(s => Number(s.trim())).filter(n => n >= 2 && n <= 5);
  }
  const n = Number(playersArg ?? defaultCount);
  return [n];
}

module.exports = { evaluateVsStatistical, parsePlayerCounts };
