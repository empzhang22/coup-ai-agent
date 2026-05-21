"use strict";

/**
 * Converts headless CoupEnv state/history into shapes expected by kqw4 StatisticalAI.
 * Statistical code expects revealedCards as a per-player array (not a Map).
 */
function historyForStatistical(history, playerCount) {
  const revealedCards = Array.from({ length: playerCount }, () => []);
  if (history?.revealedCards) {
    for (let id = 0; id < playerCount; id++) {
      const cards = history.revealedCards.get(id);
      if (cards?.length) revealedCards[id] = cards.slice();
    }
  }
  return {
    revealedCards,
    actionHistory: history?.actionHistory || [],
    failedChallenges: history?.failedChallenges,
    successfulChallenges: history?.successfulChallenges
  };
}

function envToGameState(env) {
  return {
    players: env.players,
    currentPlayerIndex: env.currentPlayerIndex,
    turnCount: env.turnCount,
    gameOver: env.gameOver,
    deck: env.deck,
    pendingAction: env.pendingAction
  };
}

module.exports = { envToGameState, historyForStatistical };
