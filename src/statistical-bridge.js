"use strict";

/**
 * Converts headless CoupEnv state/history into shapes expected by kqw4 StatisticalAI.
 * revealedCards matches coup contest: sparse array indexed by player id.
 */
function historyForStatistical(history, playerCount) {
  const revealedCards = Array.isArray(history?.revealedCards)
    ? history.revealedCards
    : [];
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
