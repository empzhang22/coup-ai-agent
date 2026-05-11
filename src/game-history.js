"use strict";

class GameHistory {
  constructor() {
    this.revealedCards = new Map();
    this.actionHistory = [];
    this.failedChallenges = new Map();
    this.successfulChallenges = new Map();
  }

  recordAction(turn, playerId, action, result) {
    this.actionHistory.push({ turn, playerId, action, ...result });
  }

  recordCardReveal(playerId, card) {
    if (!this.revealedCards.has(playerId)) this.revealedCards.set(playerId, []);
    this.revealedCards.get(playerId).push(card);
  }

  recordChallenge(challengerId, claimantId, action, success) {
    const map = success ? this.successfulChallenges : this.failedChallenges;
    if (!map.has(claimantId)) map.set(claimantId, new Map());
    const actionMap = map.get(claimantId);
    actionMap.set(action, (actionMap.get(action) || 0) + 1);
  }
}

module.exports = { GameHistory };
