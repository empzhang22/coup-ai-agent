"use strict";

const { actionToIndex } = require("./constants");
const { legalActionIndexes } = require("./encoder");
const { StatisticalAI } = require("./kqw4-statistical");
const { envToGameState, historyForStatistical } = require("./statistical-bridge");

class StatisticalMaskedAgent {
  constructor({ playerId, seed } = {}) {
    this.playerId = playerId;
    this.ai = new StatisticalAI(playerId, "statistical");
    if (seed !== undefined) this.ai.belief._rng = seededRng(seed);
  }

  act(_observationVector, legalMask, { env } = {}) {
    if (!env) throw new Error("StatisticalMaskedAgent requires { env } in act() options");

    const playerId = env.currentPlayerId();
    const gameState = envToGameState(env);
    const gameHistory = historyForStatistical(env.history, env.playerCount);
    const player = gameState.players[playerId];
    const decision = env.decision;
    const pending = env.pendingAction;

    if (!decision) {
      return pickFirstLegal(legalMask);
    }

    if (decision.type === "main") {
      const chosen = this.ai.chooseAction(player, gameState, gameHistory);
      const actionStr = formatMainAction(chosen);
      return { actionIndex: actionToIndex(actionStr) };
    }

    if (decision.type === "challenge") {
      const actionName = pending?.action;
      const claimantId = decision.claimantId ?? pending?.playerId;
      const shouldChallenge = this.ai.decideChallengeAction(
        actionName,
        claimantId,
        gameState,
        gameHistory
      );
      const actionStr = shouldChallenge ? "challenge" : "pass";
      return { actionIndex: actionToIndex(actionStr) };
    }

    if (decision.type === "block") {
      const actionName = pending?.action;
      const actorId = decision.actorId ?? pending?.playerId;
      const shouldBlock = this.ai.decideBlockAction(actionName, actorId, gameState, gameHistory);
      if (!shouldBlock) return { actionIndex: actionToIndex("pass") };

      const claim = this.ai.decideBlockClaim(actionName, actorId, gameState, gameHistory);
      const preferred = claim ? [`block:${claim}`, "pass"] : ["pass"];
      return pickPreferredLegal(legalMask, preferred);
    }

    if (decision.type === "challenge_block") {
      const actionName = pending?.action;
      const shouldChallenge = this.ai.decideChallengeBlock(
        actionName,
        decision.blockerId,
        decision.blockChar,
        gameState,
        gameHistory
      );
      return { actionIndex: actionToIndex(shouldChallenge ? "challenge" : "pass") };
    }

    return pickFirstLegal(legalMask);
  }
}

function formatMainAction(chosen) {
  if (!chosen) return "income";
  if (chosen.targetId !== undefined && chosen.targetId !== null) {
    return `${chosen.action}:${chosen.targetId}`;
  }
  return chosen.action;
}

function pickPreferredLegal(legalMask, preferences) {
  for (const action of preferences) {
    const index = actionToIndex(action);
    if (legalMask[index]) return { actionIndex: index };
  }
  return pickFirstLegal(legalMask);
}

function pickFirstLegal(legalMask) {
  const legal = legalActionIndexes(legalMask);
  if (!legal.length) throw new Error("StatisticalMaskedAgent: no legal actions");
  return { actionIndex: legal[0] };
}

function seededRng(seed) {
  let state = (seed >>> 0) || 1;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

module.exports = { StatisticalMaskedAgent };
