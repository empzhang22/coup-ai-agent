"use strict";

const {
  ACTIONS,
  CHARACTERS,
  MAX_PLAYERS,
  blockCardsForAction,
  indexToAction,
  requiredCardForAction
} = require("./constants");
const { GameHistory } = require("./game-history");
const { createRng, shuffle } = require("./rng");

class CoupEnv {
  constructor({ playerCount = 3, seed = Date.now(), maxTurns = 500, log = false } = {}) {
    if (playerCount < 2 || playerCount > MAX_PLAYERS) {
      throw new Error(`playerCount must be between 2 and ${MAX_PLAYERS}`);
    }
    this.playerCount = playerCount;
    this.maxTurns = maxTurns;
    this.logEnabled = log;
    this.rng = typeof seed === "function" ? seed : createRng(seed);
    this.reset();
  }

  reset() {
    this.deck = [];
    for (const character of CHARACTERS) {
      for (let i = 0; i < 3; i++) this.deck.push(character);
    }
    shuffle(this.deck, this.rng);

    this.players = Array.from({ length: this.playerCount }, (_, id) => ({
      id,
      name: `Player ${id + 1}`,
      coins: 2,
      cards: [
        { character: this.deck.pop(), revealed: false },
        { character: this.deck.pop(), revealed: false }
      ],
      eliminated: false,
      eliminatedOnTurn: null
    }));

    this.currentPlayerIndex = Math.floor(this.rng() * this.playerCount);
    this.turnCount = 0;
    this.gameOver = false;
    this.pendingAction = null;
    this.decision = { type: "main", playerId: this.currentPlayerIndex };
    this.history = new GameHistory();
    this.eventLog = [];
    this.lastRewards = Array(this.playerCount).fill(0);
    return this.observe(this.decision.playerId);
  }

  clonePublicState() {
    return {
      players: this.players,
      currentPlayerIndex: this.currentPlayerIndex,
      deck: this.deck,
      pendingAction: this.pendingAction,
      gameOver: this.gameOver,
      turnCount: this.turnCount,
      history: this.history
    };
  }

  currentPlayerId() {
    return this.decision?.playerId ?? this.currentPlayerIndex;
  }

  activePlayers() {
    return this.players.filter(p => !p.eliminated);
  }

  legalActions(playerId = this.currentPlayerId()) {
    if (this.gameOver || !this.decision || this.decision.playerId !== playerId) return [];

    if (this.decision.type === "main") {
      const player = this.players[playerId];
      if (!player || player.eliminated) return [];

      const actions = ["income"];
      const targets = this.players.filter(p => p.id !== playerId && !p.eliminated);

      if (player.coins >= 10) return targets.map(t => `coup:${t.id}`);
      if (player.coins >= 7) actions.push(...targets.map(t => `coup:${t.id}`));

      actions.push("foreign-aid", "tax", "exchange");
      if (player.coins >= 3) actions.push(...targets.map(t => `assassinate:${t.id}`));
      actions.push(...targets.map(t => `steal:${t.id}`));
      return actions;
    }

    if (this.decision.type === "challenge") {
      return ["pass", "challenge"];
    }

    if (this.decision.type === "block") {
      return ["pass", ...blockCardsForAction(this.pendingAction.action).map(card => `block:${card}`)];
    }

    return [];
  }

  legalActionMask(playerId = this.currentPlayerId()) {
    const mask = Array(ACTIONS.length).fill(false);
    for (const action of this.legalActions(playerId)) {
      const index = ACTIONS.indexOf(action);
      if (index !== -1) mask[index] = true;
    }
    return mask;
  }

  observe(playerId = this.currentPlayerId()) {
    const revealedCounts = Object.fromEntries(CHARACTERS.map(c => [c, 0]));
    for (const cards of this.history.revealedCards) {
      if (!cards) continue;
      for (const card of cards) revealedCounts[card]++;
    }

    return {
      selfId: playerId,
      currentPlayerIndex: this.currentPlayerIndex,
      decision: this.decision ? { ...this.decision } : null,
      turnCount: this.turnCount,
      players: this.players.map(player => ({
        id: player.id,
        coins: player.coins,
        influence: this.activeInfluence(player),
        eliminated: player.eliminated,
        revealed: player.cards.filter(c => c.revealed).map(c => c.character),
        ownCards: player.id === playerId
          ? player.cards.filter(c => !c.revealed).map(c => c.character)
          : undefined
      })),
      revealedCounts,
      actionHistoryLength: this.history.actionHistory.length
    };
  }

  step(actionInput) {
    if (this.gameOver) return this.result();

    const actorId = this.currentPlayerId();
    const legal = this.legalActions(actorId);
    const action = typeof actionInput === "number" ? indexToAction(actionInput) : actionInput;
    if (!legal.includes(action)) {
      throw new Error(`Illegal action "${action}" for ${this.decision.type} by player ${actorId}. Legal: ${legal.join(", ")}`);
    }

    this.lastRewards = Array(this.playerCount).fill(0);

    if (this.decision.type === "main") this.applyMainAction(action);
    else if (this.decision.type === "challenge") this.applyChallengeDecision(action);
    else if (this.decision.type === "block") this.applyBlockDecision(action);
    else {
      throw new Error(`Unexpected decision type: ${this.decision.type}`);
    }

    if (!this.gameOver && this.turnCount >= this.maxTurns) {
      this.gameOver = true;
      this.decision = null;
    }

    return this.result();
  }

  result() {
    return {
      observation: this.gameOver ? null : this.observe(this.currentPlayerId()),
      rewards: this.lastRewards.slice(),
      done: this.gameOver,
      winner: this.winnerId(),
      info: {
        turnCount: this.turnCount,
        decision: this.decision ? { ...this.decision } : null
      }
    };
  }

  applyMainAction(actionId) {
    const parsed = parseAction(actionId);
    const player = this.players[this.currentPlayerIndex];
    this.turnCount++;

    this.pendingAction = {
      action: parsed.action,
      playerId: player.id,
      targetId: parsed.targetId ?? null,
      challengeOrder: this.shuffledOtherPlayerIds(player.id),
      challengeCursor: 0,
      blockOrder: [],
      blockCursor: 0,
      blockClaim: null,
      blockPlayerId: null
    };

    this.log(`Turn ${this.turnCount}: Player ${player.id + 1} chooses ${actionId}`);

    if (requiredCardForAction(parsed.action)) {
      this.advanceChallengeOrResolution();
      return;
    }

    this.advanceBlockOrResolution();
  }

  applyChallengeDecision(action) {
    if (action === "pass") {
      this.pendingAction.challengeCursor++;
      this.advanceChallengeOrResolution();
      return;
    }

    const challengerId = this.decision.playerId;
    const claimantId = this.pendingAction.playerId;
    const actionName = this.pendingAction.action;
    const challengeStopsAction = !this.claimantHasRequiredCard(claimantId, actionName);
    this.history.recordChallenge(challengerId, claimantId, actionName, challengeStopsAction);

    if (challengeStopsAction) {
      this.history.recordAction(this.turnCount, claimantId, this.pendingAction.action, {
        challenged: true,
        success: false
      });
      this.resolveSuccessfulChallenge(challengerId, claimantId, actionName);
    } else {
      this.history.recordAction(this.turnCount, claimantId, this.pendingAction.action, {
        challenged: true,
        success: true
      });
      this.resolveFailedChallenge(challengerId, claimantId, actionName);
    }
  }

  applyBlockDecision(action) {
    if (action === "pass") {
      this.pendingAction.blockCursor++;
      this.advanceBlockOrResolution();
      return;
    }

    const card = action.split(":")[1];
    this.pendingAction.blockClaim = card;
    this.pendingAction.blockPlayerId = this.decision.playerId;
    this.history.recordAction(this.turnCount, this.pendingAction.playerId, this.pendingAction.action, {
      blocked: true,
      blockClaim: card,
      success: false
    });
    this.pendingAction = null;
    this.nextTurn();
  }

  advanceChallengeOrResolution() {
    while (this.pendingAction.challengeCursor < this.pendingAction.challengeOrder.length) {
      const challengerId = this.pendingAction.challengeOrder[this.pendingAction.challengeCursor];
      if (!this.players[challengerId].eliminated) {
        this.decision = { type: "challenge", playerId: challengerId, claimantId: this.pendingAction.playerId };
        return;
      }
      this.pendingAction.challengeCursor++;
    }
    this.advanceBlockOrResolution();
  }

  advanceBlockOrResolution() {
    const action = this.pendingAction.action;
    if (action === "foreign-aid" && this.pendingAction.blockOrder.length === 0) {
      this.pendingAction.blockOrder = this.shuffledOtherPlayerIds(this.pendingAction.playerId);
    } else if (["assassinate", "steal"].includes(action) && this.pendingAction.blockOrder.length === 0) {
      const target = this.pendingAction.targetId;
      this.pendingAction.blockOrder = target !== null && !this.players[target].eliminated ? [target] : [];
    }

    while (this.pendingAction.blockCursor < this.pendingAction.blockOrder.length) {
      const blockerId = this.pendingAction.blockOrder[this.pendingAction.blockCursor];
      if (!this.players[blockerId].eliminated) {
        this.decision = { type: "block", playerId: blockerId, actorId: this.pendingAction.playerId };
        return;
      }
      this.pendingAction.blockCursor++;
    }

    this.history.recordAction(this.turnCount, this.pendingAction.playerId, this.pendingAction.action, {
      challenged: false,
      blocked: false,
      success: true
    });
    this.resolveAction();
  }

  claimantHasRequiredCard(claimantId, action) {
    const claimant = this.players[claimantId];
    const required = requiredCardForAction(action);
    return claimant.cards.some(c => !c.revealed && c.character === required);
  }

  resolveFailedChallenge(challengerId, claimantId, action) {
    const claimant = this.players[claimantId];
    const challenger = this.players[challengerId];
    const required = requiredCardForAction(action);
    const cardIndex = claimant.cards.findIndex(c => !c.revealed && c.character === required);

    this.log(`Challenge by Player ${challengerId + 1} fails`);
    this.replaceClaimedCard(claimant, cardIndex);
    this.lastRewards[claimantId] += 0.05;
    this.lastRewards[challengerId] -= 0.05;
    this.forceReveal(challenger, () => this.advanceBlockOrResolution());
  }

  resolveSuccessfulChallenge(challengerId, claimantId, action) {
    const claimant = this.players[claimantId];
    this.log(`Challenge by Player ${challengerId + 1} succeeds`);
    if (action === "assassinate") claimant.coins += 3;
    this.lastRewards[claimantId] -= 0.05;
    this.lastRewards[challengerId] += 0.05;
    this.forceReveal(claimant, () => {
      this.pendingAction = null;
      this.checkGameOver();
      if (!this.gameOver) this.nextTurn();
    });
  }

  resolveAction() {
    const { action, playerId, targetId } = this.pendingAction;
    const player = this.players[playerId];

    if (action === "income") player.coins += 1;
    else if (action === "foreign-aid") player.coins += 2;
    else if (action === "tax") player.coins += 3;
    else if (action === "coup") {
      player.coins -= 7;
      this.forceReveal(this.players[targetId], () => this.afterActionResolved());
      return;
    } else if (action === "assassinate") {
      player.coins -= 3;
      this.forceReveal(this.players[targetId], () => this.afterActionResolved());
      return;
    } else if (action === "steal") {
      const target = this.players[targetId];
      const stolen = Math.min(2, target.coins);
      target.coins -= stolen;
      player.coins += stolen;
    } else if (action === "exchange") {
      this.exchangeCards(player);
    }

    this.afterActionResolved();
  }

  afterActionResolved() {
    this.pendingAction = null;
    this.checkGameOver();
    if (!this.gameOver) this.nextTurn();
  }

  forceReveal(player, afterReveal) {
    const index = player.cards.findIndex(c => !c.revealed);
    if (index === -1) {
      afterReveal();
      return;
    }
    this.revealCard(player, index);
    afterReveal();
  }

  revealCard(player, index) {
    const card = player.cards[index];
    if (!card || card.revealed) throw new Error(`Player ${player.id} cannot reveal card ${index}`);
    card.revealed = true;
    this.history.recordCardReveal(player.id, card.character);
    this.lastRewards[player.id] -= 0.03;
    this.checkElimination(player);
  }

  replaceClaimedCard(player, index) {
    this.deck.push(player.cards[index].character);
    shuffle(this.deck, this.rng);
    player.cards[index].character = this.deck.pop();
  }

  exchangeCards(player) {
    const activeIndexes = player.cards
      .map((card, index) => (!card.revealed ? index : null))
      .filter(index => index !== null);
    for (const index of activeIndexes) this.deck.push(player.cards[index].character);
    shuffle(this.deck, this.rng);
    for (const index of activeIndexes) {
      if (this.deck.length) player.cards[index].character = this.deck.pop();
    }
  }

  nextTurn() {
    if (this.checkGameOver()) return;
    do {
      this.currentPlayerIndex = (this.currentPlayerIndex + 1) % this.playerCount;
    } while (this.players[this.currentPlayerIndex].eliminated);
    this.decision = { type: "main", playerId: this.currentPlayerIndex };
  }

  checkElimination(player) {
    if (this.activeInfluence(player) === 0 && !player.eliminated) {
      player.eliminated = true;
      player.eliminatedOnTurn = this.turnCount;
      player.coins = 0;
      this.lastRewards[player.id] -= 0.25;
    }
  }

  checkGameOver() {
    const active = this.activePlayers();
    if (active.length === 1) {
      this.gameOver = true;
      this.decision = null;
      this.lastRewards[active[0].id] += 1;
      for (const player of this.players) {
        if (player.id !== active[0].id) this.lastRewards[player.id] -= 1;
      }
      return true;
    }
    return false;
  }

  winnerId() {
    const active = this.activePlayers();
    return this.gameOver && active.length === 1 ? active[0].id : null;
  }

  activeInfluence(player) {
    return player.cards.filter(c => !c.revealed).length;
  }

  shuffledOtherPlayerIds(playerId) {
    return shuffle(
      this.players.filter(p => p.id !== playerId && !p.eliminated).map(p => p.id),
      this.rng
    );
  }

  log(message) {
    if (this.logEnabled) this.eventLog.push(message);
  }
}

function parseAction(actionId) {
  const [action, target] = actionId.split(":");
  return {
    action,
    targetId: target === undefined ? null : Number(target)
  };
}

module.exports = { CoupEnv, parseAction };
