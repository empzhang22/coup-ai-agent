// policy.js
// Belief-aware rollout policy.
// Defines global: CoupPolicy
//
// Expected globals:
// - CHARACTERS
// - ParticleBeliefTracker

class CoupPolicy {
  constructor(playerId, belief, opts = {}) {
    this.playerId = playerId;
    this.belief = belief; // ParticleBeliefTracker
    this.rng = opts.rng ?? Math.random;

    // Rollout randomness
    this.epsilonExplore = opts.epsilonExplore ?? 0.12; // random action
    this.epsilonBluff = opts.epsilonBluff ?? 0.18;    // occasionally claim without having
  }

  // ---------- Core API used by Challenger rollouts ----------

  chooseAction(playerIdx, gameState, gameHistory) {
    const player = gameState.players[playerIdx];
    const actions = this.getPossibleActions(player, gameState);

    if (actions.length === 0) return { action: "income" };
    if (actions.length === 1) return actions[0];

    // exploration
    if (this.rng() < this.epsilonExplore) {
      return actions[Math.floor(this.rng() * actions.length)];
    }

    // score
    let best = actions[0];
    let bestScore = -Infinity;
    for (const a of actions) {
      const s = this.scoreAction(playerIdx, a, gameState, gameHistory);
      if (s > bestScore) {
        bestScore = s;
        best = a;
      }
    }
    return best;
  }

  shouldChallenge(challengerIdx, claimantIdx, actionName, gameState, gameHistory) {
    const required = this.requiredCardForAction(actionName);
    if (!required) return false;

    // if claimant is on 1 influence, they bluff less BUT challenger also risks endgame—still, bluffs get punished hard
    const claimant = gameState.players[claimantIdx];
    const challenger = gameState.players[challengerIdx];
    if (!claimant || !challenger) return false;

    const pHas = this.belief.probHas(claimantIdx, required);
    const myInf = (challenger.cards || []).filter(c => !c.revealed).length;
    const theirInf = (claimant.cards || []).filter(c => !c.revealed).length;

    // revealed count → challenges rise sharply
    const revealedCount = this.countRevealed(required, gameHistory);
    const base = [0.20, 0.30, 0.70, 1.00][Math.min(3, revealedCount)] ?? 0.25;

    // if we think they likely have it, don’t challenge
    let challengeProb = base * (1 - pHas);

    // special: target at 1 influence tends to challenge bluffs, so people bluff less; we challenge more when it matters
    if (actionName === "assassinate" && theirInf === 1) challengeProb *= 1.35;

    // if we're at 1 influence, avoid coinflip challenges unless very confident
    if (myInf === 1) challengeProb *= 0.65;

    return this.rng() < Math.max(0, Math.min(1, challengeProb));
  }

  shouldBlock(blockerIdx, actorIdx, actionName, gameState) {
    // blocks:
    // - foreign-aid blocked by duke
    // - steal blocked by captain OR ambassador
    // - assassinate blocked by contessa
    const blocker = gameState.players[blockerIdx];
    if (!blocker || blocker.eliminated) return false;

    const inf = (blocker.cards || []).filter(c => !c.revealed).length;
    if (inf <= 0) return false;

    const has = (card) => this.belief.probHas(blockerIdx, card);

    let pBlock = 0;
    if (actionName === "foreign-aid") pBlock = has("duke");
    if (actionName === "steal") pBlock = Math.max(has("captain"), has("ambassador"));
    if (actionName === "assassinate") pBlock = has("contessa");

    // allow occasional bluff-block
    const bluff = this.epsilonBluff;
    const finalProb = Math.max(pBlock, bluff * 0.35);

    return this.rng() < Math.max(0, Math.min(1, finalProb));
  }

  // ---------- Scoring ----------

  scoreAction(playerIdx, actionObj, gameState, gameHistory) {
    const player = gameState.players[playerIdx];
    if (!player) return -1e9;

    // winning / forcing tactics
    if (this.isWinningMove(playerIdx, actionObj, gameState)) return 1e6;

    let v = 0;

    // base utility
    v += this.baseValue(actionObj, player, gameState);

    // challenge risk if action claims a role and we might not have it
    if (this.isChallengeable(actionObj.action)) {
      const req = this.requiredCardForAction(actionObj.action);
      const have = this.playerHasRole(player, req);

      // if we don't have it, simulate expected penalty weighted by estimated challenge rate
      if (!have) {
        const pChall = this.estimateChallengeRate(req, actionObj, playerIdx, gameState, gameHistory);
        const myInf = (player.cards || []).filter(c => !c.revealed).length;
        const penalty = (myInf === 1) ? 1e5 : 60;
        v = v * (1 - pChall) - penalty * pChall;
      }
    }

    // belief: target can block (steal / assassinate)
    if (actionObj.action === "steal" && actionObj.targetId !== undefined) {
      const t = this.coerceToIndex(gameState, actionObj.targetId);
      if (t !== -1) {
        const pBlock = Math.max(this.belief.probHas(t, "captain"), this.belief.probHas(t, "ambassador"));
        v += (1 - pBlock) * 5;
      }
    }
    if (actionObj.action === "assassinate" && actionObj.targetId !== undefined) {
      const t = this.coerceToIndex(gameState, actionObj.targetId);
      if (t !== -1) {
        const pContessa = this.belief.probHas(t, "contessa");
        v += (1 - pContessa) * 8;
      }
    }

    // mild endgame pressure: prefer coups/assassinations when rich
    v += this.strategicModifiers(actionObj, player, gameState);

    return v + this.rng() * 0.001;
  }

  // ---------- Helpers ----------

  getPossibleActions(player, gameState) {
    // Minimal mirror of your existing logic (Challenger already has a full version).
    // Keep simple for rollouts.
    if (!player || player.eliminated) return [];

    const actions = [{ action: "income" }];

    if (player.coins >= 7) {
      const targets = gameState.players.filter(p => p && !p.eliminated && p.id !== player.id);
      for (const t of targets) actions.push({ action: "coup", targetId: t.id });
      return actions;
    }

    actions.push({ action: "foreign-aid" });
    actions.push({ action: "tax" });

    const targets = gameState.players.filter(p => p && !p.eliminated && p.id !== player.id);
    for (const t of targets) {
      if (player.coins >= 3) actions.push({ action: "assassinate", targetId: t.id });
      actions.push({ action: "steal", targetId: t.id });
    }

    actions.push({ action: "exchange" });
    return actions;
  }

  isWinningMove(playerIdx, actionObj, gameState) {
    if (!["coup", "assassinate"].includes(actionObj.action)) return false;
    if (actionObj.targetId === undefined || actionObj.targetId === null) return false;

    const alive = gameState.players.filter(p => p && !p.eliminated);
    if (alive.length !== 2) return false;

    const t = this.coerceToIndex(gameState, actionObj.targetId);
    return t !== -1 && !gameState.players[t].eliminated;
  }

  baseValue(actionObj, player, gameState) {
    const values = { income: 1, "foreign-aid": 2, tax: 3, exchange: 1.8, coup: 35, assassinate: 42 };
    if (actionObj.action === "steal" && actionObj.targetId !== undefined) {
      const t = this.coerceToIndex(gameState, actionObj.targetId);
      const target = t !== -1 ? gameState.players[t] : null;
      return target ? Math.min(2, target.coins) * 2.2 : 0;
    }
    return values[actionObj.action] ?? 0;
  }

  strategicModifiers(actionObj, player, gameState) {
    const alive = gameState.players.filter(p => p && !p.eliminated);
    const avgCoins = alive.reduce((s, p) => s + p.coins, 0) / Math.max(1, alive.length);

    let m = 0;
    if (player.coins > avgCoins && ["coup", "assassinate", "steal"].includes(actionObj.action)) m += 4;
    if (player.coins < avgCoins && ["income", "foreign-aid", "tax"].includes(actionObj.action)) m += 2.5;
    if (player.coins >= 6 && actionObj.action === "tax") m += 2; // set up coup
    return m;
  }

  isChallengeable(actionName) {
    return ["tax", "assassinate", "steal", "exchange"].includes(actionName);
  }

  requiredCardForAction(actionName) {
    const map = { tax: "duke", assassinate: "assassin", steal: "captain", exchange: "ambassador" };
    return map[actionName] ?? null;
  }

  playerHasRole(player, role) {
    return (player.cards || []).some(c => !c.revealed && c.character === role);
  }

  estimateChallengeRate(required, actionObj, playerIdx, gameState, gameHistory) {
    const revealedCount = this.countRevealed(required, gameHistory);
    const base = [0.25, 0.35, 0.70, 1.00][Math.min(3, revealedCount)] ?? 0.30;

    // more likely challenged if target at 1 influence for assassinate
    if (actionObj.action === "assassinate" && actionObj.targetId !== undefined) {
      const t = this.coerceToIndex(gameState, actionObj.targetId);
      const target = t !== -1 ? gameState.players[t] : null;
      if (target) {
        const inf = (target.cards || []).filter(c => !c.revealed).length;
        const me = gameState.players[playerIdx];
        const have = this.playerHasRole(me, required);
        if (inf === 1 && !have) return 1.0;
      }
    }
    return base;
  }

  countRevealed(card, gameHistory) {
    const rc = gameHistory?.revealedCards;
    if (!Array.isArray(rc)) return 0;
    let n = 0;
    for (const arr of rc) {
      if (!Array.isArray(arr)) continue;
      for (const c of arr) if (c === card) n++;
    }
    return n;
  }

  coerceToIndex(gameState, maybeId) {
    if (typeof maybeId === "number") return maybeId;
    const ps = gameState.players || [];
    for (let i = 0; i < ps.length; i++) if (ps[i] && ps[i].id === maybeId) return i;
    return -1;
  }
}