// challenger-ai.js
// ChallengerAI (Determinization + MCTS/UCT) upgraded with particle-filter beliefs.
// Requires globals: AIEngine, ParticleBeliefTracker, CoupPolicy

class ChallengerAI extends AIEngine {
  constructor(playerId, aiType) {
    super(playerId, aiType);

    this.MCTS_ITERATIONS = 250;        // Increased from 180
    this.CHALLENGE_ITERATIONS = 100;   // Increased from 60
    this.BLOCK_ITERATIONS = 80;        // Increased from 45
    this.MAX_PLAYOUT_DEPTH = 14;
    this.UCT_C = 1.0; // Reduced from 1.25 - less exploration, more exploitation

    // Belief + rollout policy
    this.belief = new ParticleBeliefTracker(playerId, {
      numParticles: 450,
      truthLikelihood: 0.88,
      bluffLikelihood: 0.12,
      resampleEssFrac: 0.55
    });

    this.rolloutPolicy = new CoupPolicy(playerId, this.belief, {
      epsilonExplore: 0.12,
      epsilonBluff: 0.18
    });
  }

  // ===== Public API =====

  chooseAction(player, gameState, gameHistory) {
    this.belief.sync(gameState, gameHistory);

    if (player.coins >= 10) return this.chooseBestCoupTarget(player, gameState);

    const actions = this.getPossibleActions(player, gameState);
    if (actions.length === 0) return { action: "income" };

    // Check for obvious winning moves first (cheap check, huge value)
    for (const action of actions) {
      if (this.isWinningMove(action, gameState)) {
        return action; // Guaranteed win - take it immediately
      }
    }

    // Filter out suicide moves before MCTS wastes iterations on them
    const safeActions = actions.filter(a =>
      !this.isSuicideMove(a, player, gameState, gameHistory)
    );

    const rootActions = safeActions.length > 0 ? safeActions : actions;
    if (rootActions.length === 1) return rootActions[0];

    // CRITICAL: Check for guaranteed winning moves (eliminate last opponent)
    const alive = gameState.players.filter(p => !p.eliminated);
    if (alive.length === 2) {
      for (const action of rootActions) {
        if (this.isWinningMove(action, player, gameState)) {
          return action; // Take the win immediately
        }
      }
    }

    // Scale iterations for endgame - more computation when it matters most
    const iterationMultiplier = alive.length === 2 ? 1.5 : (alive.length === 3 ? 1.3 : 1.0);
    const iterations = Math.floor(this.MCTS_ITERATIONS * iterationMultiplier);

    const root = new MCTSNode(null, null, gameState.currentPlayerIndex);

    for (let i = 0; i < iterations; i++) {
      const det = this.determinize(gameState, gameHistory);

      let node = root;
      let simState = det;

      if (!node.actions) node.actions = rootActions.slice();

      let depth = 0;
      while (true) {
        if (depth > this.MAX_PLAYOUT_DEPTH) break;

        const curIndex = simState.currentPlayerIndex;
        const curPlayer = simState.players[curIndex];

        if (!curPlayer || curPlayer.eliminated) {
          simState.currentPlayerIndex = this.nextAliveAfter(simState, curIndex);
          depth++;
          if (this.isTerminal(simState)) break;
          continue;
        }

        if (curIndex === this.playerId) {
          if (!node.children) node.children = new Map();
          if (!node.actions) node.actions = this.getPossibleActions(curPlayer, simState);

          const untried = node.actions.filter(a => !node.children.has(this.actionKey(a)));
          if (untried.length > 0) {
            const a = untried[Math.floor(Math.random() * untried.length)];
            const key = this.actionKey(a);
            const child = new MCTSNode(node, a, this.nextAliveAfter(simState, curIndex));
            node.children.set(key, child);

            simState = this.applyFullTurn(simState, a);
            node = child;
            break;
          }

          let bestChild = null;
          let bestScore = -Infinity;
          for (const child of node.children.values()) {
            const uct = this.uctScore(child, node);
            if (uct > bestScore) {
              bestScore = uct;
              bestChild = child;
            }
          }
          if (!bestChild) break;

          simState = this.applyFullTurn(simState, bestChild.action);
          node = bestChild;
        } else {
          // Opponent turn: use simple heuristic policy for unbiased simulation
          const oppAction = this.opponentPolicyChooseAction(curPlayer, simState);
          simState = this.applyFullTurn(simState, oppAction);
        }

        if (this.isTerminal(simState)) break;
        depth++;
      }

      const reward = this.rollout(simState, gameHistory);
      this.backpropagate(node, reward);
    }

    // pick best mean
    let best = rootActions[0];
    let bestVal = -Infinity;
    if (root.children) {
      for (const a of rootActions) {
        const child = root.children.get(this.actionKey(a));
        if (!child || child.visits === 0) continue;
        const mean = child.totalValue / child.visits;
        if (mean > bestVal) {
          bestVal = mean;
          best = a;
        }
      }
    }
    return best;
  }

  decideChallengeAction(action, claimantId, gameState, gameHistory) {
    this.belief.sync(gameState, gameHistory);

    const required = this.getRequiredCard(action);
    if (!required) return false;

    const player = gameState.players[this.playerId];
    if (!player) return false; // Safety check
    const myInfluences = player.cards.filter(c => !c.revealed).length;

    // Count revealed cards
    let revealedCount = 0;
    if (gameHistory?.revealedCards) {
      for (const arr of gameHistory.revealedCards) {
        revealedCount += (arr || []).filter(c => c === required).length;
      }
    }

    if (revealedCount === 3) return true; // Impossible claim
    if (revealedCount === 2) return Math.random() < 0.55; // Very likely a bluff

    // CRITICAL: Being assassinated on last influence
    if (action === 'assassinate' && myInfluences === 1) {
      const hasContessa = player.cards.some(c =>
        !c.revealed && c.character === 'contessa'
      );
      if (hasContessa) {
        // Don't challenge - we'll block instead (more reliable defense)
        return false;
      } else {
        // No Contessa, must challenge (only chance to survive)
        return true;
      }
    }

    // Use belief-based challenge decisions (MCTS doesn't work well for challenges
    // because it simulates random worlds, but challenge success depends on fixed hidden cards)
    const pHas = this.belief.probHas(claimantId, required);

    // Danger-aware threshold matching Statistical AI
    const targetIsMe = gameState.pendingAction?.targetId === this.playerId;
    let danger = 0;
    if (action === 'assassinate' && targetIsMe) danger += 0.10;
    if (action === 'steal' && targetIsMe) danger += 0.06;

    // Be more conservative when we only have 1 influence left
    if (myInfluences === 1) danger -= 0.10;

    let threshold = 0.30 - danger;
    threshold = Math.max(0.05, Math.min(0.90, threshold));

    return pHas < threshold;
  }

  decideBlockAction(action, actorId, gameState, gameHistory) {
    this.belief.sync(gameState, gameHistory);

    const me = gameState.players[this.playerId];
    if (!me) return false;

    const blockCards = this.getPossibleBlockCards(action);
    if (blockCards.length === 0) return false;

    const myInfluences = me.cards.filter(c => !c.revealed).length;

    // SPECIAL CASE: Being assassinated on last influence
    if (action === 'assassinate' && myInfluences === 1) {
      const hasContessa = me.cards.some(c => !c.revealed && c.character === 'contessa');
      if (hasContessa) {
        return true; // Always block with real Contessa
      } else {
        // No Contessa - bluff block ~50% of the time as alternative to challenging
        return Math.random() < 0.50;
      }
    }

    // Check if we actually have a blocking card
    const hasRealBlock = blockCards.some(c =>
      me.cards.some(x => !x.revealed && x.character === c)
    );

    // If we don't have a blocking card, rarely bluff
    if (!hasRealBlock) {
      if (action === 'assassinate' && Math.random() < 0.06) return true;
      if (action === 'steal' && Math.random() < 0.08) return true;
      if (action === 'foreign-aid' && Math.random() < 0.05) return true;
      return false;
    }

    // We have a real block - use it with high probability (matching Statistical AI)
    // Don't use MCTS for blocks because whether we get challenged depends on
    // opponent's actual beliefs about us, not random simulations
    if (action === 'assassinate') return Math.random() < 0.95;
    if (action === 'steal') return Math.random() < 0.88;
    if (action === 'foreign-aid') return Math.random() < 0.88;
    return Math.random() < 0.88;
  }

  decideBlockClaim(action, actorId, gameState, gameHistory) {
    if (action === 'foreign-aid') {
      return 'duke';
    }
    if (action === 'assassinate') {
      return 'contessa';
    }
    if (action === 'steal') {
      // Use our actual hand for self-claims; belief particles do not track self.
      const me = gameState.players[this.playerId];
      const hasCaptain = me?.cards.some(c => !c.revealed && c.character === 'captain');
      const hasAmbassador = me?.cards.some(c => !c.revealed && c.character === 'ambassador');

      if (hasCaptain && !hasAmbassador) return 'captain';
      if (hasAmbassador && !hasCaptain) return 'ambassador';
      return Math.random() < 0.5 ? 'captain' : 'ambassador';
    }
    return null; // Should not happen for blockable actions
  }

  // ===== Terminal state detection =====

  isWinningMove(actionObj, gameState) {
    if (!['coup', 'assassinate'].includes(actionObj.action)) return false;
    if (actionObj.targetId === undefined || actionObj.targetId === null) return false;

    const alive = gameState.players.filter(p => !p.eliminated);
    if (alive.length !== 2) return false; // Not down to 2 players

    const targetIdx = this._coerceToIndex(gameState, actionObj.targetId);
    const target = targetIdx !== -1 ? gameState.players[targetIdx] : null;
    return target && !target.eliminated; // Eliminating last opponent = win
  }

  isSuicideMove(actionObj, player, gameState, gameHistory) {
    // Only challengeable actions can lead to suicide via failed challenge
    if (!['tax', 'assassinate', 'steal', 'exchange'].includes(actionObj.action)) {
      return false;
    }

    const required = this.getRequiredCard(actionObj.action);
    if (!required) return false;

    // Check if we actually have the required card
    const hasCard = player.cards.some(c => !c.revealed && c.character === required);

    // If we have the card, we can't lose a challenge
    if (hasCard) return false;

    // We're bluffing - check if we only have 1 influence left
    const myInfluences = player.cards.filter(c => !c.revealed).length;
    if (myInfluences !== 1) return false; // Not suicide if we have 2+ influences

    // Check if all copies of the required card are publicly revealed
    const revealed = gameHistory?.revealedCards;
    if (!Array.isArray(revealed)) return false;

    let revealedCount = 0;
    for (const arr of revealed) {
      if (!Array.isArray(arr)) continue;
      for (const c of arr) {
        if (c === required) revealedCount++;
      }
    }

    // Definitely suicide if all 3 copies revealed
    if (revealedCount === 3) return true;

    // Highly likely suicide if 2 revealed (only 1 card left in game)
    if (revealedCount === 2) {
      const probOpponentHasIt = this.estimateProbOpponentHasCard(
        required,
        gameState,
        gameHistory
      );

      // If >75% chance opponent has it AND we're on last influence, treat as suicide
      if (probOpponentHasIt > 0.75) return true;
    }

    return false;
  }

  estimateProbOpponentHasCard(card, gameState, gameHistory) {
    this.belief.sync(gameState, gameHistory);

    const opponents = gameState.players.filter(p => p.id !== this.playerId && !p.eliminated);

    let maxProb = 0;
    for (const opp of opponents) {
      const oppIdx = this._coerceToIndex(gameState, opp.id);
      if (oppIdx === -1) continue;

      const prob = this.belief.probHas(oppIdx, card);
      maxProb = Math.max(maxProb, prob);
    }

    return maxProb;
  }

  // ===== MCTS for critical decisions =====

  mctsDecideChallenge(action, claimantId, gameState, gameHistory) {
    let challengeValue = 0;
    let noChallengeValue = 0;
    let validIterations = 0;

    // Scale iterations based on game state criticality
    const alive = gameState.players.filter(p => !p.eliminated).length;
    const iterationMultiplier = alive === 2 ? 1.5 : (alive === 3 ? 1.2 : 1.0);
    const iterations = Math.floor(this.CHALLENGE_ITERATIONS * iterationMultiplier);

    for (let i = 0; i < iterations; i++) {
      const det = this.determinize(gameState, gameHistory);

      // Branch 1: We challenge
      const challengeSim = this.cloneState(det);
      const required = this.getRequiredCard(action);
      const claimant = challengeSim.players[claimantId];
      if (!claimant) continue; // Safety check
      const hasCard = claimant.cards.some(c =>
        !c.revealed && c.character === required
      );

      if (hasCard) {
        // We lose influence
        this.simLoseInfluence(challengeSim.players[this.playerId], challengeSim);
        this.simResolveAction(claimantId, { action, targetId: gameState.pendingAction?.targetId }, challengeSim);
      } else {
        // They lose influence
        this.simLoseInfluence(claimant, challengeSim);
        if (action === 'assassinate') claimant.coins += 3;
      }
      challengeSim.currentPlayerIndex = this.nextAliveAfter(challengeSim, claimantId);
      challengeValue += this.rollout(challengeSim);

      // Branch 2: We don't challenge
      const noChallengeSim = this.cloneState(det);
      this.simResolveAction(claimantId, { action, targetId: gameState.pendingAction?.targetId }, noChallengeSim);
      noChallengeSim.currentPlayerIndex = this.nextAliveAfter(noChallengeSim, claimantId);
      noChallengeValue += this.rollout(noChallengeSim);

      validIterations++;
    }

    if (validIterations === 0) return false; // No valid simulations
    return (challengeValue / validIterations) > (noChallengeValue / validIterations);
  }

  mctsDecideBlock(action, actorId, gameState, gameHistory) {
    let blockValue = 0;
    let noBlockValue = 0;
    let validIterations = 0;

    // Scale iterations based on game state criticality
    const alive = gameState.players.filter(p => !p.eliminated).length;
    const iterationMultiplier = alive === 2 ? 1.5 : (alive === 3 ? 1.2 : 1.0);
    const iterations = Math.floor(this.BLOCK_ITERATIONS * iterationMultiplier);

    for (let i = 0; i < iterations; i++) {
      const det = this.determinize(gameState, gameHistory);

      // Safety check
      if (!det.players[actorId]) continue;

      // Branch 1: We block
      const blockSim = this.cloneState(det);
      // Action is blocked - nothing happens, actor's turn ends
      blockSim.currentPlayerIndex = this.nextAliveAfter(blockSim, actorId);
      blockValue += this.rollout(blockSim);

      // Branch 2: We don't block (action proceeds)
      const noBlockSim = this.cloneState(det);
      this.simResolveAction(actorId, { action, targetId: gameState.pendingAction?.targetId }, noBlockSim);
      noBlockSim.currentPlayerIndex = this.nextAliveAfter(noBlockSim, actorId);
      noBlockValue += this.rollout(noBlockSim);

      validIterations++;
    }

    if (validIterations === 0) return false; // No valid simulations
    return (blockValue / validIterations) > (noBlockValue / validIterations);
  }

  // ===== Key upgrade: belief-based determinization =====

  determinize(gameState, gameHistory) {
    // Prefer belief-based determinization to keep tree evaluation consistent.
    const base = this.cloneState(gameState);

    this.belief.sync(gameState, gameHistory);
    const world = this.belief.sampleParticleWorld();

    if (world && Array.isArray(world.hands)) {
      // Assign hidden cards from the sampled particle.
      for (let i = 0; i < base.players.length; i++) {
        const p = base.players[i];
        if (!p || p.id === this.playerId) continue; // Keep our real cards

        const hand = world.hands[i] || [];
        let hIdx = 0;
        for (const card of p.cards) {
          if (!card.revealed && hIdx < hand.length) {
            card.character = hand[hIdx++];
          }
        }
      }

      base.deck = Array.isArray(world.deck) ? world.deck.slice() : base.deck;
      return base;
    }

    // Fallback: uniform random determinization
    const probCalc = new ProbabilityCalculator(gameHistory, base, this.playerId);
    const unseen = probCalc.getUnseenCards();
    this.shuffleInPlace(unseen);
    for (const p of base.players) {
      if (p.id === this.playerId) continue;
      for (const card of p.cards) {
        if (!card.revealed && unseen.length > 0) {
          card.character = unseen.pop();
        }
      }
    }
    base.deck = unseen;
    return base;
  }

  // ===== Rollout logic from MinimaxAI (simple, robust heuristics) =====

  rollout(state) {
    // Random-ish rollout with simple policies; returns reward from OUR perspective
    let sim = this.cloneState(state);
    let stuckCounter = 0; // Prevent infinite loops

    for (let d = 0; d < this.MAX_PLAYOUT_DEPTH; d++) {
        if (this.isTerminal(sim)) break;

        const actorIndex = sim.currentPlayerIndex;
        const actor = sim.players[actorIndex];

        if (!actor || actor.eliminated) {
            sim.currentPlayerIndex = this.nextAliveAfter(sim, actorIndex);
            stuckCounter++;
            if (stuckCounter > sim.players.length) break; // Safety: prevent infinite loops
            d--; // Don't count this as a depth iteration
            continue;
        }
        
        stuckCounter = 0; // Reset counter when we find a valid player

        const action = (actorIndex === this.playerId)
            ? this.selfRolloutPolicyChooseAction(actor, sim)
            : this.opponentPolicyChooseAction(actor, sim);

        sim = this.applyFullTurn(sim, action);
    }

    return this.evaluate(sim);
  }

  opponentPolicyChooseAction(player, s) {
    if (player.coins >= 10) return this.chooseBestCoupTarget(player, s);

    const actions = this.getPossibleActions(player, s);
    if (actions.length === 0) return { action: 'income' };
    
    const scored = actions.map(a => ({
        a,
        ev: this.fastExpectedValue(a, player, s)
    }));
    
    scored.sort((x, y) => y.ev - x.ev);
    
    const roll = Math.random();
    if (roll < 0.80 && scored.length > 0) return scored[0].a;
    if (roll < 0.95 && scored.length > 1) return scored[1].a;
    return scored[Math.floor(Math.random() * scored.length)].a;
  }

  selfRolloutPolicyChooseAction(player, s) {
    if (player.coins >= 10) return this.chooseBestCoupTarget(player, s);

    const actions = this.getPossibleActions(player, s);
    if (actions.length === 0) return { action: 'income' };
    
    const scored = actions.map(a => ({
        a,
        ev: this.fastExpectedValue(a, player, s)
    }));
    
    scored.sort((x, y) => y.ev - x.ev);
    
    const roll = Math.random();
    if (roll < 0.90 && scored.length > 0) return scored[0].a;
    if (roll < 0.98 && scored.length > 1) return scored[1].a;
    return scored[Math.floor(Math.random() * scored.length)].a;
  }

  fastExpectedValue(action, player, s) {
    const a = action.action;
    let value = 0;
    
    if (a === 'coup') {
        if (player.coins >= 7) {
            value = 100 + this.targetThreatScore(action.targetId, s);
        } else { return -1000; }
    } else if (a === 'assassinate') {
        if (player.coins >= 3) {
            value = 70 + this.targetThreatScore(action.targetId, s);
            value -= 15;
        } else { return -1000; }
    } else if (a === 'steal') {
        const target = s.players[action.targetId];
        if (target && !target.eliminated) {
            const stolen = Math.min(2, target.coins);
            value = stolen * 15 + target.coins * 2;
            value *= 0.6;
        } else { return -1000; }
    } else if (a === 'tax') {
        value = 25;
        value -= 3;
    } else if (a === 'foreign-aid') {
        value = 18;
        value *= 0.75;
    } else if (a === 'income') {
        value = 10;
    } else if (a === 'exchange') {
        value = 20;
        value -= 5;
    }
    
    const activePlayers = s.players.filter(p => !p.eliminated);
    if (activePlayers.length === 0) return value;
    const avgCoins = activePlayers.reduce((sum, p) => sum + p.coins, 0) / activePlayers.length;
    
    if (player.coins < avgCoins - 2) {
        if (['income', 'foreign-aid', 'tax'].includes(a)) value += 8;
    }
    if (player.coins > avgCoins + 2) {
        if (['coup', 'assassinate', 'steal'].includes(a)) value += 12;
    }
    if (activePlayers.length <= 3) {
        if (a === 'coup' || a === 'assassinate') value += 20;
    }
    
    if (a === 'coup' && player.coins < 7) return -1000;
    if (a === 'assassinate' && player.coins < 3) return -1000;
    
    return value;
  }

  targetThreatScore(tid, s) {
    const t = s.players[tid];
    if (!t || t.eliminated) return 0;
    const inf = t.cards.filter(c => !c.revealed).length;
    return t.coins * 5 + inf * 45; // Increased weights (4→5, 40→45)
  }

  // ===== Simulation challenge/block hooks using simple heuristics =====

  simPolicyChallenge(challengerId, actionName, claimantId, simState) {
    const required = this.getRequiredCard(actionName);
    const claimant = simState.players[claimantId];
    const challenger = simState.players[challengerId];

    if (!claimant || !challenger) return false;

    // Count revealed cards of this type (public information)
    let revealedOfType = 0;
    for (const p of simState.players) {
      for (const card of p.cards) {
        if (card.revealed && card.character === required) {
          revealedOfType++;
        }
      }
    }

    // If all 3 copies revealed, definitely challenge (impossible claim)
    if (revealedOfType === 3) return true;

    // Estimate probability claimant has the card based on public info
    const totalOfType = 3;
    const unseenOfType = totalOfType - revealedOfType;
    const claimantActiveCards = claimant.cards.filter(c => !c.revealed).length;

    if (claimantActiveCards === 0 || unseenOfType === 0) return true;

    // Rough probability estimate (intentionally imperfect)
    const baseProbHas = unseenOfType / (unseenOfType + 10);

    // Adjust for danger - challenge more aggressively if action threatens us
    let dangerBonus = 0;
    if (actionName === 'assassinate' || actionName === 'steal') {
      dangerBonus = 0.12; // Conservative
    }
    if (challenger.cards.filter(c => !c.revealed).length === 1) {
      // More cautious when we only have one card left
      dangerBonus -= 0.15; // Very cautious
    }

    // Challenge threshold with noise
    const threshold = 0.30 - dangerBonus; // Conservative base
    const noise = Math.random() * 0.08 - 0.04; // Small noise

    return (baseProbHas + noise) < threshold;
  }

  simPolicyBlock(blockerId, actionName, actorId, simState) {
    const blocker = simState.players[blockerId];
    if (!blocker) return false;

    const cards = this.getPossibleBlockCards(actionName);
    if (cards.length === 0) return false;

    // Check if blocker actually has a blocking card
    const has = cards.some(req => blocker.cards.some(c => !c.revealed && c.character === req));

    if (has) {
      // Has real block - very likely to use it
      if (actionName === 'assassinate') return Math.random() < 0.95; // Increased from 0.93
      if (actionName === 'steal') return Math.random() < 0.90; // Increased from 0.88
      if (actionName === 'foreign-aid') return Math.random() < 0.85; // Increased from 0.82
      return Math.random() < 0.88; // Increased from 0.85
    }

    // Bluffing blocks (much rarer)
    let revealedBlockers = 0;
    for (const p of simState.players) {
      for (const card of p.cards) {
        if (card.revealed && cards.includes(card.character)) {
          revealedBlockers++;
        }
      }
    }

    // More likely to bluff if fewer blocking cards are revealed
    const bluffFactor = Math.max(0, 1 - revealedBlockers / 4);

    if (actionName === 'assassinate') return Math.random() < 0.08 * bluffFactor;
    if (actionName === 'steal') return Math.random() < 0.05 * bluffFactor;
    if (actionName === 'foreign-aid') return Math.random() < 0.03 * bluffFactor;

    return false;
  }

  // ===== Everything below is mostly your existing minimax-ai.js utilities =====
  // I kept these intact; if you want, I can also add “challenge-after-block” micro-branching next.

  // ---------- MCTS utilities ----------
  uctScore(child, parent) {
    if (child.visits === 0) return Infinity;
    const exploit = child.totalValue / child.visits;
    const explore = this.UCT_C * Math.sqrt(Math.log(parent.visits + 1) / child.visits);
    return exploit + explore;
  }

  backpropagate(node, reward) {
    while (node) {
      node.visits += 1;
      node.totalValue += reward;
      node = node.parent;
    }
  }

  actionKey(a) {
    if (!a) return "null";
    return `${a.action}:${a.targetId ?? ""}`;
  }

  // ---------- Game helpers (same style as your existing ChallengerAI) ----------
  cloneState(state) {
    const new_state = {
      currentPlayerIndex: state.currentPlayerIndex,
      deck: Array.isArray(state.deck) ? state.deck.slice() : [],
      players: new Array(state.players.length),
      // pendingAction is a simple object, so a shallow copy is sufficient and fast
      pendingAction: state.pendingAction ? { ...state.pendingAction } : null,
      gameOver: state.gameOver,
      turnCount: state.turnCount,
      // The history object is not needed for simulation, so we don't copy it.
    };

    for (let i = 0; i < state.players.length; i++) {
      const p = state.players[i];
      const new_player = {
        id: p.id,
        coins: p.coins,
        eliminated: p.eliminated,
        cards: new Array(p.cards.length),
        // We don't copy name, isHuman, or aiEngine, as they are not needed for simulation logic.
      };
      for (let j = 0; j < p.cards.length; j++) {
        const c = p.cards[j];
        new_player.cards[j] = {
          character: c.character,
          revealed: c.revealed,
        };
      }
      new_state.players[i] = new_player;
    }
    return new_state;
  }

  isTerminal(state) {
    const alive = state.players.filter(p => p && !p.eliminated);
    return alive.length <= 1;
  }

  nextAliveAfter(state, idx) {
    const n = state.players.length;
    for (let k = 1; k <= n; k++) {
      const j = (idx + k) % n;
      if (state.players[j] && !state.players[j].eliminated) return j;
    }
    return idx;
  }

  evaluate(state) {
    const alive = state.players.filter(p => p && !p.eliminated);
    if (alive.length === 1) return alive[0].id === this.playerId ? 100000 : -100000;

    const me = state.players[this.playerId];
    if (!me) return -100000;

    const myInf = me.cards.filter(c => !c.revealed).length;
    if (myInf === 0) return -100000; // We're eliminated

    // Simpler evaluation without belief contamination
    // Use the actual cards in the determinized state, not beliefs
    let score = myInf * 400 + me.coins * 15;

    for (const p of state.players) {
      if (!p || p.id === this.playerId || p.eliminated) continue;
      const inf = p.cards.filter(c => !c.revealed).length;

      // Simple threat assessment based on coins and influences
      score -= inf * 220;
      score -= p.coins * 10;

      // In 2-player endgame, heavily penalize opponent strength
      if (alive.length === 2) {
        score -= inf * 200; // Extra penalty for opponent influences
        if (p.coins >= 7) score -= 150; // Opponent can coup us
      }
    }

    return score;
  }

  chooseBestCoupTarget(player, gameState) {
    const targets = gameState.players.filter(p => p && !p.eliminated && p.id !== player.id);
    if (targets.length === 0) return { action: "income" };
    targets.sort((a, b) => {
      const aInf = a.cards.filter(c => !c.revealed).length;
      const bInf = b.cards.filter(c => !c.revealed).length;
      if (aInf !== bInf) return bInf - aInf; // Fixed: Target players with MORE influences
      return b.coins - a.coins;
    });
    return { action: "coup", targetId: targets[0].id };
  }

  getPossibleActions(player, gameState) {
    // Keep your existing logic if you prefer; this is a reasonable superset.
    if (!player || player.eliminated) return [];
    const actions = [{ action: "income" }];

    if (player.coins < 10) actions.push({ action: "foreign-aid" });
    actions.push({ action: "tax" });
    actions.push({ action: "exchange" });

    const targets = gameState.players.filter(p => p && !p.eliminated && p.id !== player.id);
    for (const t of targets) {
      actions.push({ action: "steal", targetId: t.id });
      if (player.coins >= 3) actions.push({ action: "assassinate", targetId: t.id });
      if (player.coins >= 7) actions.push({ action: "coup", targetId: t.id });
    }

    // Forced coup rule
    if (player.coins >= 10) {
      return [this.chooseBestCoupTarget(player, gameState)];
    }
    return actions;
  }

  // ===== Full turn simulation logic (from minimax-ai.js) =====

  applyFullTurn(state, actionObj) {
    let s = this.cloneState(state);
    const actorIndex = s.currentPlayerIndex;
    const actor = s.players[actorIndex];

    if (!actor || actor.eliminated) {
        s.currentPlayerIndex = this.nextAliveAfter(s, actorIndex);
        return s;
    }

    if (actor.coins >= 10 && actionObj.action !== 'coup') {
        actionObj = this.chooseBestCoupTarget(actor, s);
    }

    // Challenge phase
    const challengeable = ['tax', 'assassinate', 'steal', 'exchange'].includes(actionObj.action);
    if (challengeable) {
        const challengers = s.players.filter(p => p.id !== actorIndex && !p.eliminated);
        this.shuffleInPlace(challengers);

        for (const ch of challengers) {
            // This now correctly calls the belief-aware simPolicyChallenge from ChallengerAI
            const shouldChallenge = this.simPolicyChallenge(ch.id, actionObj.action, actorIndex, s);
            if (shouldChallenge) {
                const challengeStopsAction = this.simResolveChallenge(ch.id, actorIndex, actionObj.action, s);
                if (challengeStopsAction) {
                    s.currentPlayerIndex = this.nextAliveAfter(s, actorIndex);
                    return s;
                }
                break; 
            }
        }
    }

    // Block phase
    if (actionObj.action === 'foreign-aid') {
        const blockers = s.players.filter(p => p.id !== actorIndex && !p.eliminated);
        this.shuffleInPlace(blockers);
        for (const b of blockers) {
            // This now correctly calls the belief-aware simPolicyBlock from ChallengerAI
            if (this.simPolicyBlock(b.id, 'foreign-aid', actorIndex, s)) {
                s.currentPlayerIndex = this.nextAliveAfter(s, actorIndex);
                return s;
            }
        }
    } else if ((actionObj.action === 'assassinate' || actionObj.action === 'steal') && actionObj.targetId != null) {
        const target = s.players[actionObj.targetId];
        if (target && !target.eliminated) {
            // This now correctly calls the belief-aware simPolicyBlock from ChallengerAI
            if (this.simPolicyBlock(target.id, actionObj.action, actorIndex, s)) {
                s.currentPlayerIndex = this.nextAliveAfter(s, actorIndex);
                return s;
            }
        }
    }

    // Resolve action
    this.simResolveAction(actorIndex, actionObj, s);

    // Advance turn
    s.currentPlayerIndex = this.nextAliveAfter(s, actorIndex);
    return s;
  }

  simResolveAction(actorId, actionObj, s) {
      const actor = s.players[actorId];
      if (!actor) return;
      
      const tId = actionObj.targetId;

      switch (actionObj.action) {
          case 'income':
              actor.coins += 1;
              break;
          case 'foreign-aid':
              actor.coins += 2;
              break;
          case 'tax':
              actor.coins += 3;
              break;
          case 'coup':
              if (actor.coins >= 7) {
                  actor.coins -= 7;
                  const target = s.players[tId];
                  if (target) this.simLoseInfluence(target, s);
              }
              break;
          case 'assassinate':
              if (actor.coins >= 3) {
                  actor.coins -= 3;
                  const target = s.players[tId];
                  if (target) this.simLoseInfluence(target, s);
              }
              break;
          case 'steal':
              const target = s.players[tId];
              if (target) {
                  const stolen = Math.min(2, target.coins);
                  target.coins -= stolen;
                  actor.coins += stolen;
              }
              break;
          case 'exchange':
              const aliveCards = actor.cards.filter(c => !c.revealed);
              const n = aliveCards.length;
              const drawn = [];
              for (let i = 0; i < n && s.deck.length > 0; i++) {
                  drawn.push(s.deck.pop());
              }
              for (const c of aliveCards) {
                  s.deck.push(c.character);
              }
              this.shuffleInPlace(s.deck);
              let idx = 0;
              for (const c of actor.cards) {
                  if (!c.revealed) {
                      c.character = drawn.length > idx ? drawn[idx++] : c.character;
                  }
              }
              break;
      }
      this.simCheckElims(s);
  }

  simResolveChallenge(challengerId, claimantId, action, s) {
      const claimant = s.players[claimantId];
      const challenger = s.players[challengerId];
      if (!claimant || !challenger) return true;
      
      const required = this.getRequiredCard(action);
      const hasCardIndex = claimant.cards.findIndex(c => !c.revealed && c.character === required);

      if (hasCardIndex !== -1) {
          // Challenge failed: challenger loses influence
          this.simLoseInfluence(challenger, s);
          // Claimant shuffles card and gets a new one
          if (s.deck.length > 0) {
              s.deck.push(claimant.cards[hasCardIndex].character);
              this.shuffleInPlace(s.deck);
              claimant.cards[hasCardIndex].character = s.deck.pop();
          }
          return false; // Action proceeds
      } else {
          // Challenge succeeded: claimant loses influence
          if (action === 'assassinate') {
              claimant.coins += 3; // Refund cost
          }
          this.simLoseInfluence(claimant, s);
          return true; // Action is blocked
      }
  }

  simLoseInfluence(player, s) {
      if (!player) return;
      const card = player.cards.find(c => !c.revealed);
      if (card) {
          card.revealed = true;
      }
      this.simCheckElims(s);
  }

  simCheckElims(s) {
      for (const p of s.players) {
          if (!p.eliminated && p.cards.every(c => c.revealed)) {
              p.eliminated = true;
          }
      }
  }
  
  shuffleInPlace(arr) {
      for (let i = arr.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [arr[i], arr[j]] = [arr[j], arr[i]];
      }
  }

  getRequiredCard(action) {
      const map = { tax: 'duke', assassinate: 'assassin', steal: 'captain', exchange: 'ambassador' };
      return map[action];
  }

  getPossibleBlockCards(action) {
      if (action === 'foreign-aid') return ['duke'];
      if (action === 'assassinate') return ['contessa'];
      if (action === 'steal') return ['captain', 'ambassador'];
      return [];
  }

  // ===== Helper for id/index conversion =====

  _coerceToIndex(gameState, idOrIndex) {
    const N = gameState.players.length;

    if (typeof idOrIndex === 'number' && Number.isInteger(idOrIndex)) {
      if (idOrIndex >= 0 && idOrIndex < N) return idOrIndex;
    }

    for (let i = 0; i < N; i++) {
      if (gameState.players[i] && gameState.players[i].id === idOrIndex) return i;
    }

    if (typeof idOrIndex === 'string') {
      const x = Number(idOrIndex);
      if (Number.isInteger(x) && x >= 0 && x < N) return x;
    }

    return -1;
  }
}

class MCTSNode {
  constructor(parent, action, nextPlayerIndex) {
    this.parent = parent;
    this.action = action;
    this.nextPlayerIndex = nextPlayerIndex;

    this.children = null;
    this.actions = null;

    this.visits = 0;
    this.totalValue = 0;
  }
}
