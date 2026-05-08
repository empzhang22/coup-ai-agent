// ChallengerAI - Tree search AI (Determinization + MCTS/UCT) for Coup (your simplified rules)
//
// IMPORTANT CONVENTION: Throughout this game engine, player.id === array index.
// This means that:
// - this.playerId is an INDEX into the players[] array
// - Parameters named *Id (challengerId, claimantId, etc.) are INDICES, not separate IDs
// - We can use players[playerId] directly without searching
//
// This convention is enforced by the game initialization code which sets:
//   gameState.players.push({ id: i, ... })  where i is the loop index

class MinimaxAI extends AIEngine {
    constructor(playerId, aiType) {
        super(playerId, aiType);

        // Tuning knobs (keep small for speed in large simulations)
        this.MCTS_ITERATIONS = 180;   // per move (action selection)
        this.CHALLENGE_ITERATIONS = 60; // per challenge decision (NEW)
        this.BLOCK_ITERATIONS = 40;     // per block decision (NEW)
        this.MAX_PLAYOUT_DEPTH = 14;  // number of turns simulated in rollouts
        this.UCT_C = 1.25;            // exploration constant
    }

    // ===== Public API used by the game engine =====

    chooseAction(player, gameState, gameHistory) {
        // Forced coup rule
        if (player.coins >= 10) return this.chooseBestCoupTarget(player, gameState);

        const rootActions = this.getPossibleActions(player, gameState);
        if (rootActions.length === 0) return { action: 'income' }; // Safety fallback
        if (rootActions.length === 1) return rootActions[0];

        // Run MCTS from current position
        const root = new MCTSNode(null, null, gameState.currentPlayerIndex);

        for (let i = 0; i < this.MCTS_ITERATIONS; i++) {
            // Determinize hidden info for this simulation
            const det = this.determinize(gameState, gameHistory);

            // Selection + Expansion
            let node = root;
            let simState = det;

            // Expand root actions lazily
            if (!node.actions) node.actions = rootActions.slice();

            // Walk tree
            let depth = 0;
            while (true) {
                if (depth > this.MAX_PLAYOUT_DEPTH) break;

                // currentPlayerIndex is an INDEX into players[]
                const curIndex = simState.currentPlayerIndex;
                const curPlayer = simState.players[curIndex];

                if (!curPlayer || curPlayer.eliminated) {
                    simState.currentPlayerIndex = this.nextAliveAfter(simState, curIndex);
                    depth++; // Increment to prevent infinite loops
                    if (this.isTerminal(simState)) break;
                    continue;
                }

                if (curIndex === this.playerId) {
                    // Our turn: select/expand using UCT
                    if (!node.children) node.children = new Map();

                    // Initialize actions for this node if not done
                    if (!node.actions) {
                        node.actions = this.getPossibleActions(curPlayer, simState);
                    }

                    // If untried actions exist, expand one
                    const untried = node.actions.filter(a => !node.children.has(this.actionKey(a)));
                    if (untried.length > 0) {
                        const a = untried[Math.floor(Math.random() * untried.length)];
                        const key = this.actionKey(a);
                        const child = new MCTSNode(node, a, this.nextAliveAfter(simState, curIndex));
                        node.children.set(key, child);

                        // Apply action in sim
                        simState = this.applyFullTurn(simState, a);
                        node = child;
                        break;
                    }

                    // Otherwise select best UCT child
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
                    // Opponent turn: one policy step (no branching in the tree to keep it fast)
                    const oppAction = this.opponentPolicyChooseAction(curPlayer, simState);
                    simState = this.applyFullTurn(simState, oppAction);
                }

                // Stop if game ended
                if (this.isTerminal(simState)) break;

                depth++;
            }

            // Rollout (from simState)
            const reward = this.rollout(simState);

            // Backprop
            this.backpropagate(node, reward);
        }

        // Pick the action with best mean value
        let best = null;
        let bestMean = -Infinity;

        if (root.children) {
            for (const child of root.children.values()) {
                const mean = child.totalValue / Math.max(1, child.visits);
                if (mean > bestMean) {
                    bestMean = mean;
                    best = child.action;
                }
            }
        }

        // Fallback if something went weird
        if (!best) {
            return this.safeHeuristicFallback(player, gameState);
        }

        return best;
    }

    decideChallengeAction(action, claimantId, gameState, gameHistory) {
        // Only challenge actions that require a character (as your engine does)
        if (!['tax', 'assassinate', 'steal', 'exchange'].includes(action)) return false;

        const required = this.getRequiredCard(action);

        // If lots of that card is already revealed, challenges are strong
        let revealedOfType = 0;
        gameHistory.revealedCards.forEach(cards => {
            revealedOfType += cards.filter(c => c === required).length;
        });
        if (revealedOfType === 3) return true;      // impossible - definitely challenge

        // For critical situations, use MCTS to decide
        // Critical = 2+ cards revealed, or dangerous action, or endgame
        const isDangerous = (action === 'assassinate' || action === 'steal');
        const isEndgame = gameState.players.filter(p => !p.eliminated).length <= 3;
        const isCritical = revealedOfType >= 2 || isDangerous || isEndgame;

        if (isCritical) {
            return this.mctsDecideChallenge(action, claimantId, gameState, gameHistory);
        }

        // For non-critical, use fast probabilistic heuristic
        return this.fastChallengeHeuristic(action, claimantId, gameState, gameHistory);
    }

    fastChallengeHeuristic(action, claimantId, gameState, gameHistory) {
        // Fast probability-based decision for non-critical situations
        const required = this.getRequiredCard(action);
        
        const probCalc = new ProbabilityCalculator(gameHistory, gameState, this.playerId);
        const probs = probCalc.calculateCardProbabilities(claimantId);
        const pHas = probs[required] ?? 0.25;

        // More willing to challenge if the action is dangerous to us
        const danger = this.challengeDangerBonus(action, claimantId, gameState);

        // Threshold with a bit of noise to avoid being deterministic
        const threshold = 0.28 - danger; // lower threshold => more challenges
        return (pHas + (Math.random() * 0.06 - 0.03)) < threshold;
    }

    mctsDecideChallenge(action, claimantId, gameState, gameHistory) {
        // Use MCTS to evaluate: challenge vs don't challenge
        // Run determinized simulations for both branches
        
        let challengeValue = 0;
        let noChallengeValue = 0;
        let challengeCount = 0;
        let noChallengeCount = 0;
        const iterations = this.CHALLENGE_ITERATIONS;

        for (let i = 0; i < iterations; i++) {
            // Determinize once, use for both branches
            const det = this.determinize(gameState, gameHistory);
            
            // Branch 1: We challenge
            const challengeSim = this.cloneState(det);
            const challengeOutcome = this.simulateChallengeOutcome(
                this.playerId, claimantId, action, challengeSim
            );
            if (challengeOutcome !== null) {
                challengeValue += this.rollout(challengeOutcome);
                challengeCount++;
            }
            
            // Branch 2: We don't challenge (action proceeds)
            const noChallengeSim = this.cloneState(det);
            const noChallengeOutcome = this.simulateActionProceeds(
                claimantId, action, gameState.pendingAction, noChallengeSim
            );
            if (noChallengeOutcome !== null) {
                noChallengeValue += this.rollout(noChallengeOutcome);
                noChallengeCount++;
            }
        }

        // Compare average values (avoid division by zero)
        const avgChallengeValue = challengeCount > 0 ? challengeValue / challengeCount : 0;
        const avgNoChallengeValue = noChallengeCount > 0 ? noChallengeValue / noChallengeCount : 0;
        
        // Challenge if expected value is better
        return avgChallengeValue > avgNoChallengeValue;
    }

    simulateChallengeOutcome(challengerId, claimantId, action, state) {
        // Simulate the challenge resolution
        const claimant = state.players[claimantId];
        const challenger = state.players[challengerId];
        
        if (!claimant || !challenger) return null;
        
        const required = this.getRequiredCard(action);
        const hasCard = claimant.cards.some(c => !c.revealed && c.character === required);

        if (hasCard) {
            // Challenge fails - we (challenger) lose influence
            this.simLoseInfluence(challenger, state);
            
            // Claimant proves card and action proceeds
            // (In real game they'd shuffle card back, but we skip for simulation speed)
            const actionObj = { 
                action: action, 
                targetId: state.pendingAction?.targetId 
            };
            this.simResolveAction(claimantId, actionObj, state);
        } else {
            // Challenge succeeds - claimant loses influence, action fails
            if (action === 'assassinate') {
                claimant.coins += 3; // Refund
            }
            this.simLoseInfluence(claimant, state);
        }

        // Advance turn (claimant's turn is over)
        state.currentPlayerIndex = this.nextAliveAfter(state, claimantId);
        
        return state;
    }

    simulateActionProceeds(actorId, action, pendingAction, state) {
        // Simulate the action going through without challenge
        const actionObj = {
            action: action,
            targetId: pendingAction?.targetId
        };
        
        this.simResolveAction(actorId, actionObj, state);
        
        // Advance turn (actor's turn is over)
        state.currentPlayerIndex = this.nextAliveAfter(state, actorId);
        
        return state;
    }

    decideBlockAction(action, actorId, gameState, gameHistory) {
        const me = gameState.players[this.playerId];
        if (!me) return false;

        const blockCards = this.getPossibleBlockCards(action);
        if (blockCards.length === 0) return false;

        // Check if we actually have a blocking card
        const hasRealBlock = blockCards.some(c => me.cards.some(x => !x.revealed && x.character === c));
        
        // If we don't have a blocking card, rarely bluff
        if (!hasRealBlock) {
            if (action === 'assassinate' && Math.random() < 0.06) return true;
            if (action === 'steal' && Math.random() < 0.08) return true;
            if (action === 'foreign-aid' && Math.random() < 0.05) return true;
            return false;
        }

        // We have a real block - decide whether to use it
        // For dangerous actions (assassinate/steal targeting us), use MCTS
        const isDangerous = (action === 'assassinate' || action === 'steal') && 
                           gameState.pendingAction?.targetId === this.playerId;
        const isEndgame = gameState.players.filter(p => !p.eliminated).length <= 3;
        
        if (isDangerous || isEndgame) {
            return this.mctsDecideBlock(action, actorId, gameState, gameHistory);
        }

        // For non-critical blocks, use high probability heuristic
        if (action === 'assassinate') return Math.random() < 0.92;
        if (action === 'steal') return Math.random() < 0.88;
        if (action === 'foreign-aid') return Math.random() < 0.80;
        return Math.random() < 0.85;
    }

    mctsDecideBlock(action, actorId, gameState, gameHistory) {
        // Use MCTS to evaluate: block vs don't block
        
        let blockValue = 0;
        let noBlockValue = 0;
        let blockCount = 0;
        let noBlockCount = 0;
        const iterations = this.BLOCK_ITERATIONS;

        for (let i = 0; i < iterations; i++) {
            const det = this.determinize(gameState, gameHistory);
            
            // Branch 1: We block
            const blockSim = this.cloneState(det);
            const blockOutcome = this.simulateBlockOutcome(
                this.playerId, actorId, action, blockSim
            );
            if (blockOutcome !== null) {
                blockValue += this.rollout(blockOutcome);
                blockCount++;
            }
            
            // Branch 2: We don't block (action proceeds)
            const noBlockSim = this.cloneState(det);
            const noBlockOutcome = this.simulateActionProceeds(
                actorId, action, gameState.pendingAction, noBlockSim
            );
            if (noBlockOutcome !== null) {
                noBlockValue += this.rollout(noBlockOutcome);
                noBlockCount++;
            }
        }

        // Compare average values (avoid division by zero)
        const avgBlockValue = blockCount > 0 ? blockValue / blockCount : 0;
        const avgNoBlockValue = noBlockCount > 0 ? noBlockValue / noBlockCount : 0;
        
        // Block if expected value is better
        return avgBlockValue > avgNoBlockValue;
    }

    simulateBlockOutcome(blockerId, actorId, action, state) {
        // Simulate the block - action is prevented
        // In real game, block could be challenged, but we simplify here
        // (Challenge-after-block is rare and complex to simulate)
        
        // Action is blocked - nothing happens, actor's turn ends
        state.currentPlayerIndex = this.nextAliveAfter(state, actorId);
        
        return state;
    }

    decideBlockClaim(action, actorId, gameState, gameHistory) {
        const me = gameState.players[this.playerId];
        if (!me) return null;

        if (action === 'assassinate') return 'contessa';

        if (action === 'steal') {
            const hasCaptain = me.cards.some(c => !c.revealed && c.character === 'captain');
            const hasAmb = me.cards.some(c => !c.revealed && c.character === 'ambassador');
            if (hasCaptain) return 'captain';
            if (hasAmb) return 'ambassador';
            return Math.random() < 0.5 ? 'captain' : 'ambassador';
        }

        if (action === 'foreign-aid') return 'duke';
        return null;
    }

    // ===== MCTS / Simulation Helpers =====

    uctScore(child, parent) {
        if (child.visits === 0) return Infinity;
        const exploit = child.totalValue / child.visits;
        const explore = this.UCT_C * Math.sqrt(Math.log(Math.max(1, parent.visits)) / child.visits);
        return exploit + explore;
    }

    backpropagate(node, reward) {
        let cur = node;
        while (cur) {
            cur.visits += 1;
            cur.totalValue += reward;
            cur = cur.parent;
        }
    }

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

    evaluate(state) {
        const alive = state.players.filter(p => !p.eliminated);
        if (alive.length === 1) {
            return alive[0].id === this.playerId ? 10000 : -10000;
        }

        const me = state.players[this.playerId];
        if (!me) return -10000;
        
        const myInf = me.cards.filter(c => !c.revealed).length;

        let score = 0;
        score += myInf * 400;
        score += me.coins * 18;

        for (const p of state.players) {
            if (p.id === this.playerId) continue;
            const inf = p.cards.filter(c => !c.revealed).length;
            score -= inf * 220;
            score -= p.coins * 10;
        }

        return score;
    }

    determinize(gameState, gameHistory) {
        const base = this.cloneState(gameState);

        const probCalc = new ProbabilityCalculator(gameHistory, base, this.playerId);
        const unseen = probCalc.getUnseenCards();

        this.shuffleInPlace(unseen);

        // Assign cards to other players
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
                if (this.simPolicyBlock(b.id, 'foreign-aid', actorIndex, s)) {
                    s.currentPlayerIndex = this.nextAliveAfter(s, actorIndex);
                    return s;
                }
            }
        } else if ((actionObj.action === 'assassinate' || actionObj.action === 'steal') && actionObj.targetId != null) {
            const target = s.players[actionObj.targetId];
            if (target && !target.eliminated && this.simPolicyBlock(target.id, actionObj.action, actorIndex, s)) {
                s.currentPlayerIndex = this.nextAliveAfter(s, actorIndex);
                return s;
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

            case 'coup': {
                actor.coins -= 7;
                const target = s.players[tId];
                if (target) this.simLoseInfluence(target, s);
                break;
            }

            case 'assassinate': {
                actor.coins -= 3;
                const target = s.players[tId];
                if (target) this.simLoseInfluence(target, s);
                break;
            }

            case 'steal': {
                const target = s.players[tId];
                if (!target) break;
                const stolen = Math.min(2, target.coins);
                target.coins -= stolen;
                actor.coins += stolen;
                break;
            }

            case 'exchange': {
                const aliveCards = actor.cards.filter(c => !c.revealed);
                const n = aliveCards.length;

                const drawn = [];
                for (let i = 0; i < n && s.deck.length > 0; i++) {
                    drawn.push(s.deck.pop());
                }

                for (const c of aliveCards) {
                    s.deck.push(c.character);
                }

                let idx = 0;
                for (const c of actor.cards) {
                    if (!c.revealed && idx < drawn.length) {
                        c.character = drawn[idx++];
                    }
                }

                this.shuffleInPlace(s.deck);
                break;
            }
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
            // Challenge failed - challenger loses influence
            this.simLoseInfluence(challenger, s);

            // Claimant returns card and draws new one
            if (s.deck.length > 0) {
                s.deck.push(claimant.cards[hasCardIndex].character);
                this.shuffleInPlace(s.deck);
                claimant.cards[hasCardIndex].character = s.deck.pop();
            }

            return false; // Action proceeds
        } else {
            // Challenge succeeded - claimant loses influence
            if (action === 'assassinate') {
                claimant.coins += 3; // Refund assassinate cost
            }
            this.simLoseInfluence(claimant, s);
            return true; // Action blocked
        }
    }

    simLoseInfluence(player, s) {
        if (!player) return;
        
        const idx = player.cards.findIndex(c => !c.revealed);
        if (idx === -1) return;
        player.cards[idx].revealed = true;
        if (player.cards.every(c => c.revealed)) {
            player.eliminated = true;
            player.coins = 0;
        }
    }

    simCheckElims(s) {
        for (const p of s.players) {
            if (!p.eliminated && p.cards.every(c => c.revealed)) {
                p.eliminated = true;
                p.coins = 0;
            }
        }
    }

    // ===== Policies used inside simulations =====

    simPolicyChallenge(challengerId, action, claimantId, s) {
        const required = this.getRequiredCard(action);
        const claimant = s.players[claimantId];
        const challenger = s.players[challengerId];
        
        if (!claimant || !challenger) return false;

        // Use belief-based challenge decisions (NOT perfect information)
        // Simulate what the challenger would believe based on history
        
        // Count revealed cards of this type (public information)
        let revealedOfType = 0;
        // Note: In simulation we don't have full game history, use approximation
        for (const p of s.players) {
            for (const card of p.cards) {
                if (card.revealed && card.character === required) {
                    revealedOfType++;
                }
            }
        }
        
        // If all 3 copies revealed, definitely challenge (impossible claim)
        if (revealedOfType === 3) return true;
        
        // Estimate probability claimant has the card based on public info
        // Simple heuristic: base probability adjusted by revealed cards
        const totalOfType = 3;
        const unseenOfType = totalOfType - revealedOfType;
        const claimantActiveCards = claimant.cards.filter(c => !c.revealed).length;
        
        if (claimantActiveCards === 0 || unseenOfType === 0) return true;
        
        // Rough probability estimate (not perfect - intentionally imperfect)
        const baseProbHas = unseenOfType / (unseenOfType + 10); // Rough approximation
        
        // Adjust for danger - challenge more aggressively if action threatens us
        let dangerBonus = 0;
        if (action === 'assassinate' || action === 'steal') {
            // Check if we might be the target (we don't know targetId in policy, but can estimate)
            dangerBonus = 0.15;
        }
        if (challenger.cards.filter(c => !c.revealed).length === 1) {
            // More cautious when we only have one card left
            dangerBonus -= 0.10;
        }
        
        // Challenge threshold with noise
        const threshold = 0.35 - dangerBonus;
        const noise = Math.random() * 0.12 - 0.06; // +/- 6% noise
        
        return (baseProbHas + noise) < threshold;
    }

    simPolicyBlock(blockerId, action, actorId, s) {
        const blocker = s.players[blockerId];
        if (!blocker) return false;
        
        const cards = this.getPossibleBlockCards(action);
        if (cards.length === 0) return false;

        // Check if blocker actually has a blocking card
        const has = cards.some(req => blocker.cards.some(c => !c.revealed && c.character === req));
        
        if (has) {
            // Has real block - very likely to use it
            // But not 100% to simulate occasional strategic non-blocks
            if (action === 'assassinate') return Math.random() < 0.93;
            if (action === 'steal') return Math.random() < 0.88;
            if (action === 'foreign-aid') return Math.random() < 0.82;
            return Math.random() < 0.85;
        }

        // Bluffing blocks (much rarer, belief-based)
        // Only bluff if:
        // 1. Action is dangerous enough
        // 2. Not too many of the blocking cards are revealed
        let revealedBlockers = 0;
        for (const p of s.players) {
            for (const card of p.cards) {
                if (card.revealed && cards.includes(card.character)) {
                    revealedBlockers++;
                }
            }
        }
        
        // More likely to bluff if fewer blocking cards are revealed
        const bluffFactor = Math.max(0, 1 - revealedBlockers / 4);
        
        if (action === 'assassinate') return Math.random() < 0.08 * bluffFactor;
        if (action === 'steal') return Math.random() < 0.05 * bluffFactor;
        if (action === 'foreign-aid') return Math.random() < 0.03 * bluffFactor;
        
        return false;
    }

    opponentPolicyChooseAction(player, s) {
        if (player.coins >= 10) return this.chooseBestCoupTarget(player, s);

        const actions = this.getPossibleActions(player, s);
        if (actions.length === 0) return { action: 'income' };
        
        // Use smart expected value scoring (fast version of StatisticalAI logic)
        const scored = actions.map(a => ({
            a,
            ev: this.fastExpectedValue(a, player, s)
        }));
        
        // Sort by expected value
        scored.sort((x, y) => y.ev - x.ev);
        
        // Mostly pick best (80%), sometimes pick 2nd best (15%), rarely random (5%)
        const roll = Math.random();
        if (roll < 0.80 && scored.length > 0) return scored[0].a;
        if (roll < 0.95 && scored.length > 1) return scored[1].a;
        return scored[Math.floor(Math.random() * scored.length)].a;
    }

    selfRolloutPolicyChooseAction(player, s) {
        if (player.coins >= 10) return this.chooseBestCoupTarget(player, s);

        const actions = this.getPossibleActions(player, s);
        if (actions.length === 0) return { action: 'income' };
        
        // For our own rollouts, be even more greedy
        const scored = actions.map(a => ({
            a,
            ev: this.fastExpectedValue(a, player, s)
        }));
        
        scored.sort((x, y) => y.ev - x.ev);
        
        // Pick best 90% of time, 2nd best 8%, random 2%
        const roll = Math.random();
        if (roll < 0.90 && scored.length > 0) return scored[0].a;
        if (roll < 0.98 && scored.length > 1) return scored[1].a;
        return scored[Math.floor(Math.random() * scored.length)].a;
    }

    fastExpectedValue(action, player, s) {
        const a = action.action;
        
        // Base values for each action type
        let value = 0;
        
        if (a === 'coup') {
            if (player.coins >= 7) {
                value = 100 + this.targetThreatScore(action.targetId, s);
            } else {
                return -1000; // Invalid
            }
        } else if (a === 'assassinate') {
            if (player.coins >= 3) {
                value = 70 + this.targetThreatScore(action.targetId, s);
                // Risk of challenge (rough estimate)
                value -= 15; // Average risk penalty
            } else {
                return -1000; // Invalid
            }
        } else if (a === 'steal') {
            const target = s.players[action.targetId];
            if (target && !target.eliminated) {
                const stolen = Math.min(2, target.coins);
                value = stolen * 15 + target.coins * 2;
                // Risk of block
                value *= 0.6; // Expect ~40% block rate
            } else {
                return -1000; // Invalid target
            }
        } else if (a === 'tax') {
            value = 25;
            // Small risk of challenge
            value -= 3;
        } else if (a === 'foreign-aid') {
            value = 18;
            // Risk of duke block
            value *= 0.75; // Expect ~25% block rate
        } else if (a === 'income') {
            value = 10; // Safe but low value
        } else if (a === 'exchange') {
            value = 20; // Card quality improvement
            // Risk of challenge
            value -= 5;
        }
        
        // Strategic modifiers based on game state
        const activePlayers = s.players.filter(p => !p.eliminated);
        
        // Guard against division by zero
        if (activePlayers.length === 0) return value;
        
        const avgCoins = activePlayers.reduce((sum, p) => sum + p.coins, 0) / activePlayers.length;
        
        // Prefer income-generating actions when behind on coins
        if (player.coins < avgCoins - 2) {
            if (['income', 'foreign-aid', 'tax'].includes(a)) value += 8;
        }
        
        // Prefer aggressive actions when ahead
        if (player.coins > avgCoins + 2) {
            if (['coup', 'assassinate', 'steal'].includes(a)) value += 12;
        }
        
        // Endgame: prefer elimination actions
        if (activePlayers.length <= 3) {
            if (a === 'coup' || a === 'assassinate') value += 20;
        }
        
        // Avoid actions we can't afford
        if (a === 'coup' && player.coins < 7) return -1000;
        if (a === 'assassinate' && player.coins < 3) return -1000;
        
        return value;
    }

    targetThreatScore(tid, s) {
        const t = s.players[tid];
        if (!t || t.eliminated) return 0;
        const inf = t.cards.filter(c => !c.revealed).length;
        return t.coins * 4 + inf * 40;
    }

    targetCoinScore(tid, s) {
        const t = s.players[tid];
        if (!t || t.eliminated) return 0;
        return Math.min(2, t.coins) * 10 + t.coins;
    }

    // ===== Utility / Rules =====

    getPossibleActions(player, gameState) {
        if (!player) return [{ action: 'income' }];
        
        const actions = [];
        const targets = gameState.players.filter(p => p && p.id !== player.id && !p.eliminated);

        actions.push({ action: 'income' });
        actions.push({ action: 'foreign-aid' });
        actions.push({ action: 'tax' });
        actions.push({ action: 'exchange' });

        if (player.coins >= 3 && targets.length > 0) {
            for (const t of targets) actions.push({ action: 'assassinate', targetId: t.id });
        }

        if (targets.length > 0) {
            for (const t of targets) actions.push({ action: 'steal', targetId: t.id });
        }

        if (player.coins >= 7 && targets.length > 0) {
            for (const t of targets) actions.push({ action: 'coup', targetId: t.id });
        }

        return actions;
    }

    chooseBestCoupTarget(player, gameState) {
        if (!player) return { action: 'income' };
        
        const targets = gameState.players.filter(p => p.id !== player.id && !p.eliminated);
        if (targets.length === 0) return { action: 'income' };

        targets.sort((a, b) => {
            const aInf = a.cards.filter(c => !c.revealed).length;
            const bInf = b.cards.filter(c => !c.revealed).length;
            const dInf = bInf - aInf;
            if (dInf !== 0) return dInf;
            return b.coins - a.coins;
        });

        return { action: 'coup', targetId: targets[0].id };
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

    challengeDangerBonus(action, claimantId, gameState) {
        if (gameState.pendingAction && gameState.pendingAction.targetId === this.playerId) return 0.06;

        const c = gameState.players[claimantId];
        if (!c) return 0;
        if (c.coins >= 7) return 0.04;
        return 0.0;
    }

    safeHeuristicFallback(player, gameState) {
        if (!player) return { action: 'income' };
        
        if (player.coins >= 7) return this.chooseBestCoupTarget(player, gameState);
        if (player.coins >= 3) {
            const targets = gameState.players.filter(p => p && p.id !== player.id && !p.eliminated);
            if (targets.length) return { action: 'assassinate', targetId: targets[0].id };
        }
        return { action: 'tax' };
    }

    isTerminal(s) {
        return s.players.filter(p => !p.eliminated).length <= 1;
    }

    nextAliveAfter(s, fromIndex) {
        const n = s.players.length;
        for (let k = 1; k <= n; k++) {
            const i = (fromIndex + k) % n;
            if (s.players[i] && !s.players[i].eliminated) return i;
        }
        return fromIndex;
    }

    actionKey(a) {
        return a.action + ':' + (a.targetId ?? 'none');
    }

    cloneState(gameState) {
        return {
            currentPlayerIndex: gameState.currentPlayerIndex,
            deck: Array.isArray(gameState.deck) ? gameState.deck.slice() : [],
            players: gameState.players.map(p => ({
                id: p.id,
                coins: p.coins,
                eliminated: !!p.eliminated,
                cards: p.cards.map(c => ({ character: c.character, revealed: !!c.revealed }))
            }))
        };
    }

    shuffleInPlace(arr) {
        for (let i = arr.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [arr[i], arr[j]] = [arr[j], arr[i]];
        }
    }
}