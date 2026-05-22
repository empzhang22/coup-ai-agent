// statistical-ai.js
// StatisticalAI with persistent particle filtering (history-consistent beliefs)
// FIXED VERSION with proper terminal state evaluation and endgame tactics

class ParticleBeliefTracker {
    constructor(selfId, opts = {}) {
        this.selfId = selfId;

        // Tuning knobs
        this.N = opts.numParticles ?? 500;
        this.resampleEssFrac = opts.resampleEssFrac ?? 0.55;

        // Likelihood model for unchallenged claims
        this.truthLikelihood = opts.truthLikelihood ?? 0.88; // P(claim action | has card)
        this.bluffLikelihood = opts.bluffLikelihood ?? 0.12; // P(claim action | lacks card)

        // Constrained repair attempts for hard constraints
        this.repairAttempts = opts.repairAttempts ?? 12;

        this._rng = opts.rng ?? Math.random;

        // Persistent state
        this.particles = null; // [{ hands: string[][], deck: string[], w: number }]
        this.lastHistoryLen = 0;

        // For detecting public info changes
        this._lastRevealedKey = "";
    }

    // --- Public API ---

    sync(gameState, gameHistory) {
        const histLen = (gameHistory?.actionHistory?.length ?? 0);

        // Re-init triggers:
        // - first time
        // - history reset/truncated
        // - revealed cards changed (public info update)
        const revealedKey = this._fingerprintRevealed(gameHistory);
        const needInit =
            !this.particles ||
            this.particles.length !== this.N ||
            histLen < this.lastHistoryLen ||
            revealedKey !== this._lastRevealedKey;

        if (needInit) {
            this._initFromPublicState(gameState, gameHistory);
            this.lastHistoryLen = histLen;
            this._lastRevealedKey = revealedKey;
            return;
        }

        // Apply only new history entries
        for (let i = this.lastHistoryLen; i < histLen; i++) {
            const ev = gameHistory.actionHistory[i];
            this._applyHistoryEvent(gameState, gameHistory, ev);
        }

        this.lastHistoryLen = histLen;
    }

    probHas(playerIndex, card) {
        if (!this.particles || this.particles.length === 0) return 0.2;
        let num = 0;
        let den = 0;

        for (const p of this.particles) {
            const w = p.w;
            den += w;
            const hand = p.hands[playerIndex] || [];
            if (hand.includes(card)) num += w;
        }
        return den > 0 ? num / den : 0.2;
    }

    calculateCardProbabilities(playerIndex) {
        const probs = {};
        if (!this.particles || this.particles.length === 0) {
            const base = 1 / CHARACTERS.length;
            for (const c of CHARACTERS) probs[c] = base;
            return probs;
        }

        let den = 0;
        const acc = {};
        for (const c of CHARACTERS) acc[c] = 0;

        for (const p of this.particles) {
            const w = p.w;
            den += w;
            const hand = p.hands[playerIndex] || [];
            for (const c of CHARACTERS) {
                if (hand.includes(c)) acc[c] += w;
            }
        }

        if (den <= 0) {
            const base = 1 / CHARACTERS.length;
            for (const c of CHARACTERS) probs[c] = base;
            return probs;
        }

        // Normalize to sum to 1 (like old ProbabilityCalculator)
        let sum = 0;
        for (const c of CHARACTERS) {
            probs[c] = acc[c] / den;
            sum += probs[c];
        }
        if (sum > 0) {
            for (const c of CHARACTERS) probs[c] /= sum;
        }
        return probs;
    }

    sampleParticleWorld() {
        if (!this.particles || this.particles.length === 0) return null;

        // Normalize weights just in case
        this._normalizeWeights();

        const cdf = [];
        let acc = 0;
        for (const p of this.particles) {
            acc += p.w;
            cdf.push(acc);
        }

        const u = this._rng() * acc;
        let i = 0;
        while (u > cdf[i]) {
            i++;
        }
        
        const sampledParticle = this.particles[i];
        if (!sampledParticle) return null;

        // Return a deep copy to prevent mutation of the particle itself
        return {
            hands: sampledParticle.hands.map(h => h.slice()),
            deck: sampledParticle.deck.slice()
        };
    }

    // --- Initialization (public info -> initial belief) ---

    _initFromPublicState(gameState, gameHistory) {
        const playerCount = gameState.players.length;

        // All cards multiset: 3 copies each character
        const all = [];
        for (const c of CHARACTERS) for (let i = 0; i < 3; i++) all.push(c);

        // Remove revealed cards (public info)
        for (let i = 0; i < playerCount; i++) {
            const revealed = this._revealedForIndex(gameHistory, i);
            for (const rc of revealed) {
                const idx = all.indexOf(rc);
                if (idx !== -1) all.splice(idx, 1);
            }
        }

        // Remove our known unrevealed cards from the unseen multiset
        const self = gameState.players[this.selfId];
        if (self) {
            for (const cardObj of self.cards) {
                if (!cardObj.revealed) {
                    const idx = all.indexOf(cardObj.character);
                    if (idx !== -1) all.splice(idx, 1);
                }
            }
        }

        // How many hidden cards to sample for each opponent (alive influences)
        const hiddenCounts = new Array(playerCount).fill(0);
        for (let i = 0; i < playerCount; i++) {
            const pl = gameState.players[i];
            if (!pl || pl.eliminated) {
                hiddenCounts[i] = 0;
                continue;
            }
            const alive = pl.cards.filter(c => !c.revealed).length;
            hiddenCounts[i] = (i === this.selfId) ? 0 : alive;
        }

        // Build particles
        this.particles = [];
        for (let n = 0; n < this.N; n++) {
            const deck = all.slice();
            this._shuffle(deck);

            const hands = new Array(playerCount);
            for (let i = 0; i < playerCount; i++) hands[i] = [];

            for (let i = 0; i < playerCount; i++) {
                const k = hiddenCounts[i];
                for (let t = 0; t < k; t++) {
                    if (deck.length === 0) break;
                    hands[i].push(deck.pop());
                }
            }

            this.particles.push({ hands, deck, w: 1.0 });
        }

        this._normalizeWeightsOrReinit(gameState, gameHistory);
    }

    // --- Filtering: apply one observed history event ---

    _applyHistoryEvent(gameState, gameHistory, ev) {
        if (!ev || !ev.action) return;

        const playerCount = gameState.players.length;

        // actionHistory might store ids OR indices; normalize to indices here
        const claimantIdx = this._coerceToIndex(gameState, ev.playerId);
        const targetIdx = (ev.targetId !== undefined && ev.targetId !== null)
            ? this._coerceToIndex(gameState, ev.targetId)
            : null;

        if (claimantIdx < 0 || claimantIdx >= playerCount) return;

        const action = ev.action;
        const required = this._requiredCard(action);
        const isClaim = !!required;

        // 1) Soft evidence: unchallenged claim slightly increases weight if particle supports it
        if (isClaim && !ev.challenged) {
            for (const p of this.particles) {
                const has = (p.hands[claimantIdx] || []).includes(required);
                p.w *= has ? this.truthLikelihood : this.bluffLikelihood;
            }
            // periodic normalization to prevent underflow
            this._normalizeIfTiny();
        }

        // 2) Hard constraints if challenged
        if (isClaim && ev.challenged) {
            const claimantHadCard = !!ev.success; // see assumption above

            // Enforce constraints (repair if needed)
            for (const p of this.particles) {
                if (p.w <= 0) continue;

                const hand = p.hands[claimantIdx] || [];
                const has = hand.includes(required);

                if (claimantHadCard && !has) {
                    const ok = this._repairForceHas(p, claimantIdx, required);
                    if (!ok) p.w = 0;
                }

                if (!claimantHadCard && has) {
                    // Optional repair to force NOT-have could be added; simplest is kill
                    p.w = 0;
                }
            }

            // Normalize; if all died, re-init immediately
            if (!this._normalizeWeightsOrReinit(gameState, gameHistory)) return;

            // Apply transition for challenge failed (claimant showed card then replaced it)
            if (claimantHadCard) {
                for (const p of this.particles) {
                    if (p.w <= 0) continue;
                    this._applyChallengeFailedReplacement(p, claimantIdx, required);
                }
            }
        }

        // 3) Exchange transition (even if unchallenged)
        if (action === 'exchange') {
            for (const p of this.particles) {
                if (p.w <= 0) continue;
                this._applyExchangeTransition(p, claimantIdx);
            }
        }

        // Degeneracy check -> resample + rejuvenate
        const ess = this._effectiveSampleSize();
        if (ess < this.resampleEssFrac * this.N) {
            this._resampleSystematic();
            this._rejuvenate(1);
        }
    }

    // --- Transitions ---

    _applyChallengeFailedReplacement(p, claimantIdx, required) {
        const hand = p.hands[claimantIdx] || [];
        const idx = hand.indexOf(required);
        if (idx === -1) return;

        // return proven card to deck, shuffle, draw replacement
        p.deck.push(required);
        this._shuffle(p.deck);

        if (p.deck.length > 0) {
            hand[idx] = p.deck.pop();
        }
    }

    _applyExchangeTransition(p, claimantIdx) {
        // For self exchanges, we know our real cards; belief doesn't need to sample self.
        if (claimantIdx === this.selfId) return;

        const hand = p.hands[claimantIdx] || [];
        const n = hand.length;
        if (n <= 0) return;

        const pool = hand.slice();
        for (let i = 0; i < 2 && p.deck.length > 0; i++) {
            pool.push(p.deck.pop());
        }

        this._shuffle(pool);
        p.hands[claimantIdx] = pool.slice(0, n);
        p.deck.push(...pool.slice(n));
        this._shuffle(p.deck);
    }

    // --- Constrained repairs ---

    _repairForceHas(p, claimantIdx, required) {
        for (let attempt = 0; attempt < this.repairAttempts; attempt++) {
            // Try to pull required from deck
            const deckIdx = p.deck.indexOf(required);
            if (deckIdx !== -1) {
                p.deck.splice(deckIdx, 1);
                const hand = p.hands[claimantIdx] || [];
                if (hand.length > 0) {
                    const swapIdx = Math.floor(this._rng() * hand.length);
                    const giveBack = hand[swapIdx];
                    hand[swapIdx] = required;
                    p.deck.push(giveBack);
                } else {
                    p.hands[claimantIdx] = [required];
                }
                return true;
            }

            // Otherwise steal by swapping with another player's hidden hand
            const other = this._randOtherPlayer(p.hands.length, claimantIdx);
            if (other === null) break;

            const otherHand = p.hands[other] || [];
            const oIdx = otherHand.indexOf(required);
            if (oIdx !== -1) {
                const claimantHand = p.hands[claimantIdx] || [];
                if (claimantHand.length === 0) {
                    otherHand.splice(oIdx, 1);
                    claimantHand.push(required);
                    p.hands[claimantIdx] = claimantHand;

                    if (p.deck.length > 0) otherHand.push(p.deck.pop());
                    return true;
                } else {
                    const swapIdx = Math.floor(this._rng() * claimantHand.length);
                    const temp = claimantHand[swapIdx];
                    claimantHand[swapIdx] = required;
                    otherHand[oIdx] = temp;
                    return true;
                }
            }
        }
        return false;
    }

    _randOtherPlayer(playerCount, excludeIdx) {
        if (playerCount <= 1) return null;
        let tries = 0;
        while (tries++ < 10) {
            const j = Math.floor(this._rng() * playerCount);
            if (j !== excludeIdx) return j;
        }
        for (let j = 0; j < playerCount; j++) if (j !== excludeIdx) return j;
        return null;
    }

    // --- Resampling / rejuvenation ---

    _effectiveSampleSize() {
        if (!this.particles || this.particles.length === 0) return 0;
        let sumW = 0;
        let sumW2 = 0;
        for (const p of this.particles) {
            sumW += p.w;
            sumW2 += p.w * p.w;
        }
        if (sumW2 <= 0) return 0;
        return (sumW * sumW) / sumW2;
    }

    _resampleSystematic() {
        // Normalize first; if all died, caller should have re-init'd already
        this._normalizeWeights();

        const N = this.N;
        const newParts = new Array(N);

        // Build CDF
        const cdf = new Array(N);
        let acc = 0;
        for (let i = 0; i < N; i++) {
            acc += this.particles[i].w;
            cdf[i] = acc;
        }

        const step = 1 / N;
        let u = this._rng() * step;
        let i = 0;

        for (let j = 0; j < N; j++) {
            while (u > cdf[i]) i++;

            const src = this.particles[i];
            const hands = src.hands.map(h => h.slice());
            const deck = src.deck.slice();
            newParts[j] = { hands, deck, w: 1.0 };

            u += step;
        }

        this.particles = newParts;
        this._normalizeWeights();
    }

    _rejuvenate(passes = 1) {
        // Light random swap moves to avoid particle collapse
        for (let pass = 0; pass < passes; pass++) {
            for (const p of this.particles) {
                if (p.w <= 0) continue;
                if (this._rng() < 0.35) this._randomSwapMove(p);
            }
        }
        this._normalizeWeights();
    }

    _randomSwapMove(p) {
        const P = p.hands.length;
        if (P <= 1) return;

        const a = Math.floor(this._rng() * P);
        const b = Math.floor(this._rng() * P);
        if (a === b) return;

        const ha = p.hands[a] || [];
        const hb = p.hands[b] || [];
        if (ha.length === 0 || hb.length === 0) return;

        const ia = Math.floor(this._rng() * ha.length);
        const ib = Math.floor(this._rng() * hb.length);

        const tmp = ha[ia];
        ha[ia] = hb[ib];
        hb[ib] = tmp;
    }

    // --- Weight normalization helpers ---

    _normalizeWeights() {
        let sum = 0;
        for (const p of this.particles) sum += p.w;
        if (sum <= 0) return false;
        for (const p of this.particles) p.w /= sum;
        return true;
    }

    _normalizeWeightsOrReinit(gameState, gameHistory) {
        const ok = this._normalizeWeights();
        if (ok) return true;
        // all particles died -> re-init from public state
        this._initFromPublicState(gameState, gameHistory);
        return false; // caller should stop applying this event (fresh init already reflects public info)
    }

    _normalizeIfTiny() {
        // prevent underflow: if all weights become extremely tiny, renormalize
        let maxW = 0;
        for (const p of this.particles) if (p.w > maxW) maxW = p.w;
        if (maxW < 1e-30) this._normalizeWeights();
    }

    // --- Utilities ---

    _shuffle(arr) {
        for (let i = arr.length - 1; i > 0; i--) {
            const j = Math.floor(this._rng() * (i + 1));
            [arr[i], arr[j]] = [arr[j], arr[i]];
        }
    }

    _requiredCard(action) {
        const map = {
            tax: 'duke',
            assassinate: 'assassin',
            steal: 'captain',
            exchange: 'ambassador'
        };
        return map[action] || null;
    }

    _fingerprintRevealed(gameHistory) {
        // Stable-ish fingerprint of revealed cards per player index
        const r = gameHistory?.revealedCards;
        if (!Array.isArray(r)) return "none";
        // Sort each player's revealed list for stability
        const parts = r.map(arr => (Array.isArray(arr) ? arr.slice().sort().join(',') : '')).join('|');
        return parts;
    }

    _revealedForIndex(gameHistory, playerIndex) {
        const r = gameHistory?.revealedCards;
        if (!Array.isArray(r)) return [];
        const arr = r[playerIndex];
        return Array.isArray(arr) ? arr : [];
    }

    _coerceToIndex(gameState, idOrIndex) {
        // Supports either:
        // - already an index (0..N-1)
        // - a player object id (gameState.players[i].id === idOrIndex)
        const N = gameState.players.length;

        if (typeof idOrIndex === 'number' && Number.isInteger(idOrIndex)) {
            if (idOrIndex >= 0 && idOrIndex < N) return idOrIndex;
        }

        // Try match by id
        for (let i = 0; i < N; i++) {
            if (gameState.players[i] && gameState.players[i].id === idOrIndex) return i;
        }

        // Last resort: parse numeric strings
        if (typeof idOrIndex === 'string') {
            const x = Number(idOrIndex);
            if (Number.isInteger(x) && x >= 0 && x < N) return x;
        }

        return -1;
    }
}

// --------------------- StatisticalAI using ParticleBeliefTracker ---------------------

class StatisticalAI extends AIEngine {
    constructor(playerId, aiType) {
        super(playerId, aiType);

        // Belief uses "selfId" as index
        this.belief = new ParticleBeliefTracker(playerId, {
            numParticles: 500,
            truthLikelihood: 0.88,
            bluffLikelihood: 0.12,
            resampleEssFrac: 0.55
        });
    }

    chooseAction(player, gameState, gameHistory) {
        this.belief.sync(gameState, gameHistory);

        // Forced coup rule
        if (player.coins >= 10) return this.chooseBestCoupTarget(player, gameState);

        const actions = this.getPossibleActions(player, gameState);
        if (actions.length === 1) return actions[0];

        const scored = actions.map(a => ({
            action: a,
            expectedValue: this.calculateExpectedValue(a, player, gameState, gameHistory),
            tiebreaker: Math.random()
        }));

        scored.sort((x, y) => {
            const d = y.expectedValue - x.expectedValue;
            return d !== 0 ? d : (y.tiebreaker - x.tiebreaker);
        });

        return scored[0].action;
    }

    calculateExpectedValue(actionObj, player, gameState, gameHistory) {
        this.belief.sync(gameState, gameHistory);

        // CRITICAL: Check for terminal states first
        if (this.isWinningMove(actionObj, gameState)) {
            return 100000;  // Guaranteed win dominates everything
        }

        if (this.isSuicideMove(actionObj, player, gameState, gameHistory)) {
            return -100000;  // Avoid guaranteed elimination
        }

        let value = this.getBaseActionValue(actionObj, player, gameState);

        // Risk of being challenged if we claim a character we likely don't have
        if (this.isChallengeable(actionObj.action)) {
            const required = this.getRequiredCard(actionObj.action);
            const hasCard = this.playerHasCard(player, required);

            let challengeProb = this.estimateChallengeProbability(
                required,
                actionObj,
                player,
                gameState,
                gameHistory
            );

            if (!hasCard) {
                const myInfluences = player.cards.filter(c => !c.revealed).length;
                const penalty = (myInfluences === 1) ? 100000 : 50;

                value = value * (1 - challengeProb) - penalty * challengeProb;
            }
        }

        value += this.strategicModifiers(actionObj, player, gameState);

        // Use belief on whether target can block
        if (actionObj.action === 'steal' && actionObj.targetId !== undefined) {
            const targetIdx = this._coerceToIndex(gameState, actionObj.targetId);
            if (targetIdx !== -1) {
                const pBlockCaptain = this.belief.probHas(targetIdx, 'captain');
                const pBlockAmb = this.belief.probHas(targetIdx, 'ambassador');
                const pCanBlock = Math.max(pBlockCaptain, pBlockAmb);
                value += (1 - pCanBlock) * 4;
            }
        }

        if (actionObj.action === 'assassinate' && actionObj.targetId !== undefined) {
            const targetIdx = this._coerceToIndex(gameState, actionObj.targetId);
            if (targetIdx !== -1) {
                const pContessa = this.belief.probHas(targetIdx, 'contessa');
                value += (1 - pContessa) * 6;
            }
        }

        return value;
    }

    isWinningMove(actionObj, gameState) {
        if (actionObj.action !== 'coup') return false;
        if (actionObj.targetId === undefined || actionObj.targetId === null) return false;

        const alive = gameState.players.filter(p => !p.eliminated);
        if (alive.length !== 2) return false;  // Not down to 2 players

        const targetIdx = this._coerceToIndex(gameState, actionObj.targetId);
        const target = targetIdx !== -1 ? gameState.players[targetIdx] : null;
        const targetInfluences = target ? target.cards.filter(c => !c.revealed).length : 0;
        return target && !target.eliminated && targetInfluences === 1;
    }

    isSuicideMove(actionObj, player, gameState, gameHistory) {
        // Only challengeable actions can lead to suicide via failed challenge
        if (!this.isChallengeable(actionObj.action)) {
            return false;
        }

        const required = this.getRequiredCard(actionObj.action);
        if (!required) return false;

        // Check if we actually have the required card
        const hasCard = this.playerHasCard(player, required);

        // If we have the card, we can't lose a challenge
        if (hasCard) return false;

        // We're bluffing - check if we only have 1 influence left
        const myInfluences = player.cards.filter(c => !c.revealed).length;
        if (myInfluences !== 1) return false;  // Not suicide if we have 2+ influences

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

    estimateChallengeProbability(required, actionObj, player, gameState, gameHistory) {
        // Count how many copies are revealed
        const revealed = gameHistory?.revealedCards;
        let revealedCount = 0;

        if (Array.isArray(revealed)) {
            for (const arr of revealed) {
                if (!Array.isArray(arr)) continue;
                for (const c of arr) {
                    if (c === required) revealedCount++;
                }
            }
        }

        // Base challenge rates increase as more cards are revealed
        const baseChallengeRates = {
            0: 0.25,  // No cards revealed - normal bluffing
            1: 0.35,  // 1 revealed - slightly higher
            2: 0.70,  // 2 revealed - very likely to be challenged
            3: 1.00   // All revealed - certain challenge
        };

        let challengeProb = baseChallengeRates[revealedCount] || 0.30;

        // SPECIAL CASE: Assassinating a 1-influence player who will always challenge bluffs
        if (actionObj.action === 'assassinate' && actionObj.targetId !== undefined) {
            const targetIdx = this._coerceToIndex(gameState, actionObj.targetId);
            const target = targetIdx !== -1 ? gameState.players[targetIdx] : null;

            if (target) {
                const targetInfluences = target.cards.filter(c => !c.revealed).length;
                const hasCard = this.playerHasCard(player, required);

                if (targetInfluences === 1 && !hasCard) {
                    // Target has nothing to lose - will always challenge a bluff
                    challengeProb = 1.00;  // 100% challenge rate
                }
            }
        }

        return challengeProb;
    }

    getBaseActionValue(action, player, gameState) {
        const values = {
            income: 1,
            'foreign-aid': 2,
            tax: 3,
            coup: 30,
            assassinate: 40,
            exchange: 15
        };

        if (action.action === 'steal' && action.targetId !== undefined) {
            const tIdx = this._coerceToIndex(gameState, action.targetId);
            const target = tIdx !== -1 ? gameState.players[tIdx] : null;
            return target ? Math.min(2, target.coins) * 2 : 0;
        }

        return values[action.action] || 0;
    }

    strategicModifiers(action, player, gameState) {
        let modifier = 0;
        const active = gameState.players.filter(p => p && !p.eliminated);
        const avgCoins = active.reduce((s, p) => s + p.coins, 0) / Math.max(1, active.length);

        if (player.coins > avgCoins && ['coup', 'assassinate', 'steal'].includes(action.action)) modifier += 5;
        if (player.coins < avgCoins && ['income', 'foreign-aid', 'tax'].includes(action.action)) modifier += 3;

        return modifier;
    }

    chooseBestCoupTarget(player, gameState) {
        const targets = gameState.players
            .filter(p => p && p.id !== player.id && !p.eliminated);

        if (targets.length === 0) return { action: 'income' };

        targets.forEach(t => (t._random = Math.random()));
        targets.sort((a, b) => {
            const cd = b.coins - a.coins;
            return cd !== 0 ? cd : (b._random - a._random);
        });

        return { action: 'coup', targetId: targets[0].id };
    }

    getPossibleActions(player, gameState) {
        const actions = [];
        const targets = gameState.players.filter(p => p && p.id !== player.id && !p.eliminated);

        actions.push({ action: 'income' });
        actions.push({ action: 'foreign-aid' });
        actions.push({ action: 'tax' });
        actions.push({ action: 'exchange' });

        if (player.coins >= 3 && targets.length > 0) {
            targets.forEach(t => actions.push({ action: 'assassinate', targetId: t.id }));
        }

        if (targets.length > 0) {
            targets.forEach(t => actions.push({ action: 'steal', targetId: t.id }));
        }

        if (player.coins >= 7 && targets.length > 0) {
            targets.forEach(t => actions.push({ action: 'coup', targetId: t.id }));
        }

        return actions;
    }

    playerHasCard(player, cardType) {
        return player.cards.some(c => !c.revealed && c.character === cardType);
    }

    isChallengeable(action) {
        return ['tax', 'assassinate', 'steal', 'exchange'].includes(action);
    }

    getRequiredCard(action) {
        const map = { tax: 'duke', assassinate: 'assassin', steal: 'captain', exchange: 'ambassador' };
        return map[action] || null;
    }

    // --- Engine hooks (challenge / block decisions) ---

    decideChallengeAction(action, claimantId, gameState, gameHistory) {
        this.belief.sync(gameState, gameHistory);

        const player = gameState.players[this.playerId];
        const myInfluences = player.cards.filter(c => !c.revealed).length;

        // SPECIAL CASE: Being assassinated on last influence - must act or die
        // Only applies when WE are the actual target of the assassination.
        if (action === 'assassinate' && myInfluences === 1 &&
                gameState.pendingAction?.targetId === this.playerId) {
            const hasContessa = player.cards.some(c => !c.revealed && c.character === 'contessa');

            if (hasContessa) {
                // Don't challenge - we'll block instead (handled in decideBlockAction)
                return false;
            } else {
                // No Contessa, must challenge (any chance > 0% survival)
                return true;  // Always challenge if we can't block
            }
        }

        // SPECIAL CASE: Being stolen from on last influence
        // Only applies when WE are the actual target of the steal.
        if (action === 'steal' && myInfluences === 1 &&
                gameState.pendingAction?.targetId === this.playerId) {
            const hasCaptain = player.cards.some(c => !c.revealed && c.character === 'captain');
            const hasAmb = player.cards.some(c => !c.revealed && c.character === 'ambassador');

            if (hasCaptain || hasAmb) {
                return false;  // Will block instead
            }
            // For steal, it's not as critical as assassinate, so continue to normal logic
        }

        const required = this.getRequiredCard(action);
        if (!required) return false;

        const claimantIdx = this._coerceToIndex(gameState, claimantId);
        if (claimantIdx === -1) return false;

        // If all 3 copies are publicly revealed, impossible => always challenge
        const revealed = gameHistory?.revealedCards;
        let revealedCount = 0;
        if (Array.isArray(revealed)) {
            for (const arr of revealed) {
                if (!Array.isArray(arr)) continue;
                for (const c of arr) if (c === required) revealedCount++;
            }
        }
        if (revealedCount >= 3) return true;
        if (revealedCount === 2) return Math.random() < 0.55;

        const pHas = this.belief.probHas(claimantIdx, required);

        // Danger-aware threshold (clamped)
        let danger = 0;
        if (action === 'assassinate') danger += 0.10;
        if (action === 'steal') danger += 0.06;

        // If we only have 1 influence left, be more conservative about challenging
        if (myInfluences === 1) danger -= 0.10;

        let threshold = 0.30 - danger;
        threshold = Math.max(0.05, Math.min(0.90, threshold));

        return pHas < threshold;
    }

    decideBlockAction(action, actorId, gameState, gameHistory) {
        this.belief.sync(gameState, gameHistory);

        const player = gameState.players[this.playerId];
        const myInfluences = player.cards.filter(c => !c.revealed).length;

        // Handle assassination: blockable by Contessa at any influence count
        if (action === 'assassinate') {
            const hasContessa = player.cards.some(c => !c.revealed && c.character === 'contessa');

            if (hasContessa) {
                // Last influence: life-or-death, always block
                if (myInfluences === 1) return true;
                // 2 influences: block with high probability to preserve both cards
                return Math.random() < 0.88;
            } else if (myInfluences === 1) {
                // No Contessa on last influence: bluff-block as alternative to challenging
                return Math.random() < 0.50;
            }
            return false;
        }

        // Handle steal: blockable by Captain OR Ambassador
        if (action === 'steal') {
            const hasCaptain = player.cards.some(c => !c.revealed && c.character === 'captain');
            const hasAmb = player.cards.some(c => !c.revealed && c.character === 'ambassador');
            if (hasCaptain || hasAmb) return Math.random() < 0.88;

            // Bluff block sometimes
            const actorIdx = this._coerceToIndex(gameState, actorId);
            const pActorCaptain = actorIdx !== -1 ? this.belief.probHas(actorIdx, 'captain') : 0.5;
            const bluffChance = (1 - pActorCaptain) * 0.12;
            return Math.random() < bluffChance;
        }

        // Handle foreign-aid (duke block)
        if (action === 'foreign-aid') {
            const hasDuke = player.cards.some(c => !c.revealed && c.character === 'duke');
            if (hasDuke) return Math.random() < 0.88;
            return false;  // Don't bluff block foreign-aid usually
        }

        return false;
    }

    decideBlockClaim(action, actorId, gameState, gameHistory) {
        const player = gameState.players[this.playerId];

        if (action === 'foreign-aid') return 'duke';
        if (action === 'assassinate') return 'contessa';

        if (action === 'steal') {
            const hasCaptain = player.cards.some(c => !c.revealed && c.character === 'captain');
            const hasAmb = player.cards.some(c => !c.revealed && c.character === 'ambassador');
            if (hasCaptain) return 'captain';
            if (hasAmb) return 'ambassador';
            return Math.random() < 0.5 ? 'captain' : 'ambassador';
        }

        return null;
    }

    decideChallengeBlock(action, blockerId, blockChar, gameState, gameHistory) {
        this.belief.sync(gameState, gameHistory);

        const player = gameState.players[this.playerId];
        const myInfluences = player.cards.filter(c => !c.revealed).length;

        const blockerIdx = this._coerceToIndex(gameState, blockerId);
        if (blockerIdx === -1) return false;

        const revealed = gameHistory?.revealedCards;
        let revealedCount = 0;
        if (Array.isArray(revealed)) {
            for (const arr of revealed) {
                if (!Array.isArray(arr)) continue;
                for (const c of arr) if (c === blockChar) revealedCount++;
            }
        }
        if (revealedCount >= 3) return true;
        if (revealedCount === 2) return Math.random() < 0.55;

        const pHas = this.belief.probHas(blockerIdx, blockChar);

        let danger = 0;
        if (action === 'assassinate') danger += 0.10;
        if (action === 'steal') danger += 0.06;
        if (myInfluences === 1) danger -= 0.10;

        let threshold = 0.30 - danger;
        threshold = Math.max(0.05, Math.min(0.90, threshold));

        return pHas < threshold;
    }

    // --- Local helper (id/index robust) ---

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
