// ProbabilityCalculator - Calculates card probabilities based on game state and history
// Note: CHARACTERS constant is defined in index.html

class ProbabilityCalculator {
    constructor(gameHistory, gameState, calculatingPlayerId) {
        this.gameHistory = gameHistory;
        this.gameState = gameState;
        this.calculatingPlayerId = calculatingPlayerId;
    }

    calculateCardProbabilities(targetPlayerId) {
        const target = this.gameState?.players?.[targetPlayerId];
        if (!target) {
            // Defensive fallback
            const base = 1 / CHARACTERS.length;
            const out = {};
            CHARACTERS.forEach(char => (out[char] = base));
            return out;
        }
        const unseenCards = this.getUnseenCards();
        const targetActiveCards = target.cards.filter(c => !c.revealed).length;

        const cardCounts = {};
        CHARACTERS.forEach(char => {
            cardCounts[char] = unseenCards.filter(c => c === char).length;
        });

        const probs = {};
        CHARACTERS.forEach(char => {
            if (targetActiveCards === 0 || cardCounts[char] === 0) {
                probs[char] = 0;
            } else {
                probs[char] = 1 - this.hypergeometricProb(
                    unseenCards.length,
                    cardCounts[char],
                    targetActiveCards,
                    0
                );
            }
        });

        this.updateProbsFromActionHistory(probs, targetPlayerId);

        // IMPORTANT: These are marginal probabilities of "player has >=1 copy" of each character.
        // They are NOT mutually exclusive events, so do NOT normalize them to sum to 1.
        // Also clamp to [0, 1] because the action-history heuristic can boost values.
        for (const char of Object.keys(probs)) {
            const v = probs[char];
            probs[char] = Math.max(0, Math.min(1, Number.isFinite(v) ? v : 0));
        }

        return probs;
    }

    hypergeometricProb(N, K, n, k) {
        if (k > n || k > K || n - k > N - K) return 0;
        return this.combination(K, k) * this.combination(N - K, n - k) / this.combination(N, n);
    }

    combination(n, k) {
        if (k > n || k < 0) return 0;
        if (k === 0 || k === n) return 1;
        let result = 1;
        for (let i = 0; i < k; i++) {
            result *= (n - i) / (i + 1);
        }
        return result;
    }

    updateProbsFromActionHistory(probs, targetPlayerId) {
        const hist = this.gameHistory?.actionHistory;
        const targetActions = Array.isArray(hist)
            ? hist.filter(a => a && a.playerId === targetPlayerId)
            : [];

        const actionCardMap = {
            'tax': 'duke',
            'assassinate': 'assassin',
            'steal': 'captain',
            'exchange': 'ambassador'
        };

        targetActions.forEach(action => {
            const likelyCard = actionCardMap[action.action];
            if (!likelyCard) return;

            if (action.challenged && action.success) {
                probs[likelyCard] = Math.min(1, probs[likelyCard] * 3.0);
            } else if (action.challenged && !action.success) {
                probs[likelyCard] = 0;
            } else {
                probs[likelyCard] = Math.min(1, probs[likelyCard] * 1.5);
            }
        });
    }

    getUnseenCards() {
        const allCards = [];
        CHARACTERS.forEach(char => {
            for (let i = 0; i < 3; i++) allCards.push(char);
        });

        // Remove revealed cards (these are already tracked in history)
        const revealed = this.gameHistory?.revealedCards;
        if (Array.isArray(revealed)) {
            revealed.forEach(cards => {
                if (!Array.isArray(cards)) return;
                cards.forEach(c => {
                    const idx = allCards.indexOf(c);
                    if (idx !== -1) allCards.splice(idx, 1);
                });
            });
        }

        // Remove the calculating player's own UNREVEALED cards (they know what they have)
        // Don't remove revealed cards again - they're already removed above
        if (this.calculatingPlayerId !== undefined) {
            const calculatingPlayer = this.gameState?.players?.[this.calculatingPlayerId];
            if (calculatingPlayer) {
                calculatingPlayer.cards.forEach(card => {
                    if (!card.revealed) {
                        const idx = allCards.indexOf(card.character);
                        if (idx !== -1) allCards.splice(idx, 1);
                    }
                });
            }
        }

        return allCards;
    }
}
