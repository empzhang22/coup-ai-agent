// RandomAI - Baseline random implementation
class RandomAI extends AIEngine {
    chooseAction(player, gameState, gameHistory) {
        if (player.coins >= 10) {
            const targets = gameState.players.filter((p, i) => p.id !== player.id && !p.eliminated);
            const target = targets[Math.floor(Math.random() * targets.length)];
            return { action: 'coup', targetId: target.id };
        }

        const possibleActions = ['income', 'foreign-aid', 'tax'];
        if (player.coins >= 7 && Math.random() < 0.3) {
            possibleActions.push('coup');
        }
        if (player.coins >= 3) {
            possibleActions.push('assassinate');
        }
        possibleActions.push('steal', 'exchange');

        const actionName = possibleActions[Math.floor(Math.random() * possibleActions.length)];

        if (['coup', 'assassinate', 'steal'].includes(actionName)) {
            const targets = gameState.players.filter((p, i) => p.id !== player.id && !p.eliminated);
            if (targets.length === 0) return { action: 'income' };
            const target = targets[Math.floor(Math.random() * targets.length)];
            return { action: actionName, targetId: target.id };
        }

        return { action: actionName };
    }

    decideChallengeAction(action, claimantId, gameState, gameHistory) {
        // Find this AI's player object
        const myPlayer = gameState.players.find(p => p.isAI && p.aiEngine === this);
        if (!myPlayer) return Math.random() < 0.2;

        // If being assassinated with only 1 card left and no contessa, always challenge
        if (action.action === 'assassinate' && action.targetId === myPlayer.id) {
            const activeCards = myPlayer.cards.filter(c => !c.revealed);
            if (activeCards.length === 1) {
                const hasContessa = activeCards.some(c => c.role === 'contessa');
                if (!hasContessa) {
                    return true; // Challenge as last resort
                }
            }
        }

        return Math.random() < 0.2;
    }

    decideBlockAction(action, actorId, gameState, gameHistory) {
        // Find this AI's player object
        const myPlayer = gameState.players.find(p => p.isAI && p.aiEngine === this);
        if (!myPlayer) return Math.random() < 0.2;

        const activeCards = myPlayer.cards.filter(c => !c.revealed);

        // Always block assassinate if we have contessa
        if (action === 'assassinate') {
            const hasContessa = activeCards.some(c => c.role === 'contessa');
            if (hasContessa) {
                return true;
            }
        }

        // Always block steal if we have duke or ambassador
        if (action === 'steal') {
            const hasDuke = activeCards.some(c => c.role === 'duke');
            const hasAmbassador = activeCards.some(c => c.role === 'ambassador');
            if (hasDuke || hasAmbassador) {
                return true;
            }
        }

        // Default random behavior for other cases
        const blockRates = { 'foreign-aid': 0.3, 'assassinate': 0.4, 'steal': 0.4 };
        return Math.random() < (blockRates[action] || 0);
    }

    decideBlockClaim(action, actorId, gameState, gameHistory) {
        // Returns which character to claim when blocking an action
        if (action === 'foreign-aid') {
            return 'duke';
        } else if (action === 'assassinate') {
            return 'contessa';
        } else if (action === 'steal') {
            // Use the actual card we have (duke or ambassador) when possible
            const myPlayer = gameState.players.find(p => p.isAI && p.aiEngine === this);
            if (myPlayer) {
                const activeCards = myPlayer.cards.filter(c => !c.revealed);
                const hasDuke = activeCards.some(c => c.role === 'duke');
                const hasAmbassador = activeCards.some(c => c.role === 'ambassador');

                if (hasDuke && !hasAmbassador) {
                    return 'duke';
                } else if (hasAmbassador && !hasDuke) {
                    return 'ambassador';
                } else if (hasDuke && hasAmbassador) {
                    return Math.random() < 0.5 ? 'duke' : 'ambassador';
                }
            }
            // Fallback: randomly choose between captain and ambassador
            return Math.random() < 0.5 ? 'captain' : 'ambassador';
        }
        return null;
    }
}
