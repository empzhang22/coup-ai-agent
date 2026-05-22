// AIEngine Base Class - Abstract base class for all AI implementations
class AIEngine {
    constructor(playerId, aiType) {
        this.playerId = playerId;
        this.aiType = aiType;
    }

    chooseAction(player, gameState, gameHistory) {
        throw new Error('Must implement chooseAction');
    }

    // Called once per game when a winner is known.
    // Override in AI implementations that train online (e.g. NeuralAI).
    // won: true if this player won the game.
    onGameEnd(won) {}

    // Returns the index into player.cards of the card to reveal when losing influence.
    // Default: reveal the least-valuable unrevealed card by a fixed priority heuristic.
    chooseCardToReveal(player, gameState, gameHistory) {
        const CARD_PRIORITY = { assassin: 1, ambassador: 2, captain: 3, duke: 4, contessa: 5 };
        const unrevealed = player.cards
            .map((c, i) => ({ character: c.character, index: i, revealed: c.revealed }))
            .filter(c => !c.revealed);
        if (unrevealed.length === 0) return -1;
        if (unrevealed.length === 1) return unrevealed[0].index;
        unrevealed.sort((a, b) => (CARD_PRIORITY[a.character] || 0) - (CARD_PRIORITY[b.character] || 0));
        return unrevealed[0].index;
    }
}
