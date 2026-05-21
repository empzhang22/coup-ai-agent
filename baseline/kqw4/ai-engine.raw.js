// AIEngine Base Class - Abstract base class for all AI implementations
class AIEngine {
    constructor(playerId, aiType) {
        this.playerId = playerId;
        this.aiType = aiType;
    }

    chooseAction(player, gameState, gameHistory) {
        throw new Error('Must implement chooseAction');
    }
}
