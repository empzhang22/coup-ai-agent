// Browser inference wrapper for PPO models exported by scripts/export-browser-model.js.
// Defines global: RLAI

class RLAI extends AIEngine {
    constructor(playerId, aiType) {
        super(playerId, aiType);
        this.model = window.COUP_RL_MODEL || null;
        this.pendingBlockClaim = null;
    }

    chooseAction(player, gameState, gameHistory) {
        const action = this.chooseMaskedAction("main", player.id, gameState, gameHistory);
        if (action.includes(":")) {
            const [name, target] = action.split(":");
            return { action: name, targetId: Number(target) };
        }
        return { action };
    }

    decideChallengeAction(action, claimantId, gameState, gameHistory) {
        const chosen = this.chooseMaskedAction("challenge", this.playerId, gameState, gameHistory, {
            claimantId,
            action
        });
        return chosen === "challenge";
    }

    decideChallengeBlock(action, blockerId, blockChar, gameState, gameHistory) {
        const chosen = this.chooseMaskedAction("challenge_block", this.playerId, gameState, gameHistory, {
            blockerId,
            blockChar,
            action
        });
        return chosen === "challenge";
    }

    decideBlockAction(action, actorId, gameState, gameHistory) {
        const chosen = this.chooseMaskedAction("block", this.playerId, gameState, gameHistory, {
            actorId,
            action
        });
        if (chosen.startsWith("block:")) {
            this.pendingBlockClaim = chosen.split(":")[1];
            return true;
        }
        this.pendingBlockClaim = null;
        return false;
    }

    decideBlockClaim(action, actorId, gameState, gameHistory) {
        if (this.pendingBlockClaim) return this.pendingBlockClaim;
        const active = this.myActiveCards(gameState);
        if (action === "foreign-aid") return "duke";
        if (action === "assassinate") return "contessa";
        if (action === "steal") {
            if (active.includes("captain")) return "captain";
            if (active.includes("ambassador")) return "ambassador";
            return "captain";
        }
        return null;
    }

    chooseMaskedAction(type, playerId, gameState, gameHistory, extra = {}) {
        const legal = this.legalActions(type, playerId, gameState, extra);
        if (!this.model) return this.fallbackAction(type, playerId, gameState, legal);

        const observation = this.encodeObservation(type, playerId, gameState, gameHistory);
        const probs = this.forwardPolicy(observation, legal);
        let bestAction = legal[0];
        let bestProb = -Infinity;
        for (const action of legal) {
            const index = RL_ACTION_INDEX[action];
            if (probs[index] > bestProb) {
                bestProb = probs[index];
                bestAction = action;
            }
        }
        return bestAction;
    }

    legalActions(type, playerId, gameState, extra = {}) {
        const player = gameState.players[playerId];
        if (!player || player.eliminated) return [];

        if (type === "main") {
            const targets = gameState.players.filter(p => p.id !== playerId && !p.eliminated);
            if (player.coins >= 10) return targets.map(t => `coup:${t.id}`);
            const actions = ["income", "foreign-aid", "tax", "exchange"];
            if (player.coins >= 7) actions.push(...targets.map(t => `coup:${t.id}`));
            if (player.coins >= 3) actions.push(...targets.map(t => `assassinate:${t.id}`));
            actions.push(...targets.map(t => `steal:${t.id}`));
            return actions;
        }

        if (type === "challenge" || type === "challenge_block") return ["pass", "challenge"];

        if (type === "block") {
            if (extra.action === "foreign-aid") return ["pass", "block:duke"];
            if (extra.action === "assassinate") return ["pass", "block:contessa"];
            if (extra.action === "steal") return ["pass", "block:captain", "block:ambassador"];
        }

        return ["pass"];
    }

    encodeObservation(type, playerId, gameState, gameHistory) {
        const vector = [];
        for (const decisionType of RL_DECISION_TYPES) vector.push(decisionType === type ? 1 : 0);
        oneHotInto(vector, gameState.currentPlayerIndex, RL_MAX_PLAYERS);
        oneHotInto(vector, playerId, RL_MAX_PLAYERS);

        for (let i = 0; i < RL_MAX_PLAYERS; i++) {
            const player = gameState.players[i];
            if (!player) {
                vector.push(...Array(5 + RL_CHARACTERS.length).fill(0));
                continue;
            }

            const revealed = player.cards.filter(c => c.revealed).map(c => c.character);
            vector.push(
                clamp(player.coins / 12),
                clamp(player.cards.filter(c => !c.revealed).length / 2),
                player.eliminated ? 1 : 0,
                player.id === playerId ? 1 : 0,
                player.id === gameState.currentPlayerIndex ? 1 : 0
            );
            for (const character of RL_CHARACTERS) {
                vector.push(revealed.filter(c => c === character).length / 3);
            }
        }

        const ownCounts = Object.fromEntries(RL_CHARACTERS.map(c => [c, 0]));
        for (const card of this.myActiveCards(gameState)) ownCounts[card]++;
        for (const character of RL_CHARACTERS) vector.push(ownCounts[character] / 2);

        const revealedCounts = Object.fromEntries(RL_CHARACTERS.map(c => [c, 0]));
        if (gameHistory && Array.isArray(gameHistory.revealedCards)) {
            for (const cards of gameHistory.revealedCards) {
                if (!cards) continue;
                for (const card of cards) revealedCounts[card]++;
            }
        }
        for (const character of RL_CHARACTERS) vector.push((revealedCounts[character] || 0) / 3);
        vector.push(clamp((gameState.turnCount || 0) / 100));
        return vector;
    }

    forwardPolicy(input, legalActions) {
        const hidden = this.model.policy.w1.map((row, h) => {
            let value = this.model.policy.b1[h];
            for (let i = 0; i < input.length; i++) value += row[i] * input[i];
            return Math.tanh(value);
        });

        const logits = Array(RL_ACTIONS.length).fill(-Infinity);
        let max = -Infinity;
        for (const action of legalActions) {
            const index = RL_ACTION_INDEX[action];
            let value = this.model.policy.b2[index];
            for (let h = 0; h < hidden.length; h++) value += this.model.policy.w2[index][h] * hidden[h];
            logits[index] = value;
            if (value > max) max = value;
        }

        const probs = Array(RL_ACTIONS.length).fill(0);
        let sum = 0;
        for (const action of legalActions) {
            const index = RL_ACTION_INDEX[action];
            probs[index] = Math.exp(logits[index] - max);
            sum += probs[index];
        }
        for (const action of legalActions) probs[RL_ACTION_INDEX[action]] /= sum || 1;
        return probs;
    }

    fallbackAction(type, playerId, gameState, legal) {
        if (type !== "main") return legal[0];
        const player = gameState.players[playerId];
        const targets = gameState.players.filter(p => p.id !== playerId && !p.eliminated);
        const target = targets.sort((a, b) => a.cards.filter(c => !c.revealed).length - b.cards.filter(c => !c.revealed).length)[0];
        const targetId = target ? target.id : 0;
        const preferred = [];
        if (player.coins >= 7) preferred.push(`coup:${targetId}`);
        preferred.push("tax", `steal:${targetId}`, "foreign-aid", "income");
        return preferred.find(action => legal.includes(action)) || legal[0];
    }

    myActiveCards(gameState) {
        const player = gameState.players[this.playerId];
        return player.cards.filter(c => !c.revealed).map(c => c.character);
    }
}

const RL_CHARACTERS = ["duke", "assassin", "captain", "ambassador", "contessa"];
const RL_MAX_PLAYERS = 5;
const RL_DECISION_TYPES = ["main", "challenge", "block", "challenge_block"];
const RL_ACTIONS = [
    "income",
    "foreign-aid",
    "tax",
    "exchange",
    ...Array.from({ length: RL_MAX_PLAYERS }, (_, i) => `coup:${i}`),
    ...Array.from({ length: RL_MAX_PLAYERS }, (_, i) => `assassinate:${i}`),
    ...Array.from({ length: RL_MAX_PLAYERS }, (_, i) => `steal:${i}`),
    "pass",
    "challenge",
    "block:duke",
    "block:captain",
    "block:ambassador",
    "block:contessa"
];
const RL_ACTION_INDEX = Object.fromEntries(RL_ACTIONS.map((action, index) => [action, index]));

function oneHotInto(vector, index, size) {
    for (let i = 0; i < size; i++) vector.push(i === index ? 1 : 0);
}

function clamp(value) {
    if (value < 0) return 0;
    if (value > 1) return 1;
    return value;
}
