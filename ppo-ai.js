/**
 * Browser inference for trained PPO checkpoints.
 *
 * Loads policy weights from `window.COUP_PPO_MODEL` (see `scripts/export-browser-model.js`
 * and `models/ppo-browser-model.js`). Implements the Coup contest `AIEngine` API used by
 * `aicontest.html`: main actions, challenges, blocks, and block challenges.
 *
 * Training uses the same observation layout in `src/encoder.js` and the same masked
 * softmax policy in `src/neural-ppo-agent.js`. Contest play is greedy (argmax probability).
 *
 * @file ppo-ai.js
 * @global PPOAI
 * @see DESIGN.md
 */

class PPOAI extends AIEngine {
    /**
     * @param {number} playerId - Contest seat index (0 = AI 1, …, 4 = AI 5).
     * @param {string} aiType - Contest type string (e.g. `'ppo'`).
     */
    constructor(playerId, aiType) {
        super(playerId, aiType);
        /** @type {object|null} Exported NeuralPPOAgent JSON from `COUP_PPO_MODEL`. */
        this.model = window.COUP_PPO_MODEL || null;
        /** @type {string|null} Character claimed when blocking (duke, contessa, etc.). */
        this.pendingBlockClaim = null;
    }

    /**
     * Main turn: income, tax, coup, steal, etc.
     * @param {object} player - Contest player object (`id`, `coins`, `cards`, …).
     * @param {object} gameState - Full table state.
     * @param {object} gameHistory - `GameHistory` with `revealedCards`, `actionHistory`, …
     * @returns {{ action: string, targetId?: number }}
     */
    chooseAction(player, gameState, gameHistory) {
        const action = this.chooseMaskedAction("main", player.id, gameState, gameHistory);
        if (action.includes(":")) {
            const [name, target] = action.split(":");
            return { action: name, targetId: Number(target) };
        }
        return { action };
    }

    /**
     * Whether to challenge another player's claimed action (tax, assassinate, etc.).
     * @param {string} action - Claimed action name.
     * @param {number} claimantId - Player who claimed the action.
     * @param {object} gameState
     * @param {object} gameHistory
     * @returns {boolean} `true` to challenge, `false` to pass.
     */
    decideChallengeAction(action, claimantId, gameState, gameHistory) {
        const chosen = this.chooseMaskedAction("challenge", this.playerId, gameState, gameHistory, {
            claimantId,
            action
        });
        return chosen === "challenge";
    }

    /**
     * Whether to challenge an opponent's block (contest block-challenge phase).
     * @param {string} action - Underlying action being blocked (e.g. `foreign-aid`, `steal`).
     * @param {number} blockerId - Player who blocked.
     * @param {string} blockChar - Character used for the block.
     * @param {object} gameState
     * @param {object} gameHistory
     * @returns {boolean} `true` to challenge the block, `false` to pass.
     */
    decideChallengeBlock(action, blockerId, blockChar, gameState, gameHistory) {
        const chosen = this.chooseMaskedAction("challenge_block", this.playerId, gameState, gameHistory, {
            blockerId,
            blockChar,
            action
        });
        return chosen === "challenge";
    }

    /**
     * Whether to attempt a block against the current action.
     * @param {string} action - Action to block.
     * @param {number} actorId - Player performing the action.
     * @param {object} gameState
     * @param {object} gameHistory
     * @returns {boolean} `true` to block, `false` to pass.
     */
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

    /**
     * Character to claim when blocking (must match a hidden card if challenged).
     * Uses policy block choice when set; otherwise falls back to a legal card.
     * @param {string} action
     * @param {number} actorId
     * @param {object} gameState
     * @param {object} gameHistory
     * @returns {string|null} Character name or `null` if not blocking.
     */
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

    /**
     * Core decision step: build observation, run policy, pick highest-probability legal action.
     * @param {string} type - `main` | `challenge` | `block` | `challenge_block`
     * @param {number} playerId - Deciding player (usually `this.playerId`).
     * @param {object} gameState
     * @param {object} gameHistory
     * @param {object} [extra] - Phase context (`claimantId`, `action`, `blockerId`, …).
     * @returns {string} Action id (e.g. `tax`, `coup:2`, `pass`, `block:duke`).
     */
    chooseMaskedAction(type, playerId, gameState, gameHistory, extra = {}) {
        const legal = this.legalActions(type, playerId, gameState, extra);
        if (!this.model) return this.fallbackAction(type, playerId, gameState, legal);

        const observation = this.encodeObservation(type, playerId, gameState, gameHistory);
        const probs = this.forwardPolicy(observation, legal);
        let bestAction = legal[0];
        let bestProb = -Infinity;
        for (const action of legal) {
            const index = PPO_ACTION_INDEX[action];
            if (probs[index] > bestProb) {
                bestProb = probs[index];
                bestAction = action;
            }
        }
        return bestAction;
    }

    /**
     * Legal action strings for the current decision type (contest rules + masks).
     * @param {string} type - Decision phase.
     * @param {number} playerId
     * @param {object} gameState
     * @param {object} extra
     * @returns {string[]}
     */
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

    /**
     * Fixed-size observation vector (length 75) aligned with `src/encoder.js`.
     * @param {string} type - Active decision type.
     * @param {number} playerId - Perspective player (`self` in the vector).
     * @param {object} gameState
     * @param {object} gameHistory
     * @returns {number[]}
     */
    encodeObservation(type, playerId, gameState, gameHistory) {
        const vector = [];
        for (const decisionType of PPO_DECISION_TYPES) vector.push(decisionType === type ? 1 : 0);
        oneHotInto(vector, gameState.currentPlayerIndex, PPO_MAX_PLAYERS);
        oneHotInto(vector, playerId, PPO_MAX_PLAYERS);

        for (let i = 0; i < PPO_MAX_PLAYERS; i++) {
            const player = gameState.players[i];
            if (!player) {
                vector.push(...Array(5 + PPO_CHARACTERS.length).fill(0));
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
            for (const character of PPO_CHARACTERS) {
                vector.push(revealed.filter(c => c === character).length / 3);
            }
        }

        const ownCounts = Object.fromEntries(PPO_CHARACTERS.map(c => [c, 0]));
        for (const card of this.myActiveCards(gameState)) ownCounts[card]++;
        for (const character of PPO_CHARACTERS) vector.push(ownCounts[character] / 2);

        const revealedCounts = Object.fromEntries(PPO_CHARACTERS.map(c => [c, 0]));
        if (gameHistory && Array.isArray(gameHistory.revealedCards)) {
            for (const cards of gameHistory.revealedCards) {
                if (!cards) continue;
                for (const card of cards) revealedCounts[card]++;
            }
        }
        for (const character of PPO_CHARACTERS) vector.push((revealedCounts[character] || 0) / 3);
        vector.push(clamp((gameState.turnCount || 0) / 100));
        return vector;
    }

    /**
     * Two-layer policy forward pass using exported weights (`policy.w1`, `b1`, `w2`, `b2`).
     * Masked softmax over `legalActions` only.
     * @param {number[]} input - Observation vector.
     * @param {string[]} legalActions - Legal action ids for this decision.
     * @returns {number[]} Probability over all `PPO_ACTIONS` indices (illegal = 0).
     */
    forwardPolicy(input, legalActions) {
        const hidden = this.model.policy.w1.map((row, h) => {
            let value = this.model.policy.b1[h];
            for (let i = 0; i < input.length; i++) value += row[i] * input[i];
            return Math.tanh(value);
        });

        const logits = Array(PPO_ACTIONS.length).fill(-Infinity);
        let max = -Infinity;
        for (const action of legalActions) {
            const index = PPO_ACTION_INDEX[action];
            let value = this.model.policy.b2[index];
            for (let h = 0; h < hidden.length; h++) value += this.model.policy.w2[index][h] * hidden[h];
            logits[index] = value;
            if (value > max) max = value;
        }

        const probs = Array(PPO_ACTIONS.length).fill(0);
        let sum = 0;
        for (const action of legalActions) {
            const index = PPO_ACTION_INDEX[action];
            probs[index] = Math.exp(logits[index] - max);
            sum += probs[index];
        }
        for (const action of legalActions) probs[PPO_ACTION_INDEX[action]] /= sum || 1;
        return probs;
    }

    /**
     * Heuristic when `COUP_PPO_MODEL` is missing (smoke / misconfigured contest).
     * @param {string} type
     * @param {number} playerId
     * @param {object} gameState
     * @param {string[]} legal
     * @returns {string}
     */
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

    /**
     * Hidden (unrevealed) character names for this agent.
     * @param {object} gameState
     * @returns {string[]}
     */
    myActiveCards(gameState) {
        const player = gameState.players[this.playerId];
        return player.cards.filter(c => !c.revealed).map(c => c.character);
    }
}

/** @type {string[]} */
const PPO_CHARACTERS = ["duke", "assassin", "captain", "ambassador", "contessa"];
/** @type {number} */
const PPO_MAX_PLAYERS = 5;
/** @type {string[]} */
const PPO_DECISION_TYPES = ["main", "challenge", "block", "challenge_block"];
/** @type {string[]} Full action space (must match `src/constants.js` / training). */
const PPO_ACTIONS = [
    "income",
    "foreign-aid",
    "tax",
    "exchange",
    ...Array.from({ length: PPO_MAX_PLAYERS }, (_, i) => `coup:${i}`),
    ...Array.from({ length: PPO_MAX_PLAYERS }, (_, i) => `assassinate:${i}`),
    ...Array.from({ length: PPO_MAX_PLAYERS }, (_, i) => `steal:${i}`),
    "pass",
    "challenge",
    "block:duke",
    "block:captain",
    "block:ambassador",
    "block:contessa"
];
/** @type {Record<string, number>} */
const PPO_ACTION_INDEX = Object.fromEntries(PPO_ACTIONS.map((action, index) => [action, index]));

/**
 * Append a one-hot segment to `vector`.
 * @param {number[]} vector
 * @param {number} index
 * @param {number} size
 */
function oneHotInto(vector, index, size) {
    for (let i = 0; i < size; i++) vector.push(i === index ? 1 : 0);
}

/**
 * Clamp a numeric feature to [0, 1] for the observation vector.
 * @param {number} value
 * @returns {number}
 */
function clamp(value) {
    if (value < 0) return 0;
    if (value > 1) return 1;
    return value;
}
