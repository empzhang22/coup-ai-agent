"use strict";

const { actionToIndex } = require("./constants");
const { createRng } = require("./rng");

class RandomMaskedAgent {
  constructor({ seed = 10 } = {}) {
    this.rng = createRng(seed);
  }

  act(_observationVector, legalMask) {
    const legal = indexes(legalMask);
    const actionIndex = legal[Math.floor(this.rng() * legal.length)];
    return { actionIndex };
  }
}

class HeuristicAgent {
  constructor({ seed = 20 } = {}) {
    this.rng = createRng(seed);
  }

  act(_observationVector, legalMask, { observation } = {}) {
    const legal = new Set(indexes(legalMask));
    const decision = observation?.decision?.type;
    const self = observation?.players?.[observation.selfId];

    if (decision === "reveal") {
      const ownCards = self?.ownCards || [];
      const revealSecond = ownCards[0] === "duke" || ownCards[0] === "captain";
      return this.firstLegal(legalMask, revealSecond ? ["reveal:1", "reveal:0"] : ["reveal:0", "reveal:1"]);
    }

    if (decision === "challenge") {
      return this.firstLegal(legalMask, this.rng() < 0.12 ? ["challenge", "pass"] : ["pass", "challenge"]);
    }

    if (decision === "block") {
      const ownCards = new Set(self?.ownCards || []);
      for (const card of ["duke", "contessa", "captain", "ambassador"]) {
        if (ownCards.has(card) && legal.has(actionToIndex(`block:${card}`))) {
          return { actionIndex: actionToIndex(`block:${card}`) };
        }
      }
      return this.firstLegal(legalMask, this.rng() < 0.08
        ? ["block:duke", "block:contessa", "block:captain", "block:ambassador", "pass"]
        : ["pass"]);
    }

    const targets = (observation?.players || []).filter(p => p.id !== observation.selfId && !p.eliminated);
    const weakest = targets.sort((a, b) => a.influence - b.influence || b.coins - a.coins)[0];
    const targetId = weakest?.id ?? 0;
    const ownCards = new Set(self?.ownCards || []);

    const preferences = [];
    if ((self?.coins || 0) >= 7) preferences.push(`coup:${targetId}`);
    if (ownCards.has("assassin") && (self?.coins || 0) >= 3) preferences.push(`assassinate:${targetId}`);
    if (ownCards.has("duke")) preferences.push("tax");
    if (ownCards.has("captain")) preferences.push(`steal:${targetId}`);
    if (ownCards.has("ambassador")) preferences.push("exchange");
    preferences.push("tax", `steal:${targetId}`, "foreign-aid", "income");

    return this.firstLegal(legalMask, preferences);
  }

  firstLegal(legalMask, preferences) {
    for (const action of preferences) {
      const actionIndex = actionToIndex(action);
      if (legalMask[actionIndex]) return { actionIndex };
    }
    const legal = indexes(legalMask);
    return { actionIndex: legal[Math.floor(this.rng() * legal.length)] };
  }
}

function indexes(mask) {
  const out = [];
  for (let i = 0; i < mask.length; i++) if (mask[i]) out.push(i);
  return out;
}

module.exports = { HeuristicAgent, RandomMaskedAgent };
