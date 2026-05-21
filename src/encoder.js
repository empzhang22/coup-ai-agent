"use strict";

const { ACTIONS, CHARACTERS, MAX_PLAYERS } = require("./constants");

const DECISION_TYPES = ["main", "challenge", "block"];
const OBSERVATION_SIZE =
  DECISION_TYPES.length +
  MAX_PLAYERS +
  MAX_PLAYERS +
  MAX_PLAYERS * (5 + CHARACTERS.length) +
  CHARACTERS.length +
  CHARACTERS.length +
  1;

function encodeObservation(observation) {
  const vector = [];
  const decisionType = observation.decision?.type || "main";

  for (const type of DECISION_TYPES) vector.push(type === decisionType ? 1 : 0);
  oneHotInto(vector, observation.currentPlayerIndex, MAX_PLAYERS);
  oneHotInto(vector, observation.selfId, MAX_PLAYERS);

  for (let i = 0; i < MAX_PLAYERS; i++) {
    const player = observation.players[i];
    if (!player) {
      vector.push(...Array(5 + CHARACTERS.length).fill(0));
      continue;
    }

    vector.push(
      clamp(player.coins / 12),
      clamp(player.influence / 2),
      player.eliminated ? 1 : 0,
      player.id === observation.selfId ? 1 : 0,
      player.id === observation.currentPlayerIndex ? 1 : 0
    );
    for (const character of CHARACTERS) {
      vector.push((player.revealed.filter(c => c === character).length || 0) / 3);
    }
  }

  const ownCounts = Object.fromEntries(CHARACTERS.map(c => [c, 0]));
  const self = observation.players[observation.selfId];
  for (const card of self?.ownCards || []) ownCounts[card]++;
  for (const character of CHARACTERS) vector.push(ownCounts[character] / 2);

  for (const character of CHARACTERS) {
    vector.push((observation.revealedCounts[character] || 0) / 3);
  }

  vector.push(clamp(observation.turnCount / 100));

  if (vector.length !== OBSERVATION_SIZE) {
    throw new Error(`Observation encoder produced ${vector.length}, expected ${OBSERVATION_SIZE}`);
  }
  return vector;
}

function encodeLegalMask(env, playerId = env.currentPlayerId()) {
  return env.legalActionMask(playerId);
}

function legalActionIndexes(mask) {
  const indexes = [];
  for (let i = 0; i < mask.length; i++) {
    if (mask[i]) indexes.push(i);
  }
  return indexes;
}

function oneHotInto(vector, index, size) {
  for (let i = 0; i < size; i++) vector.push(i === index ? 1 : 0);
}

function clamp(value) {
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
}

module.exports = {
  ACTION_COUNT: ACTIONS.length,
  OBSERVATION_SIZE,
  encodeLegalMask,
  encodeObservation,
  legalActionIndexes
};
