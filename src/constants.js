"use strict";

const CHARACTERS = ["duke", "assassin", "captain", "ambassador", "contessa"];
const MAX_PLAYERS = 5;

const ACTIONS = [
  "income",
  "foreign-aid",
  "tax",
  "exchange",
  ...Array.from({ length: MAX_PLAYERS }, (_, i) => `coup:${i}`),
  ...Array.from({ length: MAX_PLAYERS }, (_, i) => `assassinate:${i}`),
  ...Array.from({ length: MAX_PLAYERS }, (_, i) => `steal:${i}`),
  "pass",
  "challenge",
  "block:duke",
  "block:captain",
  "block:ambassador",
  "block:contessa"
];

const ACTION_INDEX = new Map(ACTIONS.map((action, index) => [action, index]));

function actionToIndex(action) {
  const index = ACTION_INDEX.get(action);
  if (index === undefined) throw new Error(`Unknown action: ${action}`);
  return index;
}

function indexToAction(index) {
  const action = ACTIONS[index];
  if (action === undefined) throw new Error(`Unknown action index: ${index}`);
  return action;
}

function requiredCardForAction(actionName) {
  return {
    tax: "duke",
    assassinate: "assassin",
    steal: "captain",
    exchange: "ambassador"
  }[actionName] || null;
}

function blockCardsForAction(actionName) {
  if (actionName === "foreign-aid") return ["duke"];
  if (actionName === "assassinate") return ["contessa"];
  if (actionName === "steal") return ["captain", "ambassador"];
  return [];
}

module.exports = {
  ACTIONS,
  ACTION_INDEX,
  CHARACTERS,
  MAX_PLAYERS,
  actionToIndex,
  blockCardsForAction,
  indexToAction,
  requiredCardForAction
};
