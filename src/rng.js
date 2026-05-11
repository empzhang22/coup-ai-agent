"use strict";

function createRng(seed = Date.now()) {
  let state = seed >>> 0;
  return function rng() {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function choice(items, rng = Math.random) {
  if (!items.length) return undefined;
  return items[Math.floor(rng() * items.length)];
}

function shuffle(items, rng = Math.random) {
  for (let i = items.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [items[i], items[j]] = [items[j], items[i]];
  }
  return items;
}

module.exports = { choice, createRng, shuffle };
