"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { CoupEnv } = require("../src/coup-env");
const { RandomMaskedAgent } = require("../src/baseline-agents");
const { encodeLegalMask, encodeObservation, OBSERVATION_SIZE } = require("../src/encoder");
const { NeuralPPOAgent } = require("../src/neural-ppo-agent");

test("observation encoder returns fixed-size vectors", () => {
  const env = new CoupEnv({ playerCount: 3, seed: 42 });
  const playerId = env.currentPlayerId();
  const vector = encodeObservation(env.observe(playerId));
  assert.equal(vector.length, OBSERVATION_SIZE);
});

test("forced coup is the only main action at 10 coins", () => {
  const env = new CoupEnv({ playerCount: 3, seed: 43 });
  const playerId = env.currentPlayerId();
  env.players[playerId].coins = 10;
  const legal = env.legalActions(playerId);
  assert.ok(legal.length > 0);
  assert.ok(legal.every(action => action.startsWith("coup:")));
});

test("random masked agents can finish many games without illegal actions", () => {
  const games = 25;
  const playerCount = 3;
  const agents = Array.from({ length: playerCount }, (_, id) => new RandomMaskedAgent({ seed: 200 + id }));

  for (let game = 0; game < games; game++) {
    const env = new CoupEnv({ playerCount, seed: 1000 + game });
    let safety = 10000;

    while (!env.gameOver && safety-- > 0) {
      const playerId = env.currentPlayerId();
      const observation = env.observe(playerId);
      const action = agents[playerId].act(encodeObservation(observation), encodeLegalMask(env, playerId));
      env.step(action.actionIndex);
    }

    assert.ok(safety > 0, `game ${game} exceeded safety limit`);
    assert.notEqual(env.winnerId(), null);
  }
});

test("neural PPO agent only selects legal masked actions", () => {
  const env = new CoupEnv({ playerCount: 3, seed: 44 });
  const playerId = env.currentPlayerId();
  const observation = env.observe(playerId);
  const legalMask = encodeLegalMask(env, playerId);
  const agent = new NeuralPPOAgent({ seed: 99, hiddenSize: 16 });

  for (let i = 0; i < 20; i++) {
    const action = agent.act(encodeObservation(observation), legalMask);
    assert.equal(legalMask[action.actionIndex], true);
    assert.equal(typeof action.logProb, "number");
    assert.equal(typeof action.value, "number");
  }
});
