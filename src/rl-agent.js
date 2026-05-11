"use strict";

const fs = require("fs");
const { ACTIONS, indexToAction } = require("./constants");
const { ACTION_COUNT, OBSERVATION_SIZE, legalActionIndexes } = require("./encoder");
const { createRng } = require("./rng");

class LinearPolicyAgent {
  constructor({
    observationSize = OBSERVATION_SIZE,
    actionCount = ACTION_COUNT,
    learningRate = 0.015,
    valueLearningRate = 0.02,
    entropyBonus = 0.003,
    gamma = 0.98,
    seed = 1
  } = {}) {
    this.observationSize = observationSize;
    this.actionCount = actionCount;
    this.learningRate = learningRate;
    this.valueLearningRate = valueLearningRate;
    this.entropyBonus = entropyBonus;
    this.gamma = gamma;
    this.rng = createRng(seed);

    this.weights = Array.from({ length: actionCount }, () =>
      Array.from({ length: observationSize }, () => (this.rng() - 0.5) * 0.02)
    );
    this.valueWeights = Array.from({ length: observationSize }, () => 0);
  }

  act(observationVector, legalMask, { greedy = false } = {}) {
    const probabilities = this.policy(observationVector, legalMask);
    const legal = legalActionIndexes(legalMask);
    if (legal.length === 0) throw new Error("No legal actions available");

    let actionIndex = legal[0];
    if (greedy) {
      let best = -Infinity;
      for (const index of legal) {
        if (probabilities[index] > best) {
          best = probabilities[index];
          actionIndex = index;
        }
      }
    } else {
      const roll = this.rng();
      let cumulative = 0;
      for (const index of legal) {
        cumulative += probabilities[index];
        if (roll <= cumulative) {
          actionIndex = index;
          break;
        }
      }
    }

    return {
      actionIndex,
      action: indexToAction(actionIndex),
      probabilities
    };
  }

  policy(observationVector, legalMask) {
    const legal = legalActionIndexes(legalMask);
    const logits = Array(this.actionCount).fill(-Infinity);
    let max = -Infinity;

    for (const actionIndex of legal) {
      const logit = dot(this.weights[actionIndex], observationVector);
      logits[actionIndex] = logit;
      if (logit > max) max = logit;
    }

    let sum = 0;
    const probs = Array(this.actionCount).fill(0);
    for (const actionIndex of legal) {
      const value = Math.exp(logits[actionIndex] - max);
      probs[actionIndex] = value;
      sum += value;
    }
    for (const actionIndex of legal) probs[actionIndex] /= sum || 1;
    return probs;
  }

  value(observationVector) {
    return dot(this.valueWeights, observationVector);
  }

  update(trajectories) {
    const returnsByPlayer = new Map();
    for (let i = trajectories.length - 1; i >= 0; i--) {
      const step = trajectories[i];
      const nextReturn = returnsByPlayer.get(step.playerId) || 0;
      const discounted = step.reward + this.gamma * nextReturn;
      step.return = discounted;
      returnsByPlayer.set(step.playerId, discounted);
    }

    let policyLoss = 0;
    let valueLoss = 0;

    for (const step of trajectories) {
      const probs = this.policy(step.observation, step.legalMask);
      const value = this.value(step.observation);
      const advantage = clamp(step.return - value, -3, 3);
      const legal = legalActionIndexes(step.legalMask);

      policyLoss += -Math.log(Math.max(1e-9, probs[step.actionIndex])) * advantage;
      valueLoss += 0.5 * (step.return - value) ** 2;

      for (const actionIndex of legal) {
        const indicator = actionIndex === step.actionIndex ? 1 : 0;
        const entropyPush = this.entropyBonus * -Math.log(Math.max(1e-9, probs[actionIndex]));
        const scale = this.learningRate * ((indicator - probs[actionIndex]) * advantage + entropyPush);
        addScaled(this.weights[actionIndex], step.observation, scale);
      }

      addScaled(this.valueWeights, step.observation, this.valueLearningRate * (step.return - value));
    }

    return {
      transitions: trajectories.length,
      policyLoss: policyLoss / Math.max(1, trajectories.length),
      valueLoss: valueLoss / Math.max(1, trajectories.length)
    };
  }

  save(path) {
    fs.writeFileSync(path, JSON.stringify(this.toJSON(), null, 2));
  }

  toJSON() {
    return {
      type: "LinearPolicyAgent",
      observationSize: this.observationSize,
      actionCount: this.actionCount,
      learningRate: this.learningRate,
      valueLearningRate: this.valueLearningRate,
      entropyBonus: this.entropyBonus,
      gamma: this.gamma,
      actions: ACTIONS,
      weights: this.weights,
      valueWeights: this.valueWeights
    };
  }

  static load(path) {
    const data = JSON.parse(fs.readFileSync(path, "utf8"));
    const agent = new LinearPolicyAgent(data);
    agent.weights = data.weights;
    agent.valueWeights = data.valueWeights;
    return agent;
  }
}

function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

function addScaled(target, source, scale) {
  for (let i = 0; i < target.length; i++) target[i] += source[i] * scale;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

module.exports = { LinearPolicyAgent };
