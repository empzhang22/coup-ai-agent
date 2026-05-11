"use strict";

const fs = require("fs");
const { indexToAction } = require("./constants");
const { ACTION_COUNT, OBSERVATION_SIZE, legalActionIndexes } = require("./encoder");
const { createRng } = require("./rng");

class NeuralPPOAgent {
  constructor({
    observationSize = OBSERVATION_SIZE,
    actionCount = ACTION_COUNT,
    hiddenSize = 96,
    policyLearningRate = 0.0008,
    valueLearningRate = 0.0015,
    gamma = 0.99,
    lambda = 0.94,
    clipRatio = 0.2,
    entropyBonus = 0.01,
    valueLossCoef = 0.5,
    maxGradNorm = 1.0,
    seed = 7
  } = {}) {
    this.observationSize = observationSize;
    this.actionCount = actionCount;
    this.hiddenSize = hiddenSize;
    this.policyLearningRate = policyLearningRate;
    this.valueLearningRate = valueLearningRate;
    this.gamma = gamma;
    this.lambda = lambda;
    this.clipRatio = clipRatio;
    this.entropyBonus = entropyBonus;
    this.valueLossCoef = valueLossCoef;
    this.maxGradNorm = maxGradNorm;
    this.rng = createRng(seed);

    this.policy = createNetwork(actionCount, hiddenSize, observationSize, this.rng);
    this.valueNet = createNetwork(1, hiddenSize, observationSize, this.rng);
  }

  act(observationVector, legalMask, { greedy = false } = {}) {
    const { probabilities, logits } = this.forwardPolicy(observationVector, legalMask);
    const legal = legalActionIndexes(legalMask);
    if (!legal.length) throw new Error("No legal actions available");

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
      probabilities,
      logProb: Math.log(Math.max(1e-12, probabilities[actionIndex])),
      value: this.value(observationVector),
      logits
    };
  }

  forwardPolicy(observationVector, legalMask) {
    const forward = forwardNetwork(this.policy, observationVector);
    const legal = legalActionIndexes(legalMask);
    const logits = Array(this.actionCount).fill(-Infinity);
    let max = -Infinity;

    for (const index of legal) {
      logits[index] = forward.output[index];
      if (logits[index] > max) max = logits[index];
    }

    const probabilities = Array(this.actionCount).fill(0);
    let sum = 0;
    for (const index of legal) {
      const value = Math.exp(logits[index] - max);
      probabilities[index] = value;
      sum += value;
    }
    for (const index of legal) probabilities[index] /= sum || 1;
    return { probabilities, logits, cache: forward };
  }

  value(observationVector) {
    return forwardNetwork(this.valueNet, observationVector).output[0];
  }

  update(rollouts, { epochs = 4, batchSize = 128 } = {}) {
    if (!rollouts.length) return { samples: 0, policyLoss: 0, valueLoss: 0, entropy: 0 };
    const samples = prepareAdvantages(rollouts, this.gamma, this.lambda);
    normalizeAdvantages(samples);

    let policyLoss = 0;
    let valueLoss = 0;
    let entropy = 0;
    let seen = 0;

    for (let epoch = 0; epoch < epochs; epoch++) {
      shuffle(samples, this.rng);
      for (let start = 0; start < samples.length; start += batchSize) {
        const batch = samples.slice(start, start + batchSize);
        for (const sample of batch) {
          const policyStats = this.updatePolicySample(sample);
          const valueStats = this.updateValueSample(sample);
          policyLoss += policyStats.loss;
          entropy += policyStats.entropy;
          valueLoss += valueStats.loss;
          seen++;
        }
      }
    }

    return {
      samples: samples.length,
      policyLoss: policyLoss / Math.max(1, seen),
      valueLoss: valueLoss / Math.max(1, seen),
      entropy: entropy / Math.max(1, seen)
    };
  }

  updatePolicySample(sample) {
    const { probabilities, cache } = this.forwardPolicy(sample.observation, sample.legalMask);
    const newLogProb = Math.log(Math.max(1e-12, probabilities[sample.actionIndex]));
    const ratio = Math.exp(newLogProb - sample.oldLogProb);
    const clippedRatio = clamp(ratio, 1 - this.clipRatio, 1 + this.clipRatio);
    const unclipped = ratio * sample.advantage;
    const clipped = clippedRatio * sample.advantage;
    const useGradient =
      (sample.advantage >= 0 && ratio <= 1 + this.clipRatio) ||
      (sample.advantage < 0 && ratio >= 1 - this.clipRatio);

    const gradOutput = Array(this.actionCount).fill(0);
    if (useGradient) {
      const dLossDLogProb = -sample.advantage * ratio;
      for (let i = 0; i < this.actionCount; i++) {
        if (!sample.legalMask[i]) continue;
        const indicator = i === sample.actionIndex ? 1 : 0;
        gradOutput[i] += dLossDLogProb * (indicator - probabilities[i]);
      }
    }

    for (let i = 0; i < this.actionCount; i++) {
      if (!sample.legalMask[i]) continue;
      gradOutput[i] -= this.entropyBonus * entropyLogitGradient(probabilities, i, sample.legalMask);
    }

    applyNetworkGradient(this.policy, cache, gradOutput, this.policyLearningRate, this.maxGradNorm);
    return {
      loss: -Math.min(unclipped, clipped),
      entropy: categoricalEntropy(probabilities, sample.legalMask)
    };
  }

  updateValueSample(sample) {
    const cache = forwardNetwork(this.valueNet, sample.observation);
    const value = cache.output[0];
    const error = value - sample.return;
    const gradOutput = [this.valueLossCoef * error];
    applyNetworkGradient(this.valueNet, cache, gradOutput, this.valueLearningRate, this.maxGradNorm);
    return { loss: 0.5 * error * error };
  }

  snapshot() {
    const agent = new NeuralPPOAgent(this.toJSON());
    agent.policy = cloneNetwork(this.policy);
    agent.valueNet = cloneNetwork(this.valueNet);
    return agent;
  }

  save(path) {
    fs.writeFileSync(path, JSON.stringify(this.toJSON(), null, 2));
  }

  toJSON() {
    return {
      type: "NeuralPPOAgent",
      observationSize: this.observationSize,
      actionCount: this.actionCount,
      hiddenSize: this.hiddenSize,
      policyLearningRate: this.policyLearningRate,
      valueLearningRate: this.valueLearningRate,
      gamma: this.gamma,
      lambda: this.lambda,
      clipRatio: this.clipRatio,
      entropyBonus: this.entropyBonus,
      valueLossCoef: this.valueLossCoef,
      maxGradNorm: this.maxGradNorm,
      policy: this.policy,
      valueNet: this.valueNet
    };
  }

  static load(path) {
    const data = JSON.parse(fs.readFileSync(path, "utf8"));
    const agent = new NeuralPPOAgent(data);
    agent.policy = data.policy;
    agent.valueNet = data.valueNet;
    return agent;
  }
}

function prepareAdvantages(rollouts, gamma, lambda) {
  const byPlayer = new Map();
  for (const sample of rollouts) {
    if (!byPlayer.has(sample.playerId)) byPlayer.set(sample.playerId, []);
    byPlayer.get(sample.playerId).push(sample);
  }

  for (const samples of byPlayer.values()) {
    let gae = 0;
    for (let i = samples.length - 1; i >= 0; i--) {
      const current = samples[i];
      const next = samples[i + 1];
      const nextValue = next ? next.value : 0;
      const delta = current.reward + gamma * nextValue - current.value;
      gae = delta + gamma * lambda * gae;
      current.advantage = gae;
      current.return = current.advantage + current.value;
    }
  }

  return rollouts;
}

function normalizeAdvantages(samples) {
  const mean = samples.reduce((sum, s) => sum + s.advantage, 0) / samples.length;
  const variance = samples.reduce((sum, s) => sum + (s.advantage - mean) ** 2, 0) / samples.length;
  const std = Math.sqrt(variance) || 1;
  for (const sample of samples) sample.advantage = clamp((sample.advantage - mean) / std, -5, 5);
}

function createNetwork(outputSize, hiddenSize, inputSize, rng) {
  const w1Scale = Math.sqrt(2 / inputSize);
  const w2Scale = Math.sqrt(2 / hiddenSize);
  return {
    w1: Array.from({ length: hiddenSize }, () =>
      Array.from({ length: inputSize }, () => randn(rng) * w1Scale)
    ),
    b1: Array(hiddenSize).fill(0),
    w2: Array.from({ length: outputSize }, () =>
      Array.from({ length: hiddenSize }, () => randn(rng) * w2Scale)
    ),
    b2: Array(outputSize).fill(0)
  };
}

function forwardNetwork(net, input) {
  const hiddenPre = Array(net.w1.length).fill(0);
  const hidden = Array(net.w1.length).fill(0);
  for (let h = 0; h < net.w1.length; h++) {
    let value = net.b1[h];
    for (let i = 0; i < input.length; i++) value += net.w1[h][i] * input[i];
    hiddenPre[h] = value;
    hidden[h] = Math.tanh(value);
  }

  const output = Array(net.w2.length).fill(0);
  for (let o = 0; o < net.w2.length; o++) {
    let value = net.b2[o];
    for (let h = 0; h < hidden.length; h++) value += net.w2[o][h] * hidden[h];
    output[o] = value;
  }

  return { input, hiddenPre, hidden, output };
}

function applyNetworkGradient(net, cache, gradOutput, learningRate, maxGradNorm) {
  const gradHidden = Array(cache.hidden.length).fill(0);
  const gradW2 = Array.from({ length: net.w2.length }, () => Array(cache.hidden.length).fill(0));
  const gradB2 = gradOutput.slice();

  for (let o = 0; o < net.w2.length; o++) {
    for (let h = 0; h < cache.hidden.length; h++) {
      gradW2[o][h] = gradOutput[o] * cache.hidden[h];
      gradHidden[h] += gradOutput[o] * net.w2[o][h];
    }
  }

  const gradPre = gradHidden.map((g, h) => g * (1 - Math.tanh(cache.hiddenPre[h]) ** 2));
  const gradW1 = Array.from({ length: net.w1.length }, () => Array(cache.input.length).fill(0));
  const gradB1 = gradPre.slice();

  for (let h = 0; h < net.w1.length; h++) {
    for (let i = 0; i < cache.input.length; i++) {
      gradW1[h][i] = gradPre[h] * cache.input[i];
    }
  }

  const norm = gradientNorm(gradW1, gradB1, gradW2, gradB2);
  const scale = norm > maxGradNorm ? maxGradNorm / norm : 1;
  const lr = learningRate * scale;

  for (let h = 0; h < net.w1.length; h++) {
    net.b1[h] -= lr * gradB1[h];
    for (let i = 0; i < net.w1[h].length; i++) net.w1[h][i] -= lr * gradW1[h][i];
  }
  for (let o = 0; o < net.w2.length; o++) {
    net.b2[o] -= lr * gradB2[o];
    for (let h = 0; h < net.w2[o].length; h++) net.w2[o][h] -= lr * gradW2[o][h];
  }
}

function gradientNorm(...grads) {
  let sum = 0;
  for (const grad of grads) {
    if (Array.isArray(grad[0])) {
      for (const row of grad) for (const value of row) sum += value * value;
    } else {
      for (const value of grad) sum += value * value;
    }
  }
  return Math.sqrt(sum) || 1;
}

function categoricalEntropy(probabilities, legalMask) {
  let entropy = 0;
  for (let i = 0; i < probabilities.length; i++) {
    if (!legalMask[i] || probabilities[i] <= 0) continue;
    entropy -= probabilities[i] * Math.log(probabilities[i]);
  }
  return entropy;
}

function entropyLogitGradient(probabilities, actionIndex, legalMask) {
  const p = probabilities[actionIndex];
  if (!legalMask[actionIndex] || p <= 0) return 0;
  let expected = 0;
  for (let i = 0; i < probabilities.length; i++) {
    if (!legalMask[i] || probabilities[i] <= 0) continue;
    expected += probabilities[i] * (Math.log(probabilities[i]) + 1);
  }
  return -p * ((Math.log(p) + 1) - expected);
}

function randn(rng) {
  const u = Math.max(1e-12, rng());
  const v = Math.max(1e-12, rng());
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function cloneNetwork(net) {
  return {
    w1: net.w1.map(row => row.slice()),
    b1: net.b1.slice(),
    w2: net.w2.map(row => row.slice()),
    b2: net.b2.slice()
  };
}

function shuffle(items, rng) {
  for (let i = items.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [items[i], items[j]] = [items[j], items[i]];
  }
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

module.exports = { NeuralPPOAgent };
