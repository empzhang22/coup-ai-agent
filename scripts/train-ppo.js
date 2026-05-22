"use strict";

const fs = require("fs");
const path = require("path");
const { CoupEnv } = require("../src/coup-env");
const { RandomMaskedAgent } = require("../src/baseline-agents");
const { StatisticalMaskedAgent } = require("../src/statistical-agent");
const { encodeLegalMask, encodeObservation, OBSERVATION_SIZE, ACTION_COUNT } = require("../src/encoder");
const { evaluateVsStatistical } = require("../src/eval-vs-statistical");
const { NeuralPPOAgent } = require("../src/neural-ppo-agent");
const { createRng } = require("../src/rng");

const args = parseArgs(process.argv.slice(2));
const iterations = Number(args.iterations || 200);
const gamesPerIteration = Number(args["games-per-iteration"] || 64);
const playerCount = Number(args.players || 3);
const seed = Number(args.seed || 77);
const epochs = Number(args.epochs || 4);
const batchSize = Number(args["batch-size"] || 128);
const snapshotEvery = Number(args["snapshot-every"] || 10);
const evalEvery = Number(args["eval-every"] || 10);
const evalStatGames = Number(args["eval-stat-games"] || 200);
const curriculumEnd = Number(args["curriculum-end"] || Math.floor(iterations * 0.4));
const modelPath = path.resolve(args.out || "models/ppo-agent.json");
const bestStatPath = modelPath.replace(/\.json$/, "-best-statistical.json");
const rng = createRng(seed);

fs.mkdirSync(path.dirname(modelPath), { recursive: true });

const learner = loadOrCreateLearner();
const snapshots = [learner.snapshot()];
let totalGames = 0;
let bestStatWinRate = -1;
let bestStatIter = 0;

if (args.resume && fs.existsSync(path.resolve(args.resume))) {
  console.log(`WARNING: --resume loads an existing checkpoint. Old pre-reveal-parity models are incompatible.`);
}

console.log(
  `training: iterations=${iterations} games/iter=${gamesPerIteration} players=${playerCount} ` +
  `curriculumEnd=${curriculumEnd} evalStatGames=${evalStatGames}`
);
console.log(`learner seat: 0 (matches browser AI 1 = RL)`);
console.log(`opponent mix: early 50% random / 30% self-play / 20% statistical → late 10% / 25% / 65%`);

for (let iteration = 1; iteration <= iterations; iteration++) {
  const rollouts = [];
  const trainStats = {
    wins: Array(playerCount).fill(0),
    turns: 0,
    games: 0
  };

  for (let game = 0; game < gamesPerIteration; game++) {
    const opponents = buildOpponentLineup({
      learner,
      snapshots,
      playerCount,
      iteration,
      curriculumEnd,
      rng,
      seed: seed + iteration * 10000 + game
    });
    const result = collectGame({
      agents: opponents,
      learner,
      learnerPlayerIds: opponents.map((agent, id) => agent === learner ? id : -1).filter(id => id !== -1),
      playerCount,
      seed: seed + iteration * 100000 + game
    });
    rollouts.push(...result.rollouts);
    trainStats.games++;
    trainStats.turns += result.turns;
    if (result.winner !== null) trainStats.wins[result.winner]++;
  }

  totalGames += gamesPerIteration;
  const update = learner.update(rollouts, { epochs, batchSize });

  if (iteration % snapshotEvery === 0) {
    snapshots.push(learner.snapshot());
    while (snapshots.length > 12) snapshots.shift();
    const snapshotPath = modelPath.replace(/\.json$/, `-iter-${iteration}.json`);
    learner.save(snapshotPath);
  }

  const winRates = trainStats.wins.map(w => `${(100 * w / trainStats.games).toFixed(1)}%`).join(" / ");
  console.log(
    `iter=${iteration}/${iterations} games=${totalGames} samples=${update.samples} ` +
    `turns=${(trainStats.turns / trainStats.games).toFixed(1)} wins=${winRates} ` +
    `policyLoss=${update.policyLoss.toFixed(4)} valueLoss=${update.valueLoss.toFixed(4)} entropy=${update.entropy.toFixed(3)}`
  );

  if (iteration % evalEvery === 0 || iteration === iterations) {
    const statEval = evaluateVsStatistical(learner, {
      games: evalStatGames,
      playerCount,
      seed: seed + iteration * 200000
    });
    console.log(
      `stat-eval iter=${iteration} rlWin=${statEval.rlWinRate.toFixed(1)}% ` +
      `avgTurns=${statEval.averageTurns.toFixed(1)} draws=${statEval.draws}`
    );

    if (statEval.rlWinRate > bestStatWinRate) {
      bestStatWinRate = statEval.rlWinRate;
      bestStatIter = iteration;
      learner.save(bestStatPath);
      console.log(`  new best vs statistical: ${bestStatWinRate.toFixed(1)}% → ${bestStatPath}`);
    }
  }

  learner.save(modelPath);
}

console.log(`saved ${modelPath}`);
if (bestStatIter > 0) {
  console.log(`best vs statistical: ${bestStatWinRate.toFixed(1)}% at iter ${bestStatIter} → ${bestStatPath}`);
  console.log(`Run: npm run evaluate:statistical -- --model ${bestStatPath} --games 5000 --players all`);
}

function loadOrCreateLearner() {
  if (args.resume && fs.existsSync(path.resolve(args.resume))) {
    const resumePath = path.resolve(args.resume);
    const data = JSON.parse(fs.readFileSync(resumePath, "utf8"));
    if (data.type === "NeuralPPOAgent") {
      if (data.observationSize !== OBSERVATION_SIZE || data.actionCount !== ACTION_COUNT) {
        throw new Error(
          `Cannot resume ${resumePath}: incompatible architecture ` +
          `(obs ${data.observationSize}→${OBSERVATION_SIZE}, actions ${data.actionCount}→${ACTION_COUNT}). ` +
          `Retrain from scratch without --resume.`
        );
      }
      return NeuralPPOAgent.load(resumePath);
    }
  }
  return new NeuralPPOAgent({
    seed,
    hiddenSize: Number(args.hidden || 96),
    policyLearningRate: Number(args["policy-lr"] || 0.0008),
    valueLearningRate: Number(args["value-lr"] || 0.0015),
    entropyBonus: Number(args.entropy || 0.01)
  });
}

function opponentMixWeights(iteration, curriculumEnd) {
  const progress = Math.min(1, iteration / Math.max(1, curriculumEnd));
  return {
    random: lerp(0.5, 0.1, progress),
    snapshot: lerp(0.3, 0.25, progress),
    statistical: lerp(0.2, 0.65, progress)
  };
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function pickOpponentType(rng, weights, hasSnapshots) {
  let roll = rng();
  if (roll < weights.random) return "random";
  roll -= weights.random;
  if (roll < weights.snapshot && hasSnapshots) return "snapshot";
  return "statistical";
}

function buildOpponentLineup({ learner, snapshots, playerCount, iteration, curriculumEnd, rng, seed }) {
  const weights = opponentMixWeights(iteration, curriculumEnd);
  const lineup = Array(playerCount).fill(null);
  const learnerSeat = 0;
  lineup[learnerSeat] = learner;

  for (let id = 0; id < playerCount; id++) {
    if (lineup[id]) continue;
    const type = pickOpponentType(rng, weights, snapshots.length > 0);
    if (type === "snapshot") {
      lineup[id] = snapshots[Math.floor(rng() * snapshots.length)];
    } else if (type === "random") {
      lineup[id] = new RandomMaskedAgent({ seed: seed + id });
    } else {
      lineup[id] = new StatisticalMaskedAgent({ playerId: id, seed: seed + id });
    }
  }
  return lineup;
}

function collectGame({ agents, learner, learnerPlayerIds, playerCount, seed }) {
  const env = new CoupEnv({ playerCount, seed });
  const rollouts = [];
  const latestByPlayer = Array(playerCount).fill(null);
  const learnerSeats = new Set(learnerPlayerIds);
  let safety = 20000;

  while (!env.gameOver && safety-- > 0) {
    const playerId = env.currentPlayerId();
    const observation = env.observe(playerId);
    const vector = encodeObservation(observation);
    const legalMask = encodeLegalMask(env, playerId);
    const agent = agents[playerId];

    let action;
    if (agent instanceof StatisticalMaskedAgent) {
      action = agent.act(vector, legalMask, { env });
    } else {
      action = agent.act(vector, legalMask, {
        greedy: agent !== learner,
        observation
      });
    }

    if (learnerSeats.has(playerId)) {
      const transition = {
        playerId,
        observation: vector,
        legalMask,
        actionIndex: action.actionIndex,
        oldLogProb: action.logProb,
        value: action.value,
        reward: 0
      };
      rollouts.push(transition);
      latestByPlayer[playerId] = transition;
    }

    const result = env.step(action.actionIndex);
    for (let id = 0; id < result.rewards.length; id++) {
      if (latestByPlayer[id]) latestByPlayer[id].reward += result.rewards[id];
    }
  }

  if (safety <= 0) throw new Error("PPO training game exceeded safety limit");
  return { rollouts, winner: env.winnerId(), turns: env.turnCount };
}

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (!arg.startsWith("--")) continue;
    const key = arg.slice(2);
    const next = argv[i + 1];
    if (!next || next.startsWith("--")) out[key] = true;
    else out[key] = argv[++i];
  }
  return out;
}
