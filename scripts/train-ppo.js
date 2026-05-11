"use strict";

const fs = require("fs");
const path = require("path");
const { CoupEnv } = require("../src/coup-env");
const { HeuristicAgent, RandomMaskedAgent } = require("../src/baseline-agents");
const { encodeLegalMask, encodeObservation } = require("../src/encoder");
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
const modelPath = path.resolve(args.out || "models/ppo-agent.json");
const rng = createRng(seed);

fs.mkdirSync(path.dirname(modelPath), { recursive: true });

const learner = args.resume && fs.existsSync(args.resume)
  ? NeuralPPOAgent.load(path.resolve(args.resume))
  : new NeuralPPOAgent({
      seed,
      hiddenSize: Number(args.hidden || 96),
      policyLearningRate: Number(args["policy-lr"] || 0.0008),
      valueLearningRate: Number(args["value-lr"] || 0.0015),
      entropyBonus: Number(args.entropy || 0.01)
    });

const snapshots = [learner.snapshot()];
let totalGames = 0;

for (let iteration = 1; iteration <= iterations; iteration++) {
  const rollouts = [];
  const trainStats = {
    wins: Array(playerCount).fill(0),
    turns: 0,
    games: 0
  };

  for (let game = 0; game < gamesPerIteration; game++) {
    const opponents = buildOpponentLineup({ learner, snapshots, playerCount, rng, seed: seed + iteration * 10000 + game });
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
    const evalStats = evaluate(learner, { games: 100, playerCount, seed: seed + iteration * 200000 });
    console.log(
      `eval iter=${iteration} rlWin=${evalStats.rlWinRate.toFixed(1)}% ` +
      `avgTurns=${evalStats.averageTurns.toFixed(1)}`
    );
  }

  learner.save(modelPath);
}

console.log(`saved ${modelPath}`);

function buildOpponentLineup({ learner, snapshots, playerCount, rng, seed }) {
  const lineup = Array(playerCount).fill(null);
  const learnerSeat = Math.floor(rng() * playerCount);
  lineup[learnerSeat] = learner;

  for (let id = 0; id < playerCount; id++) {
    if (lineup[id]) continue;
    const roll = rng();
    if (roll < 0.45 && snapshots.length) lineup[id] = snapshots[Math.floor(rng() * snapshots.length)];
    else if (roll < 0.75) lineup[id] = new HeuristicAgent({ seed: seed + id });
    else lineup[id] = new RandomMaskedAgent({ seed: seed + id });
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
    const action = agent.act(vector, legalMask, {
      greedy: agent !== learner,
      observation
    });

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

function evaluate(agent, { games, playerCount, seed }) {
  let rlWins = 0;
  let turns = 0;
  for (let game = 0; game < games; game++) {
    const env = new CoupEnv({ playerCount, seed: seed + game });
    const agents = Array.from({ length: playerCount }, (_, id) => {
      if (id === 0) return agent;
      if (id % 2 === 0) return new HeuristicAgent({ seed: seed + game * 13 + id });
      return new RandomMaskedAgent({ seed: seed + game * 17 + id });
    });

    let safety = 20000;
    while (!env.gameOver && safety-- > 0) {
      const playerId = env.currentPlayerId();
      const observation = env.observe(playerId);
      const vector = encodeObservation(observation);
      const legalMask = encodeLegalMask(env, playerId);
      const action = agents[playerId].act(vector, legalMask, {
        greedy: playerId === 0,
        observation
      });
      env.step(action.actionIndex);
    }
    if (safety <= 0) throw new Error("PPO eval game exceeded safety limit");
    if (env.winnerId() === 0) rlWins++;
    turns += env.turnCount;
  }
  return {
    rlWinRate: 100 * rlWins / games,
    averageTurns: turns / games
  };
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
