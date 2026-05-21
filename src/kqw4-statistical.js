"use strict";

const fs = require("fs");
const path = require("path");
const Module = require("module");
const { CHARACTERS } = require("./constants");

const baselineDir = path.join(__dirname, "..", "baseline", "kqw4");

function loadBaselineClasses() {
  const aiEngineSource = fs.readFileSync(path.join(baselineDir, "ai-engine.raw.js"), "utf8");
  const statisticalSource = fs.readFileSync(path.join(baselineDir, "statistical-ai.raw.js"), "utf8");
  const filename = path.join(baselineDir, "kqw4-bundle.js");
  const code = [
    `"use strict";`,
    `const CHARACTERS = ${JSON.stringify(CHARACTERS)};`,
    aiEngineSource,
    statisticalSource,
    `module.exports = { AIEngine, StatisticalAI };`
  ].join("\n");

  const mod = new Module(filename, module);
  mod._compile(code, filename);
  return mod.exports;
}

const { AIEngine, StatisticalAI } = loadBaselineClasses();

module.exports = { AIEngine, StatisticalAI, BASELINE_COMMIT: "996bde8" };
