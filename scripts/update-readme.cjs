// Refresh the paper lists in the README and the homepage. Use --check to verify both.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const root = path.resolve(__dirname, "..");
const context = {window: {}};
vm.runInNewContext(fs.readFileSync(path.join(root, "data.js"), "utf8"), context, {filename: "data.js"});
const models = context.window.AtlasData;
const required = ["id", "name", "title", "preprint", "group", "summary", "paper", "inputs", "output"];
const ids = new Set();
for (const model of models) {
  for (const field of required) {
    assert(typeof model[field] === "string" && model[field].trim(), `${model.id || "New entry"} needs ${field}`);
  }
  assert(/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(model.id) && model.id !== "all", `Invalid model ID: ${model.id}`);
  assert(!ids.has(model.id), `Duplicate model ID: ${model.id}`);
  ids.add(model.id);
  assert(model.group !== "All models", "Choose a specific architecture group");
  const date = new Date(model.preprint + "T00:00:00Z");
  assert(Number.isFinite(date.valueOf()) && date.toISOString().slice(0, 10) === model.preprint, `Invalid date for ${model.id}`);
  for (const key of ["paper", "project", "code", "figure", "moreCode"]) {
    if (model[key]) assert(/^https?:$/.test(new URL(model[key]).protocol), `Invalid ${key} URL for ${model.id}`);
  }
  if (model.stages) assert(Array.isArray(model.stages) && model.stages.every(stage =>
    Array.isArray(stage) && stage.length === 2 && stage.every(text => typeof text === "string" && text.trim())),
    `Expected [operation, explanation] stage pairs for ${model.id}`);
}
for (const model of models) {
  if (model.compareWith) assert(ids.has(model.compareWith) && model.id !== model.compareWith, `Invalid comparison for ${model.id}`);
}

const site = "https://rongliu-leo.github.io/Awesome-Geometric-Foundation-Models/";
const lines = [];
const paperItems = [];
const escapeHtml = text => text.replace(/[&<>"']/g, char => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
})[char]);
let lastYear;
for (const model of models.sort((a, b) => a.preprint.localeCompare(b.preprint) || a.id.localeCompare(b.id))) {
  const year = model.preprint.slice(0, 4);
  if (year !== lastYear) {
    lines.push("", `### ${year}`, "");
    lastYear = year;
  }
  const links = [["Paper", model.paper], ["Code", model.code], ["Project", model.project]]
    .filter(([, url]) => url).map(([label, url]) => `[${label}](<${url}>)`);
  const name = model.name.replace(/[\\\[\]]/g, "\\$&");
  lines.push(`- [**${name}**](${site}#model/${model.id}) · ${links.join(" · ")} — ${model.summary}`);
  paperItems.push(`<li><a href="${escapeHtml(model.paper)}"><strong>${escapeHtml(model.name)}</strong></a> (${year}) — ${escapeHtml(model.summary)}</li>`);
}

const outputs = [["README.md", [...lines, ""]], ["index.html", paperItems]];
const section = /^([ \t]*)<!-- catalog:start -->\n[\s\S]*?^\1<!-- catalog:end -->/m;
const changes = [];
for (const [filename, entries] of outputs) {
  const destination = path.join(root, filename);
  const current = fs.readFileSync(destination, "utf8").replace(/\r\n/g, "\n");
  assert(section.test(current), `${filename} is missing its catalog markers`);
  const updated = current.replace(section, (match, indent) =>
    ["<!-- catalog:start -->", ...entries, "<!-- catalog:end -->"]
      .map(line => line && indent + line).join("\n"));
  if (updated !== current) changes.push({filename, destination, updated});
}
for (const {filename, destination, updated} of changes) {
  assert(!process.argv.includes("--check"), `Refresh ${filename} with node scripts/update-readme.cjs`);
  fs.writeFileSync(destination, updated);
}
console.log(`Verified ${models.length} model profiles in the README and homepage.`);
