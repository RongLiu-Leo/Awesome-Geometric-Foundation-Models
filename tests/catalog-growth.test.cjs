const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { pathToFileURL } = require('node:url');

function findChromium() {
  if (process.env.CHROME_BIN) return process.env.CHROME_BIN;
  const names = ['google-chrome', 'chromium', 'chromium-browser', 'chrome.exe', 'msedge.exe'];
  const candidates = (process.env.PATH || '').split(path.delimiter)
    .flatMap(dir => names.map(name => path.join(dir, name)));
  candidates.push('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium');
  for (const dir of [process.env.PROGRAMFILES, process.env['PROGRAMFILES(X86)'], process.env.LOCALAPPDATA].filter(Boolean)) {
    candidates.push(path.join(dir, 'Google/Chrome/Application/chrome.exe'), path.join(dir, 'Microsoft/Edge/Application/msedge.exe'));
  }
  return candidates.find(file => fs.existsSync(file));
}

// Run in a temporary copy of the entry page, immediately before app.js.
function addFixtures() {
  window.addEventListener('error', event => { document.documentElement.dataset.testError = event.message; });
  const dates = window.AtlasData.map(model => model.preprint).sort();
  const nextYear = Number(dates.at(-1).slice(0, 4)) + 1;
  for (const [id, preprint] of [['test-peer', dates[0]], ['test-future', nextYear + '-03-05']]) {
    window.AtlasData.push({
      id, preprint, name: id, title: 'Test paper ' + id, group: 'Test group',
      summary: 'Test summary for ' + id, paper: 'https://example.org/' + id,
      inputs: 'Test RGB inputs', output: 'Test points for ' + id,
      ...(id === 'test-peer' && { stages: [['Encode the pair', 'Fuse image features from both views.']] })
    });
  }
}

async function checkPages() {
  const check = (condition, message) => { if (!condition) throw new Error(message); };
  const one = selector => document.querySelector(selector);
  const cards = () => Array.from(document.querySelectorAll('.model-card'), card => card.hash);
  async function visit(hash) {
    const changed = new Promise(resolve => window.addEventListener('hashchange', resolve, { once: true }));
    location.hash = hash;
    await changed;
  }
  try {
    check(one('h1')?.textContent === 'test-future', 'A direct code URL must open the model even without a walkthrough');
    check(!one('main a[href="#model/test-future/code"]'), 'Do not offer an unavailable walkthrough');
    check(one('main').textContent.includes('Test points for test-future'), 'Missing catalog output');
    await visit('#overview');
    check(cards().at(-1) === '#model/test-future', 'The newest model belongs at the end of the index');
    check(cards().length === window.AtlasData.length, 'Missing catalog cards');
    one('.filter[data-group="Test group"]').click();
    check(cards().length === 2, 'Both new entries should match their group');
    const search = one('#model-search');
    search.value = 'Test paper test-peer';
    search.dispatchEvent(new Event('input'));
    check(cards().join() === '#model/test-peer', 'A paper title should be searchable within its group');
    await visit('#roadmap');
    for (const id of ['test-peer', 'test-future']) {
      check(one('a[href="#roadmap/' + id + '"]'), 'Missing timeline link for ' + id);
      await visit('#roadmap/' + id);
      check(document.activeElement.id === 'milestone-' + id, 'Missing focused milestone for ' + id);
    }
    await visit('#compare/all');
    check(one('tbody a[href="#model/test-peer"]') && one('tbody a[href="#model/test-future"]'), 'Missing comparison rows');
    await visit('#compare/test-peer/test-future');
    for (const id of ['test-peer', 'test-future']) {
      check(one('tbody').textContent.includes('Test points for ' + id), 'Missing pair comparison output for ' + id);
    }
    check(!one('main').textContent.includes('undefined'), 'Missing optional fields must have readable fallbacks');
    await visit('#sources');
    check(one('a[href="https://example.org/test-future"]'), 'Missing new paper in the source library');
    await visit('#model/test-peer');
    check(one('main').textContent.includes('Fuse image features from both views.'), 'Missing contributed architecture stage');
    await visit('#model/dust3r/code');
    check(one('.walk-snippet pre code')?.textContent.trim(), 'Existing walkthrough lost its original code');
    check(!document.documentElement.dataset.testError, document.documentElement.dataset.testError);
    document.title = 'PASS catalog growth';
  } catch (error) {
    document.title = 'FAIL ' + error.message;
  }
}

test('new catalog entries appear throughout the website', t => {
  const chromium = findChromium();
  if (!chromium) return t.skip('Set CHROME_BIN to an installed Chrome, Chromium or Edge executable.');
  const site = path.resolve(__dirname, '..');
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'catalog-growth-'));
  t.after(() => {
    assert.equal(path.dirname(temp), path.resolve(os.tmpdir()));
    fs.rmSync(temp, { recursive: true, force: true, maxRetries: 3 });
  });
  const fixtureFile = path.join(temp, 'fixtures.js');
  fs.writeFileSync(fixtureFile, `(${addFixtures})();\nwindow.addEventListener("load", ${checkPages});`);
  const index = fs.readFileSync(path.join(site, 'index.html'), 'utf8');
  for (const [, asset] of index.matchAll(/<(?:script|link)\b[^>]*?\b(?:src|href)="([^"]+)"/g)) {
    const destination = path.join(temp, asset);
    fs.mkdirSync(path.dirname(destination), { recursive: true });
    fs.copyFileSync(path.join(site, asset), destination);
  }
  const page = index.replace('<script defer src="app.js">', '<script defer src="fixtures.js"></script>\n<script defer src="app.js">');
  const pageFile = path.join(temp, 'index.html');
  fs.writeFileSync(pageFile, page);
  const result = spawnSync(chromium, [
    '--headless', '--disable-gpu', '--no-first-run', '--disable-background-networking',
    '--user-data-dir=' + path.join(temp, 'profile'), '--virtual-time-budget=5000', '--dump-dom',
    pathToFileURL(pageFile).href + '#model/test-future/code'
  ], { encoding: 'utf8', windowsHide: true, timeout: 30000, maxBuffer: 4 * 1024 * 1024 });
  if (result.error) throw result.error;
  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.stdout.match(/<title>(.*?)<\/title>/)?.[1], 'PASS catalog growth');
});
