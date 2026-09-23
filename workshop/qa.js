const chromium = require('@sparticuz/chromium');
const puppeteer = require('puppeteer-core');
const path = require('path');
require('fs').mkdirSync('shots', { recursive: true });
const FILE = 'file://' + path.resolve('dist/dqn-workshop.html');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const results = [];
const check = (name, ok, info = '') => { results.push({ name, ok, info }); console.log(ok ? '  ok  ' : '  FAIL', name, info); };
(async () => {
  const browser = await puppeteer.launch({ executablePath: await chromium.executablePath(), args: [...chromium.args, '--no-sandbox'], headless: true });
  const errors = [];
  async function open(vp, dark) {
    const page = await browser.newPage();
    page.on('console', (m) => { if (m.type() === 'error' && !/fonts\.g|ERR_|net::|Failed to load resource/.test(m.text())) errors.push(m.text()); });
    page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
    await page.setViewport(vp);
    await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: dark ? 'dark' : 'light' }]);
    await page.goto(FILE, { waitUntil: 'domcontentloaded' });
    await sleep(700);
    return page;
  }
  const mobile = { width: 390, height: 844, deviceScaleFactor: 2, isMobile: true, hasTouch: false };
  const page = await open(mobile, false);
  const txt = (sel) => page.$eval(sel, (e) => e.textContent.trim());
  const numOf = (s) => parseFloat(s.replace('\u2212', '-'));

  await page.screenshot({ path: 'shots/m_top.png' });
  // horizontal overflow check
  const overflow = await page.evaluate(() => ['tab-bench', 'tab-tank', 'tab-arcade'].map((t) => { window.__workshop.selectTab(t); return document.documentElement.scrollWidth - document.documentElement.clientWidth; }));
  await page.evaluate(() => window.__workshop.selectTab('tab-bench'));
  check('no horizontal scroll on phone, any tab', overflow.every((o) => o <= 0), `overflow ${overflow.join('/')}px`);

  // ---- Bench
  check('bench: trained brain loaded, q shown', !isNaN(numOf(await txt('#ro-q'))), await txt('#ro-q'));
  await page.click('.chip[data-id="ghost"]'); await sleep(100);
  const tGhost = await page.evaluate(() => window.__workshop.bench.compute());
  check('bench: walking into the ghost is terminal with target -2.91', tGhost.res.terminal && Math.abs(tGhost.target + 2.91) < 1e-6, `target ${tGhost.target}`);
  check('bench: game-over box visible', await page.$eval('#future-over', (e) => !e.hidden));
  const gap0 = Math.abs(tGhost.gap);
  await page.click('[data-nudge="10"]'); await sleep(100);
  const after = await page.evaluate(() => window.__workshop.bench.compute());
  check('bench: 10 nudges shrink the gap', Math.abs(after.gap) < gap0, `${gap0.toFixed(3)} -> ${Math.abs(after.gap).toFixed(3)}`);
  await page.click('#bench-overlay button[data-v="ripple"]'); await sleep(100);
  const rippleNote = await txt('#bench-overlay-note');
  check('bench: ripple overlay explains change', /Largest change/.test(rippleNote), rippleNote.slice(0, 90));
  await page.screenshot({ path: 'shots/m_bench_ripple.png', fullPage: false });
  await page.click('#bench-overlay button[data-v="value"]'); await sleep(100);
  await page.click('.chip[data-id="pellet"]'); await sleep(100);
  const pel = await page.evaluate(() => window.__workshop.bench.compute());
  check('bench: pellet step reward +9 points -> r = 0.09', Math.abs(pel.r - 0.09) < 1e-9 && !pel.res.terminal, `r ${pel.r}`);
  await page.screenshot({ path: 'shots/m_bench_full.png', fullPage: true });
  // undo restores
  await page.click('#bench-undo'); await sleep(50);
  // frozen off: target moves when nudging
  await page.click('#bench-overlay button[data-v="game"]');
  await page.click('.chip[data-id="flee"]'); await sleep(50);
  await page.$eval('#bench-frozen', (e) => { e.click(); }); await sleep(50);
  const t1 = (await page.evaluate(() => window.__workshop.bench.compute())).target;
  await page.click('[data-nudge="10"]'); await sleep(50);
  const t2 = (await page.evaluate(() => window.__workshop.bench.compute())).target;
  check('bench: with no frozen copy, nudging moves the target', Math.abs(t2 - t1) > 1e-4, `${t1.toFixed(4)} -> ${t2.toFixed(4)}`);
  check('bench: nudge words mention chasing', /chasing|overshoot|moved/.test(await txt('#nudge-words')), (await txt('#nudge-words')).slice(0, 80));
  await page.$eval('#bench-frozen', (e) => { e.click(); });
  // tapping the board: move pac-man
  await page.click('#bench-tool button[data-v="pellet"]');
  const cells = await page.$$('#bench-board button.cell');
  await cells[0].click(); await sleep(50);
  check('bench: tapping edits the board without errors', errors.length === 0, errors.join(' | '));
  // lr 0.05 overshoot
  await page.click('.chip[data-id="ghost"]');
  await page.evaluate(() => { const s = window.__workshop.bench.state; s.lr = 0.05; });
  await page.$eval('[data-nudge="10"]', (e) => e.click()); await sleep(50);
  check('bench: big learning rate triggers overshoot message', /overshoot/.test(await txt('#nudge-words')), (await txt('#nudge-words')).slice(0, 60));
  await page.click('#bench-undo');
  await page.evaluate(() => { window.__workshop.bench.state.lr = 0.003; });

  // ---- Tank
  await page.click('#tab-tank'); await sleep(100);
  for (let i = 0; i < 4; i++) { await page.click('#tank-sweep'); await sleep(40); }
  const known = await txt('#tank-stats');
  check('tank: 4 table sweeps spread knowledge', /Squares that know about the cherry\s*(\d+)/.test(known), known.replace(/\s+/g, ' ').slice(0, 120));
  await page.click('#tank-nums'); await sleep(50);
  await page.screenshot({ path: 'shots/m_tank.png', fullPage: true });
  await page.click('#tank-brain button[data-v="net"]'); await sleep(50);
  for (let i = 0; i < 6; i++) { await page.click('#tank-sweep'); await sleep(40); }
  const knownNet = await txt('#tank-stats');
  check('tank: network sweeps run', /Weights in the network/.test(knownNet), knownNet.replace(/\s+/g, ' ').slice(0, 120));
  await page.click('#tank-run'); await sleep(1500); await page.click('#tank-run');
  check('tank: exploring runs without errors', errors.length === 0, errors.join(' | '));
  await page.screenshot({ path: 'shots/m_tank_net.png' });

  // ---- Arcade
  await page.click('#tab-arcade'); await sleep(100);
  await page.click('#arc-speed button[data-v="turbo"]');
  await page.click('#arc-train');
  const s0 = await page.evaluate(() => window.__workshop.arcade.agent.steps);
  await sleep(8000);
  const s1 = await page.evaluate(() => window.__workshop.arcade.agent.steps);
  const exams = await page.evaluate(() => window.__workshop.arcade.state.exams.map((e) => e.step));
  check('arcade: exams run on schedule', exams[0] === 0 && exams.length === 1 + Math.floor(s1 / 2500), `exams at ${exams.join(',')}`);
  check('arcade: turbo trains', s1 > s0 + 500, `${((s1 - s0) / 8).toFixed(0)} steps/s in headless phone emulation`);
  await page.screenshot({ path: 'shots/m_arcade_training.png', fullPage: true });
  await page.click('.exp button'); await sleep(300); // frozen-copy experiment
  const evs = await page.evaluate(() => window.__workshop.arcade.state.events.map((e) => e.label));
  check('arcade: experiment marks the chart', evs.includes('no frozen copy'), evs.join(','));
  await page.evaluate(() => { document.querySelector('#arc-knob-drawer').open = true; });
  await page.click('#arc-defaults'); await sleep(50);
  check('arcade: restore defaults turns the frozen copy back on', await page.evaluate(() => window.__workshop.arcade.agent.cfg.useTarget && document.querySelector('#arc-frozen').checked));
  await page.click('#arc-load'); await sleep(200);
  // evaluate trained brain inside the page
  const ev = await page.evaluate(() => {
    const E = Engine; const ag = window.__workshop.arcade.agent; const env = new E.MiniPac({ seed: 123 }); let sc = 0, wins = 0;
    for (let i = 0; i < 30; i++) { let o = env.reset(500 + i); while (!env.done) o = env.step(ag.act(o, 0)).obs; sc += env.score; wins += env.won; }
    return { score: sc / 30, wins };
  });
  const lastExam = await page.evaluate(() => { const x = window.__workshop.arcade.state.exams; return x[x.length - 1]; });
  check('arcade: loading the trained brain runs an exam near the top', lastExam.wins >= 8, JSON.stringify(lastExam));
  check('arcade: loaded trained brain clears most boards in-page', ev.wins >= 24, `avg ${ev.score.toFixed(0)}, cleared ${ev.wins}/30`);
  await page.click('#arc-test'); await sleep(3000);
  check('arcade: test drive runs', /Test drive/.test(await txt('#arcade-status')), await txt('#arcade-status'));
  await page.screenshot({ path: 'shots/m_arcade_test.png' });
  await page.click('#arc-test');
  await page.click('#arc-send'); await sleep(300);
  check('arcade: send to bench switches tab and brain', await page.evaluate(() => !document.querySelector('#panel-bench').hidden && window.__workshop.bench.state.brainKind === 'arcade'));
  await page.close();

  // ---- dark + desktop
  const d = await open({ width: 390, height: 844, deviceScaleFactor: 2, isMobile: true, hasTouch: false }, true);
  await d.screenshot({ path: 'shots/m_dark_top.png' });
  await d.evaluate(() => window.scrollTo(0, 700)); await sleep(200);
  await d.screenshot({ path: 'shots/m_dark_bench.png' });
  await d.close();
  const w = await open({ width: 1280, height: 900, deviceScaleFactor: 1 }, false);
  await w.screenshot({ path: 'shots/d_top.png' });
  await w.evaluate(() => window.scrollTo(0, 620)); await sleep(200);
  await w.screenshot({ path: 'shots/d_bench.png' });
  await w.click('#tab-arcade'); await sleep(200);
  await w.click('#arc-load'); await w.click('#arc-speed button[data-v="turbo"]'); await w.click('#arc-train'); await sleep(5000);
  await w.screenshot({ path: 'shots/d_arcade.png' });
  await w.close();
  const wd = await open({ width: 1280, height: 900, deviceScaleFactor: 1 }, true);
  await wd.click('#tab-tank'); await sleep(100);
  for (let i = 0; i < 5; i++) { await wd.click('#tank-sweep'); await sleep(30); }
  await wd.evaluate(() => window.scrollTo(0, 620)); await sleep(200);
  await wd.screenshot({ path: 'shots/d_dark_tank.png' });
  await wd.close();

  check('no console or page errors anywhere', errors.length === 0, errors.slice(0, 5).join(' | '));
  await browser.close();
  const bad = results.filter((r) => !r.ok).length;
  console.log(`\n${results.length - bad} passed, ${bad} failed`);
  process.exit(bad ? 1 : 0);
})().catch((e) => { console.error('QA crashed:', e); process.exit(2); });
