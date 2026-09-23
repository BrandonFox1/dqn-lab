const chromium = require('@sparticuz/chromium'); const puppeteer = require('puppeteer-core'); const path = require('path');
require('fs').mkdirSync('shots', { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
(async () => {
  const b = await puppeteer.launch({ executablePath: await chromium.executablePath(), args: [...chromium.args, '--no-sandbox'], headless: true });
  const errs = [];
  const open = async (vp, dark) => { const p = await b.newPage(); p.on('pageerror', (e) => errs.push(e.message)); await p.setViewport(vp); await p.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: dark ? 'dark' : 'light' }]); await p.goto('file://' + path.resolve('dist/dqn-workshop.html')); await sleep(600); return p; };
  // schedule check: fresh agent, 7,600 steps -> exams at 0, 2500, 5000, 7500
  const p = await open({ width: 390, height: 844, deviceScaleFactor: 2, isMobile: true }, false);
  const ex = await p.evaluate(() => { const A = window.__workshop.arcade; window.__workshop.selectTab('tab-arcade'); const t0 = performance.now(); A._train(7600); return { steps: A.state.exams.map((e) => e.step), sps: Math.round(7600 / ((performance.now() - t0) / 1000)) }; });
  console.log('exam steps', ex.steps.join(','), 'ok:', JSON.stringify(ex.steps) === JSON.stringify([0, 2500, 5000, 7500]), 'raw train speed', ex.sps, 'steps/s');
  await p.evaluate(() => document.querySelector('#arc-stats').scrollIntoView({ block: 'start' })); await sleep(300);
  await p.screenshot({ path: 'shots/n_m_arcade.png' });
  // gap label: pellet scenario on bench
  await p.evaluate(() => { window.__workshop.selectTab('tab-bench'); document.querySelector('#bench-gapviz').scrollIntoView({ block: 'center' }); }); await sleep(300);
  await p.screenshot({ path: 'shots/n_m_gap.png' });
  await p.click('.chip[data-id="ghost"]'); await sleep(200);
  const g = await p.evaluate(() => { const c = window.__workshop.bench.compute(); return { q: c.q, t: c.target, gap: c.gap }; });
  console.log('ghost scenario with final brain', JSON.stringify(g));
  await p.screenshot({ path: 'shots/n_m_gap_ghost.png' });
  // desktop dark: trained brain + continued training
  const d = await open({ width: 1280, height: 900, deviceScaleFactor: 1 }, true);
  await d.evaluate(() => { const W = window.__workshop; W.selectTab('tab-arcade'); W.arcade._train(4000); W.arcade.loadTrained(); W.arcade._train(9000); document.querySelector('#panel-arcade').scrollIntoView(); }); await sleep(400);
  await d.screenshot({ path: 'shots/n_d_arcade.png' });
  console.log('errors:', errs.length ? errs.join(' | ') : 'none');
  await b.close();
})();
