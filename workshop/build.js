const fs = require('fs');
const src = (f) => fs.readFileSync(f, 'utf8');
const watch = JSON.parse(src('src/experiment-notes.json'));
let ui = src('src/ui.js')
  .replace("'__BRAIN_B64__'", JSON.stringify(src('brains/brain_final.b64').trim()))
  .replace('__BRAIN_META__', src('brains/brain_final.json').trim());
for (const [k, v] of Object.entries(watch)) ui = ui.replace(`'__${k}__'`, JSON.stringify(v));
const left = ui.match(/__[A-Z0-9_]+__/g);
if (left) { console.error('unfilled placeholders:', left); process.exit(1); }
const out = src('src/index.html')
  .replace('/*__CSS__*/', () => src('src/style.css'))
  .replace('/*__ENGINE__*/', () => src('engine.js'))
  .replace('/*__UI__*/', () => ui);
fs.mkdirSync('dist', { recursive: true });
fs.writeFileSync('dist/dqn-workshop.html', out);
console.log('built dist/dqn-workshop.html', (out.length / 1024).toFixed(0) + ' KB');
