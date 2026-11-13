import fs from 'node:fs';
import path from 'node:path';

const base = 'https://api.tcgdex.net/v2/en';
const dir = path.join('data', 'tcgdex');
fs.mkdirSync(dir, { recursive: true });
const outFile = path.join(dir, 'cards.jsonl');
const failFile = path.join(dir, 'failed.txt');

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function get(url, tries = 4) {
  for (let i = 0; i < tries; i++) {
    try {
      const res = await fetch(url);
      if (res.status === 429) throw new Error('429');
      if (res.ok) return await res.json();
      if (res.status === 404) return null;
    } catch (e) {
      if (e.message === '429') throw e;
    }
    await sleep(1000 * (i + 1));
  }
  return null;
}

const done = new Set();
if (fs.existsSync(outFile)) {
  for (const line of fs.readFileSync(outFile, 'utf8').split('\n')) {
    if (!line) continue;
    try {
      done.add(JSON.parse(line).id);
    } catch {}
  }
}

const sets = await get(`${base}/sets`);
fs.writeFileSync(path.join(dir, 'sets.json'), JSON.stringify(sets));

const ids = [];
for (const s of sets) {
  const detail = await get(`${base}/sets/${encodeURIComponent(s.id)}`);
  if (!detail) {
    fs.appendFileSync(failFile, `set ${s.id}\n`);
    continue;
  }
  for (const c of detail.cards ?? []) ids.push(c.id);
  await sleep(50);
}

const todo = ids.filter((id) => !done.has(id));
console.log('toplam kart:', ids.length, 'yapilmis:', done.size, 'kalan:', todo.length);

let idx = 0;
let count = 0;

async function worker() {
  while (true) {
    const i = idx++;
    if (i >= todo.length) return;
    const id = todo[i];
    try {
      const card = await get(`${base}/cards/${encodeURIComponent(id)}`);
      if (card) fs.appendFileSync(outFile, JSON.stringify(card) + '\n');
      else fs.appendFileSync(failFile, `card ${id}\n`);
    } catch (e) {
      console.log('durduruldu:', e.message);
      idx = todo.length;
      return;
    }
    count++;
    if (count % 200 === 0) console.log(count, '/', todo.length);
    await sleep(100);
  }
}

await Promise.all([worker(), worker(), worker()]);
console.log('bitti, bu calistirmada:', count);