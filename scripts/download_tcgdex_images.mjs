import fs from 'node:fs';
import path from 'node:path';

const quality = process.argv[2] ?? 'high';
const ext = process.argv[3] ?? 'webp';
const dir = path.join('data', 'tcgdex');
const imgDir = path.join(dir, 'images');
const failFile = path.join(dir, 'image_failed.txt');
fs.mkdirSync(imgDir, { recursive: true });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const safe = (id) => id.replace(/[^A-Za-z0-9._-]/g, '_');

const cards = fs
  .readFileSync(path.join(dir, 'cards.jsonl'), 'utf8')
  .split('\n')
  .filter(Boolean)
  .map((l) => JSON.parse(l))
  .filter((c) => c.image)
  .map((c) => ({ id: c.id, url: `${c.image}/${quality}.${ext}`, file: path.join(imgDir, `${safe(c.id)}.${ext}`) }));

const todo = cards.filter((c) => !fs.existsSync(c.file));
console.log('gorselli kart:', cards.length, 'inecek:', todo.length, 'kalite:', quality, ext);

async function fetchImage(url, tries = 4) {
  for (let i = 0; i < tries; i++) {
    try {
      const res = await fetch(url);
      if (res.status === 429) throw new Error('429');
      if (res.ok) return Buffer.from(await res.arrayBuffer());
      if (res.status === 404) return null;
    } catch (e) {
      if (e.message === '429') throw e;
    }
    await sleep(1000 * (i + 1));
  }
  return null;
}

let idx = 0;
let count = 0;
let bytes = 0;

async function worker() {
  while (true) {
    const i = idx++;
    if (i >= todo.length) return;
    const item = todo[i];
    try {
      const buf = await fetchImage(item.url);
      if (buf) {
        fs.writeFileSync(item.file + '.part', buf);
        fs.renameSync(item.file + '.part', item.file);
        bytes += buf.length;
      } else {
        fs.appendFileSync(failFile, `${item.id} ${item.url}\n`);
      }
    } catch (e) {
      console.log('durduruldu:', e.message);
      idx = todo.length;
      return;
    }
    count++;
    if (count % 500 === 0) console.log(count, '/', todo.length, Math.round(bytes / 1e6), 'MB');
    await sleep(50);
  }
}

await Promise.all([worker(), worker(), worker(), worker()]);
console.log('bitti, bu calistirmada:', count, 'toplam MB:', Math.round(bytes / 1e6));
