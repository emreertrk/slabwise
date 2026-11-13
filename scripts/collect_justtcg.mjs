import { JustTCG } from 'justtcg-js';
import fs from 'node:fs';
import path from 'node:path';

const client = new JustTCG();
const dir = 'data';
const rawDir = path.join(dir, 'raw');
fs.mkdirSync(rawDir, { recursive: true });

const maxRun = Number(process.argv[2] ?? 50);
const stateFile = path.join(dir, 'state.json');
const setsFile = path.join(dir, 'sets.json');

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const load = (f, d) => (fs.existsSync(f) ? JSON.parse(fs.readFileSync(f, 'utf8')) : d);
const save = (f, o) => fs.writeFileSync(f, JSON.stringify(o));

const shuffle = (a) => {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
};

let used = 0;
let remaining = 100;

async function getSets() {
  let sets = load(setsFile, null);
  if (sets) return sets;
  const res = await client.v1.sets.list({ game: 'pokemon' });
  sets = res.data;
  save(setsFile, sets);
  remaining = res.usage?.apiDailyRequestsRemaining ?? remaining;
  used += 1;
  return sets;
}

function buildPlan(sets) {
  const rounds = [[], [], [], []];
  for (const s of sets) {
    const n = s.cards_count ?? s.count ?? 0;
    if (n <= 0) continue;
    const pages = Math.ceil(n / 20);
    const k = Math.min(4, Math.max(1, Math.round(pages * 0.25)));
    const picks = shuffle([...Array(pages).keys()]).slice(0, k);
    picks.forEach((p, i) => rounds[i].push({ set: s.id, offset: p * 20 }));
  }
  return rounds.flatMap((r) => shuffle(r));
}

const sets = await getSets();
let state = load(stateFile, null);
if (!state) {
  state = { plan: buildPlan(sets), done: 0 };
  save(stateFile, state);
}
state.skipped = state.skipped ?? [];
console.log('plan:', state.plan.length, 'yapilan:', state.done);

let consecutiveErrors = 0;

while (state.done < state.plan.length && used < maxRun && remaining > 10) {
  const item = state.plan[state.done];
  try {
    const res = await client.v1.cards.get({
      game: 'pokemon',
      set: item.set,
      limit: 20,
      offset: item.offset
    });
    save(path.join(rawDir, `${item.set}__${item.offset}.json`), res);
    remaining = res.usage?.apiDailyRequestsRemaining ?? remaining - 1;
    used += 1;
    state.done += 1;
    consecutiveErrors = 0;
    save(stateFile, state);
    console.log(state.done, '/', state.plan.length, item.set, item.offset, 'kalan:', remaining);
  } catch (e) {
    consecutiveErrors += 1;
    console.log('hata:', item.set, item.offset, e.message, e.status ?? e.code ?? '');
    if (consecutiveErrors >= 3) break;
    state.skipped.push(item);
    state.done += 1;
    save(stateFile, state);
  }
  await sleep(7000);
}
console.log('bitti. bu calistirmada istek:', used, 'atlanan toplam:', state.skipped.length);