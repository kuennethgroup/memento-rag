// The library map and the user's models after signing in. The block between the "memento:load" markers of
// openwebui/loader.js runs here with a fake fetch, clock and storage: a try made before signing in (the login page
// gets 401) does not hold back the first try after it, and a failed try while signed in is repeated by a timer, not
// only when the page changes.
// Run: node tests/openwebui/js/loader_load.test.mjs (tests/library/test_two_libraries_loader.py).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const src = readFileSync(new URL('../../../openwebui/loader.js', import.meta.url), 'utf8');
const i = src.indexOf('/* memento:load-begin'), j = src.indexOf('/* memento:load-end */');
assert.ok(i >= 0 && j > i, 'memento:load marked in loader.js');

const OK = (body) => ({ ok: true, status: 200, body });
const MAP = { models: [{ id: 'memento-query' }] }, MINE = { data: [{ id: 'memento-query' }] };
const make = () => {
  const env = { now: 1000000, token: null, calls: [], timers: [], reply: {} };
  const ctx = {
    Date: { now: () => env.now },
    window: { localStorage: { getItem: (k) => (k === 'token' ? env.token : null) } },
    setTimeout: (f, ms) => { env.timers.push({ f, at: env.now + ms }); return env.timers.length; },
    MementoModels: { state: { map: null, mine: null } },
    fetch: (url) => {
      env.calls.push(url);
      const r = env.reply[url] || { ok: false, status: 401 };
      return Promise.resolve({ ok: r.ok, status: r.status, json: () => Promise.resolve(r.body) });
    },
  };
  vm.createContext(ctx);
  vm.runInContext(src.slice(i, j), ctx);
  return { env, ctx };
};
const settle = () => new Promise((r) => setImmediate(r));
const runTimers = async (env) => {
  const due = env.timers.splice(0);
  if (due.length) env.now = Math.max(env.now, ...due.map((t) => t.at));
  due.forEach((t) => t.f());
  await settle();
};

// the login page: both reads fail (no session) and nothing polls; a second paint within 10 s reads nothing
{
  const { env, ctx } = make();
  ctx.mementoLoadMap(); await settle();
  assert.deepEqual([...env.calls].sort(), ['/api/models', '/files/models']);
  assert.equal(env.timers.length, 0, 'no timer without a session');
  ctx.mementoLoadMap(); await settle();
  assert.equal(env.calls.length, 2, 'no new read within 10 s while the session is the same');
  // signed in 3 s later: the next paint reads both at once, and the map's listeners repaint
  env.now += 3000; env.token = 'session-1';
  env.reply = { '/files/models': OK(MAP), '/api/models': OK(MINE) };
  let painted = 0;
  ctx.mementoMapListeners.push(() => { painted += 1; });
  ctx.mementoLoadMap(); await settle();
  assert.equal(env.calls.length, 4, 'signed in: read again at once');
  assert.deepEqual(ctx.MementoModels.state.map, MAP);
  assert.deepEqual([...ctx.MementoModels.state.mine], ['memento-query']);
  assert.ok(painted >= 1, 'the listeners repaint');
}
// signed in, the map read fails once: a timer reads it again 10 s later, without a page change
{
  const { env, ctx } = make();
  env.token = 'session-1';
  env.reply = { '/api/models': OK(MINE), '/files/models': { ok: false, status: 502 } };
  ctx.mementoLoadMap(); await settle();
  assert.equal(ctx.MementoModels.state.map, null);
  assert.equal(env.timers.length, 1, 'one retry scheduled');
  env.reply['/files/models'] = OK(MAP);
  await runTimers(env);
  assert.deepEqual(ctx.MementoModels.state.map, MAP, 'the retry loaded the map');
  assert.equal(env.timers.length, 0, 'nothing more scheduled once loaded');
}
// an expired session's token on the login page, a new one after signing in: read again at once
{
  const { env, ctx } = make();
  env.token = 'expired';
  ctx.mementoLoadMap(); await settle();
  const n = env.calls.length;
  env.timers.splice(0);
  env.now += 2000; env.token = 'session-2';
  env.reply = { '/files/models': OK(MAP), '/api/models': OK(MINE) };
  ctx.mementoLoadMap(); await settle();
  assert.ok(env.calls.length > n, 'a new session reads at once');
  assert.deepEqual(ctx.MementoModels.state.map, MAP);
}
console.log('loader_load: ok');
