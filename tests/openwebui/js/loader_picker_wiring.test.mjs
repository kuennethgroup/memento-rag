// The start-screen picker's page wiring with two libraries. The real code runs here: the pure block between the
// "memento:models" markers and the picker fragment between the "memento:picker" markers of openwebui/loader.js, with a
// fake fetch and fake page hooks. A model the model map does not know yet reads no list (never the default library's)
// and its buttons do nothing; once the map knows it, its own library's list is read. The default copy's models read the
// single-library list. Run: node tests/openwebui/js/loader_picker_wiring.test.mjs
// (tests/library/test_two_libraries_loader.py).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const src = readFileSync(new URL('../../../openwebui/loader.js', import.meta.url), 'utf8');
const block = (begin, end) => {
  const i = src.indexOf(begin), j = src.indexOf(end);
  assert.ok(i >= 0 && j > i, `${begin} … ${end} marked in loader.js`);
  return src.slice(i, j);
};
const MODELS = block('/* memento:models-begin', '/* memento:models-end */');
const PICKER = block('/* memento:picker-begin', '/* memento:picker-end */');

const LISTS = {
  '/files/library': [{ source: 'comp-1', title: 'A comp-mat paper' }],
  '/files/library?collection=memento-polymer-rag': [{ source: 'poly-1', title: 'A polymer paper' }],
};
const STUDY_MAP = { models: [], copies: { memento_query_polymer: [
  { id: 'memento-query-polymer', kind: 'query', collection: 'memento-polymer-rag',
    siblings: { query: 'memento-query-polymer', upload: 'memento-ingestion-polymer', doc: 'memento-document-chat-polymer' } },
] } };

function page(selected, mine = undefined, map = undefined) {
  const seen = { fetched: [], opened: [], sent: [], topics: [], mapAsked: 0 };
  const ctx = {
    console, Date, Promise, encodeURIComponent,
    picker: null,
    paint: () => {},
    // the page's own rule (loader.js mementoSelectedId): the stored choice, else the user's only copy
    mementoSelectedId: () => ctx.MementoModels.effectiveId(selected, ctx.MementoModels.state.map,
                                                          ctx.MementoModels.state.mine),
    mementoLoadMap: () => { seen.mapAsked += 1; },
    fetch: async (url) => {
      seen.fetched.push(url);
      return { ok: url in LISTS, json: async () => ({ documents: LISTS[url] || [] }) };
    },
    closePicker: () => {},
    send: (q) => seen.sent.push(q),
  };
  vm.createContext(ctx);
  vm.runInContext(MODELS + '\n' + PICKER + '\nthis.load = load; this.onAction = onAction; this.docsNow = docsNow;', ctx);
  if (mine !== undefined) ctx.MementoModels.state.mine = mine;
  if (map !== undefined) ctx.MementoModels.state.map = map;
  ctx.openPicker = () => seen.opened.push(ctx.docsNow());
  ctx.openTopic = () => seen.topics.push(ctx.docsNow());
  return { ctx, seen };
}
// values made inside the sandbox have its own prototypes: compare plain copies
const plain = (x) => JSON.parse(JSON.stringify(x));
const PICK = { pick: true, topic: false }, ASK = { pick: false, send: () => 'List the documents' };

const tests = [];
const test = (name, fn) => tests.push([name, fn]);

test('a study model the map does not know yet reads no list and its buttons do nothing', async () => {
  const { ctx, seen } = page('memento-query-polymer');
  await ctx.load();
  assert.deepEqual(seen.fetched, []);                          // never today's old list
  ctx.onAction(PICK, {});
  ctx.onAction(ASK, {});
  ctx.onAction({ topic: true, docs: () => '' }, {});
  assert.deepEqual(plain([seen.opened, seen.sent, seen.topics]), [[], [], []]);
  assert.ok(seen.mapAsked > 0);                                 // the map is asked for again
});

test('once the map knows the study model, its own library is read and offered', async () => {
  const { ctx, seen } = page('memento-query-polymer');
  await ctx.load();
  ctx.MementoModels.state.map = STUDY_MAP;                       // the map arrives
  await ctx.load();
  assert.deepEqual(seen.fetched, ['/files/library?collection=memento-polymer-rag']);
  ctx.onAction(PICK, {});
  assert.deepEqual(plain(seen.opened), [[{ source: 'poly-1', title: 'A polymer paper' }]]);
});

test("today's model reads the old list before the map arrives, as before", async () => {
  const { ctx, seen } = page('memento-query');
  await ctx.load();
  assert.deepEqual(seen.fetched, ['/files/library']);
  ctx.onAction(PICK, {});
  ctx.onAction(ASK, {});
  assert.deepEqual(plain(seen.opened), [[{ source: 'comp-1', title: 'A comp-mat paper' }]]);
  assert.deepEqual(seen.sent, ['List the documents']);
});

const FULL_MAP = { models: [], copies: {
  memento_query: [{ id: 'memento-query', kind: 'query', collection: 'memento-comp-mat-rag',
    siblings: { query: 'memento-query', upload: 'memento-ingestion-prod', doc: 'memento-document-chat' } }],
  memento_query_polymer: STUDY_MAP.copies.memento_query_polymer } };
LISTS['/files/library?collection=memento-comp-mat-rag'] = LISTS['/files/library'];

test('a study user on a fresh page (no model stored) reads the study library, never today\'s (live check)', async () => {
  const { ctx, seen } = page(null, null, FULL_MAP);                // the user's models not read yet
  await ctx.load();
  ctx.onAction(PICK, {});
  assert.deepEqual(plain([seen.fetched, seen.opened]), [[], []]);  // waits
  ctx.MementoModels.state.mine = ['memento-query-polymer'];         // OpenWebUI says what the user may use
  await ctx.load();
  ctx.onAction(PICK, {});
  assert.deepEqual(seen.fetched, ['/files/library?collection=memento-polymer-rag']);
  assert.deepEqual(plain(seen.opened), [[{ source: 'poly-1', title: 'A polymer paper' }]]);
});

test('a default-library user on a fresh page reads the default library', async () => {
  const { ctx, seen } = page(null, ['memento-query'], FULL_MAP);
  await ctx.load();
  assert.deepEqual(seen.fetched, ['/files/library?collection=memento-comp-mat-rag']);
});

test('the page takes the selected model through the shared rule', () => {
  const i = src.indexOf('var mementoSelectedId');
  const body = src.slice(i, src.indexOf('\n};', i));
  assert.ok(i >= 0 && body.includes('MementoModels.effectiveId('), 'mementoSelectedId goes through effectiveId');
});

let failed = 0;
for (const [name, fn] of tests) {
  try { await fn(); console.log(`ok - ${name}`); } catch (e) { failed += 1; console.log(`not ok - ${name}\n${e.stack}`); }
}
process.exitCode = failed ? 1 : 0;
