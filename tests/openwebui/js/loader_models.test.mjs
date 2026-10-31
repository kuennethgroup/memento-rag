// The browser script's model and library rules for two libraries in one Open WebUI. The pure block between the
// "memento:models" markers of openwebui/loader.js runs here in Node: which copy the selected model belongs to (by its
// exact id), its links, its library, and the start-screen picker's list cache per library. Run: node
// tests/openwebui/js/loader_models.test.mjs (tests/library/test_two_libraries_loader.py).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const src = readFileSync(new URL('../../../openwebui/loader.js', import.meta.url), 'utf8');
const start = src.indexOf('/* memento:models-begin');
const end = src.indexOf('/* memento:models-end */');
assert.ok(start >= 0 && end > start, 'the pure block is marked in loader.js');
const ctx = {};
vm.runInNewContext(src.slice(start, end) + '\nthis.M = MementoModels;', ctx);
const M = ctx.M;

const today = (kind, id) => ({ id, kind, collection: 'memento-comp-mat-rag',
  siblings: { query: 'memento-query', upload: 'memento-ingestion-prod', doc: 'memento-document-chat' } });
const study = (kind, id) => ({ id, kind, collection: 'memento-polymer-rag',
  siblings: { query: 'memento-query-polymer', upload: 'memento-ingestion-polymer', doc: 'memento-document-chat-polymer' } });
const MAP = { models: [], copies: {
  memento_query: [today('query', 'memento-query'), today('query', 'memento-query-qwen35'),
                  today('doc', 'memento-document-chat'), today('upload', 'memento-ingestion-prod')],
  memento_query_polymer: [study('query', 'memento-query-polymer'), study('query', 'memento-query-polymer-qwen35'),
                          study('doc', 'memento-document-chat-polymer'), study('upload', 'memento-ingestion-polymer')] } };
// both copies' Library Chat models have the same names: only the id tells them apart
for (const list of Object.values(MAP.copies)) for (const e of list) e.name = e.kind === 'doc' ? 'Document Chat' : 'Library Chat';

const tests = [];
const test = (name, fn) => tests.push([name, fn]);

test('the selected model comes from OpenWebUI\'s own value', () => {
  const store = (v) => ({ getItem: (k) => (k === 'selectedModels' ? v : null) });
  assert.equal(M.selectedId(store('["memento-query-polymer"]')), 'memento-query-polymer');
  assert.equal(M.selectedId(store('["",""]')), null);
  assert.equal(M.selectedId(store('not json')), null);
  assert.equal(M.selectedId(store(null)), null);
  assert.equal(M.selectedId(null), null);
});

test('same-named entries resolve by their exact id', () => {
  assert.equal(M.entryFor(MAP, 'memento-query-polymer').collection, 'memento-polymer-rag');
  assert.equal(M.entryFor(MAP, 'memento-query').collection, 'memento-comp-mat-rag');
  assert.equal(M.entryFor(MAP, 'Library Chat'), null);              // never by name
  assert.equal(M.entryFor(MAP, 'memento-query-polymer-x'), null);
});

test('each copy links its own models', () => {
  assert.equal(M.linksFor(MAP, 'memento-document-chat-polymer').query, 'memento-query-polymer');
  assert.equal(M.linksFor(MAP, 'memento-query-qwen35').upload, 'memento-ingestion-prod');
  const href = M.relink('/?model=memento-ingestion-prod', M.linksFor(MAP, 'memento-query-polymer'));
  assert.equal(href, '/?model=memento-ingestion-polymer');
  assert.equal(M.relink('/?model=memento-query&q=x', M.linksFor(MAP, 'memento-ingestion-polymer')),
               '/?model=memento-query-polymer&q=x');
  assert.equal(M.relink('/?model=memento-query', M.linksFor(MAP, 'memento-query')), '/?model=memento-query');
});

test('without the map: today\'s exact ids keep today\'s links, a study id waits', () => {
  for (const id of ['memento-query', 'memento-query-qwen35', 'memento-document-chat', 'memento-ingestion-prod', null])
    assert.deepEqual({ ...M.linksFor(null, id) }, { query: 'memento-query', upload: 'memento-ingestion-prod',
                                                    doc: 'memento-document-chat' });
  for (const id of ['memento-query-polymer', 'memento-document-chat-polymer', 'memento-query-polymer-qwen35'])
    assert.equal(M.linksFor(null, id), null);
  assert.equal(M.linksFor(MAP, 'memento-query-unknown'), null);       // a map without the id: waits too
});

test('the picker reads the selected copy\'s library', () => {
  assert.equal(M.libraryUrl(M.collectionFor(MAP, 'memento-query-polymer')),
               '/files/library?collection=memento-polymer-rag');
  assert.equal(M.libraryUrl(M.collectionFor(null, 'memento-query')), '/files/library');     // today: the old file
});

test('the picker waits for a study model the map does not know, never reading today\'s list (R2)', () => {
  const st = (map, id) => JSON.parse(JSON.stringify(M.pickerLibrary(map, id)));
  assert.deepEqual(st(null, 'memento-query-polymer'), { wait: true, collection: null });
  assert.deepEqual(st(MAP, 'memento-query-polymer'), { wait: false, collection: 'memento-polymer-rag' });
  assert.deepEqual(st(null, 'memento-query'), { wait: false, collection: null });          // today's old list
  assert.deepEqual(st(MAP, 'memento-query'), { wait: false, collection: 'memento-comp-mat-rag' });
  assert.deepEqual(st(null, null), { wait: false, collection: null });                    // nothing selected yet
});

test('with no model stored yet, the page is about the only copy the user may use (live check, 15 Oct)', () => {
  // OpenWebUI clears its stored choice on a fresh page: the selected model is unknown until a message or a change
  const STUDY = ['memento-query-polymer', 'memento-query-polymer-qwen35', 'memento-document-chat-polymer'];
  const TODAY = ['memento-query', 'memento-query-qwen35', 'memento-document-chat'];
  assert.equal(M.effectiveId('memento-query-qwen35', MAP, STUDY), 'memento-query-qwen35');   // a stored choice wins
  assert.equal(M.effectiveId(null, MAP, STUDY), 'memento-query-polymer');
  assert.equal(M.effectiveId(null, MAP, TODAY), 'memento-query');
  assert.equal(M.effectiveId(null, MAP, [...STUDY, ...TODAY]), null);            // an admin: today's, as before
  assert.equal(M.effectiveId(null, null, STUDY), M.PENDING);                      // the map not read yet: wait
  assert.equal(M.effectiveId(null, MAP, null), M.PENDING);                        // the user's models not read yet
  assert.equal(JSON.parse(JSON.stringify(M.pickerLibrary(MAP, M.PENDING))).wait, true);
  assert.equal(M.linksFor(MAP, M.PENDING), null);                                 // links wait too
  assert.equal(M.pickerLibrary(MAP, M.effectiveId(null, MAP, STUDY)).collection, 'memento-polymer-rag');
});

test('the kind of a model: from the map, else today\'s ids', () => {
  assert.equal(M.kindOf(MAP, 'memento-ingestion-polymer'), 'ingest');
  assert.equal(M.kindOf(MAP, 'memento-document-chat-polymer'), 'doc');
  assert.equal(M.kindOf(MAP, 'memento-query-polymer-qwen35'), 'query');
  assert.equal(M.kindOf(null, 'memento-ingestion-prod'), 'ingest');
  assert.equal(M.kindOf(null, 'memento-document-chat'), 'doc');
  assert.equal(M.kindOf(null, 'memento-query-qwen36'), 'query');
  assert.equal(M.kindOf(null, 'something-else'), null);
});

test('a switch to another library within 10 s loads that library\'s list', () => {
  const c = M.pickerCache(10000);
  assert.ok(c.shouldLoad('A', 0));
  c.begin('A');
  assert.ok(!c.shouldLoad('A', 100));                                 // one request at a time per library
  c.done('A', [{ source: 'a' }], 100);
  assert.ok(!c.shouldLoad('A', 5000));
  assert.ok(c.shouldLoad('B', 5000));                                 // B never read: loaded although A is fresh
  assert.equal(c.docs('B'), null);
  assert.ok(c.shouldLoad('A', 10101));                                // stale after 10 s
});

test('a late answer for the other library never replaces the shown list', () => {
  const c = M.pickerCache(10000);
  c.begin('A');
  c.begin('B');
  c.done('B', [{ source: 'b' }], 10);
  c.done('A', [{ source: 'a' }], 20);                                 // A answers after B
  assert.deepEqual([...c.docs('B').map((d) => d.source)], ['b']);
  assert.deepEqual([...c.docs('A').map((d) => d.source)], ['a']);
});

test('a failed read shows an empty list only while the picker waits, and only where nothing was read yet', () => {
  const c = M.pickerCache(10000);
  c.begin('A');
  c.fail('A', false);                                                 // no picker open: still unknown, read again later
  assert.equal(c.docs('A'), null);
  assert.ok(c.shouldLoad('A', 1));
  c.begin('A');
  c.fail('A', true);                                                  // the picker waits: say empty
  assert.deepEqual([...c.docs('A')], []);
  c.done('B', [{ source: 'b' }], 1);
  c.begin('B');
  c.fail('B', true);
  assert.deepEqual([...c.docs('B').map((d) => d.source)], ['b']);
});

test('an open picker closes when the model changes to another library or copy', () => {
  assert.equal(M.pickerStale('memento-query|memento-comp-mat-rag', 'memento-query-polymer|memento-polymer-rag'), true);
  assert.equal(M.pickerStale('memento-query|memento-comp-mat-rag', 'memento-query|memento-comp-mat-rag'), false);
  assert.equal(M.pickerStale(null, 'x'), false);                       // no picker open
});

let failed = 0;
for (const [name, fn] of tests) {
  try { fn(); console.log('ok   ' + name); } catch (e) { failed++; console.log('FAIL ' + name + '\n     ' + e.message); }
}
console.log(`${tests.length - failed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
