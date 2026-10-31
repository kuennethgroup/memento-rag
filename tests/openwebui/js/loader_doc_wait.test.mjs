// A Document Chat suggestion clicked while its file still uploads starts once the upload is done. The block between
// the "memento:doc-tip" markers of openwebui/loader.js runs here; the wiring is checked in the source.
// Run: node tests/openwebui/js/loader_doc_wait.test.mjs (tests/document_chat/test_doc_chat_file_limit.py).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const src = readFileSync(new URL('../../../openwebui/loader.js', import.meta.url), 'utf8');
const i = src.indexOf('/* memento:doc-tip-begin'), j = src.indexOf('/* memento:doc-tip-end */');
assert.ok(i >= 0 && j > i, 'memento:doc-tip marked in loader.js');
const ctx = {};
vm.createContext(ctx);
vm.runInContext(src.slice(i, j) + '\nthis.docWaitReady = docWaitReady; this.DOC_WAIT_TIP = DOC_WAIT_TIP;', ctx);
const { docWaitReady, DOC_WAIT_TIP } = ctx;

const w = { label: null, at: 0 };
assert.equal(docWaitReady(w, 1, 1, 0), null, 'nothing waits');
w.label = 'Summarize the file'; w.at = 1000;
assert.equal(docWaitReady(w, 1, 1, 2000), null, 'still uploading: it waits');
assert.equal(w.label, 'Summarize the file', 'and stays queued');
assert.equal(docWaitReady(w, 1, 0, 3000), 'Summarize the file', 'uploaded: the waiting click goes');
assert.equal(w.label, null, 'once');
assert.equal(docWaitReady(w, 1, 0, 3500), null, 'only once');
w.label = 'Main findings'; w.at = 1000;
assert.equal(docWaitReady(w, 0, 0, 2000), null, 'file removed: forgotten');
assert.equal(w.label, null);
w.label = 'Methods and data'; w.at = 1000;
assert.equal(docWaitReady(w, 1, 1, 1000 + 121000), null, 'after 2 minutes: forgotten');
assert.equal(w.label, null);
assert.match(DOC_WAIT_TIP, /upload/i, 'the queued button says it waits for the upload');

// wiring: a click on a button dimmed for an upload waits; paintStart lets it go when the upload is done
assert.match(src, /dataset\.memUploading === '1'\)\s*\{[^}]*docWait\.label = /, 'a click during an upload waits');
assert.match(src, /docWaitReady\(docWait, chips\.length, uploading, Date\.now\(\)\)/, 'paintStart releases it');
assert.match(src, /if \(!desc \|\| v !== 'doc'\) docWait\.label = null;/, 'leaving the start screen forgets it');
console.log('loader_doc_wait: ok');
