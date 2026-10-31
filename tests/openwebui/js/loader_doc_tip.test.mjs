// The dimmed Document Chat buttons' tip. The block between the
// "memento:doc-tip" markers of openwebui/loader.js runs here: no file, a file still uploading, files ready.
// Run: node tests/openwebui/js/loader_doc_tip.test.mjs (tests/document_chat/test_doc_chat_file_limit.py).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const src = readFileSync(new URL('../../../openwebui/loader.js', import.meta.url), 'utf8');
const i = src.indexOf('/* memento:doc-tip-begin'), j = src.indexOf('/* memento:doc-tip-end */');
assert.ok(i >= 0 && j > i, 'memento:doc-tip marked in loader.js');
const ctx = {};
vm.createContext(ctx);
vm.runInContext(src.slice(i, j) + '\nthis.docTip = docTip;', ctx);
const { docTip } = ctx;

assert.match(docTip(0, 0), /^Attach a file first/, 'no file: attach one');
assert.equal(docTip(1, 1), 'Your file is still uploading', 'one file uploading');
assert.equal(docTip(3, 2), 'Your files are still uploading', 'several uploading');
assert.equal(docTip(2, 1), 'Your file is still uploading', 'one of two still uploading');
assert.equal(docTip(1, 0), null, 'uploaded: not dimmed');
assert.equal(docTip(3, 0), null, 'all uploaded: not dimmed');
console.log('loader_doc_tip: ok');
