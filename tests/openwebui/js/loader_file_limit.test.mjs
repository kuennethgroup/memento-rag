// Document Chat's file limit in the browser. The real code runs here: the block between the "memento:file-limit"
// markers of openwebui/loader.js, on a fake page shaped like Open WebUI 0.11.4's message box (one remove button per
// attached file or image). Run: node tests/openwebui/js/loader_file_limit.test.mjs
// (tests/document_chat/test_doc_chat_file_limit.py).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const src = readFileSync(new URL('../../../openwebui/loader.js', import.meta.url), 'utf8');
const i = src.indexOf('/* memento:file-limit-begin'), j = src.indexOf('/* memento:file-limit-end */');
assert.ok(i >= 0 && j > i, 'memento:file-limit marked in loader.js');
const BLOCK = src.slice(i, j);
const REMOVE = 'div.absolute.-top-1.-right-1 > button';

class El {
  constructor(id = '') { this.id = id; this.dataset = {}; this.attrs = {}; this.children = []; this.parentNode = null; this.textContent = ''; }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  removeAttribute(k) { delete this.attrs[k]; }
  prepend(el) { el.remove(); el.parentNode = this; this.children.unshift(el); }
  remove() {
    if (!this.parentNode) return;
    this.parentNode.children = this.parentNode.children.filter((c) => c !== this);
    this.parentNode = null;
  }
}

function page(model, files) {
  const form = new El(), input = new El('chat-input'), send = new El('send-message-button');
  const state = { files };
  form.querySelectorAll = (sel) => (sel === REMOVE ? Array.from({ length: state.files }, () => new El()) : []);
  input.closest = (sel) => (sel === 'form' ? form : sel === '#chat-input' ? input : null);
  send.closest = (sel) => (sel === '#send-message-button' ? send : null);
  form.closest = () => null;
  const handlers = {};
  let observer = null;
  const document = {
    documentElement: { dataset: model ? { memModel: model } : {} },
    body: new El(),
    getElementById: (id) => ({ 'chat-input': input, 'send-message-button': send })[id]
      || form.children.find((c) => c.id === id) || null,
    createElement: () => new El(),
    addEventListener: (type, fn, capture) => { assert.equal(capture, true, `${type} listened in the capture phase`); (handlers[type] ||= []).push(fn); },
  };
  const ctx = {
    document, requestAnimationFrame: (fn) => fn(),
    MutationObserver: class { constructor(fn) { observer = fn; } observe() {} },
  };
  vm.createContext(ctx);
  vm.runInContext(BLOCK, ctx);
  const fire = (type, target, extra = {}) => {
    const ev = { type, target, key: 'Enter', shiftKey: false, isComposing: false, prevented: false, stopped: false,
      preventDefault() { this.prevented = true; }, stopImmediatePropagation() { this.stopped = true; }, ...extra };
    (handlers[type] || []).forEach((h) => h(ev));
    return ev;
  };
  return {
    state, send,
    enter: (extra) => fire('keydown', input, extra), click: () => fire('click', send), submit: () => fire('submit', form),
    repaint: () => observer && observer([]),
    note: () => form.children.find((c) => c.id === 'memento-file-limit') || null,
  };
}
const blocked = (ev) => ev.prevented && ev.stopped;

const tests = [];
const test = (name, fn) => tests.push([name, fn]);

test('six files in Document Chat: sending is blocked and a line says to remove one', () => {
  const p = page('doc', 6);
  assert.equal(p.note().textContent, 'Document Chat reads up to 5 files per chat. Please remove 1 file.');
  assert.equal(p.note().getAttribute('role'), 'status');
  assert.equal(p.send.getAttribute('aria-disabled'), 'true');
  assert.ok(blocked(p.enter()) && blocked(p.click()) && blocked(p.submit()));
  assert.ok(!blocked(p.enter({ shiftKey: true })), 'Shift+Enter still makes a new line');
});

test('five files: nothing is blocked and no line is shown', () => {
  const p = page('doc', 5);
  assert.equal(p.note(), null);
  assert.equal(p.send.getAttribute('aria-disabled'), null);
  assert.ok(!blocked(p.enter()) && !blocked(p.click()) && !blocked(p.submit()));
});

test('removing files lifts the block', () => {
  const p = page('doc', 7);
  assert.equal(p.note().textContent, 'Document Chat reads up to 5 files per chat. Please remove 2 files.');
  p.state.files = 5;
  p.repaint();
  assert.equal(p.note(), null);
  assert.equal(p.send.getAttribute('aria-disabled'), null);
  assert.ok(!blocked(p.click()));
});

test('a file added after the line is shown updates it', () => {
  const p = page('doc', 6);
  p.state.files = 8;
  p.repaint();
  assert.equal(p.note().textContent, 'Document Chat reads up to 5 files per chat. Please remove 3 files.');
});

test('Library Upload and Library Chat are never blocked', () => {
  for (const [model, n] of [['ingest', 30], ['query', 8], [null, 9]]) {
    const p = page(model, n);
    assert.equal(p.note(), null, `${model}: no line`);
    assert.ok(!blocked(p.enter()) && !blocked(p.click()), `${model}: sends`);
  }
});

let failed = 0;
for (const [name, fn] of tests) {
  try { await fn(); console.log(`ok - ${name}`); } catch (e) { failed += 1; console.log(`not ok - ${name}\n${e.stack}`); }
}
process.exitCode = failed ? 1 : 0;
