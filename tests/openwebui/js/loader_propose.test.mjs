// Which rows of a research-gaps report's question table get "Draft a proposal". The block between the
// "memento:propose" markers of openwebui/loader.js runs here: the table headed "# | Question", rows whose question
// has an "Evidence for question N" block; older reports (no such table) get null.
// Run: node tests/openwebui/js/loader_propose.test.mjs (tests/research_gaps/test_gap_report.py).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const src = readFileSync(new URL('../../../openwebui/loader.js', import.meta.url), 'utf8');
const i = src.indexOf('/* memento:propose-begin'), j = src.indexOf('/* memento:propose-end */');
assert.ok(i >= 0 && j > i, 'memento:propose marked in loader.js');
const ctx = {};
vm.createContext(ctx);
vm.runInContext(src.slice(i, j), ctx);
const rows = (toggles, tables) => JSON.parse(JSON.stringify(ctx.mementoProposeRows(toggles, tables)));

const questions = { head: ['#', 'Question', 'Library', 'Literature'], rows: ['1', '2', '3'] };
const evidence = ['Evidence for question 1', ' Evidence for  question 2 ', 'Evidence for question 3'];

assert.deepEqual(rows(evidence, [questions]), { table: 0, rows: [[0, '1'], [1, '2'], [2, '3']] }, 'every question row');
assert.deepEqual(rows(evidence.slice(0, 2), [questions]), { table: 0, rows: [[0, '1'], [1, '2']] },
  'a question still streaming in: no button yet');
assert.deepEqual(rows(['Evidence for question 11', 'Evidence for question 12'],
  [{ ...questions, rows: ['11', '12'] }]), { table: 0, rows: [[0, '11'], [1, '12']] }, 'a later page keeps its numbers');
const byDoc = { head: ['Document', 'Questions'], rows: ['Thesis A'] };
assert.deepEqual(rows(evidence, [byDoc, questions]), { table: 1, rows: [[0, '1'], [1, '2'], [2, '3']] },
  'the question table, not another one');
assert.deepEqual(rows(['Evidence for question 1', 'Further open questions (4)'], [questions]),
  { table: 0, rows: [[0, '1']] }, 'only questions shown in full');
assert.equal(ctx.mementoProposeRows([], [questions]), null, 'no evidence blocks: the older layout');
assert.equal(ctx.mementoProposeRows(evidence, [byDoc]), null, 'no question table: the older layout');
console.log('loader_propose: ok');
