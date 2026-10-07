import assert from 'node:assert/strict';
import {fallbackNote, validateNotes} from '../app/lib/matchNotes.mjs';

const candidate = {author_id: '42', papers: [{Title: 'Published clinical data methods'}, {Title: 'A second publication'}]};
const explanation = 'This study describes clinical data methods relevant to the requested research question.';
assert.equal(fallbackNote(candidate, {teamNames: ['Avery']}).explanation, 'See supporting publication.', 'Fallback must not claim a team complement without an explanation');
for (const index of [undefined, null, '', ' ', -1, 3, 0.5, true]) {
  const [note] = validateNotes([candidate], {results: [{author_id: '42', explanation, evidence_paper_index: index}]});
  assert.equal(note.source, 'fallback', `Invalid index ${String(index)} must withhold the generated claim`);
  assert.notEqual(note.explanation, explanation);
}
const [valid] = validateNotes([candidate], {results: [{author_id: '42', explanation, evidence_paper_index: 1}]});
assert.equal(valid.explanation, explanation);
assert.equal(valid.evidence_paper_index, 1);
const [otherPerson] = validateNotes([candidate], {results: [{author_id: '43', explanation, evidence_paper_index: 0}]});
assert.equal(otherPerson.source, 'fallback');
console.log('PASS: notes retain valid person/paper evidence and withhold unsupported fallback or invalid citation claims.');
