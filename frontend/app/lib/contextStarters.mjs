// Editable researcher questions. Evidence and selection rules travel in context_choices,
// context_person_ids and the linked viewer ID, never as appended prompt instructions.
export function contextStarters({ choices = {}, people = [], personIds = [], linked = false, profileName = '' } = {}) {
  const personal = choices.personalContext || {};
  const interests = personal.useInterests !== false ? String(personal.interests || '').trim().slice(0, 600) : '';
  const selectedPeople = people.filter(p => personIds.some(id => String(id) === String(p.authorId))).slice(0, 8);
  const papers = choices.publicationScope === 'selected' ? (choices.selectedPapers || []).filter(p => p.title).slice(0, 8) : [];
  const hasProfile = linked && choices.publicationScope !== 'selected';
  const basis = interests ? 'Based on your interests' : papers.length ? 'With your selected papers'
    : hasProfile ? 'With your research profile' : 'Example question';
  const selectedName = selectedPeople.length === 1 ? String(selectedPeople[0].name || '').slice(0, 120) : '';
  const explore = linked ? 'Who are my top 10 recorded coauthors? Which publications do we share?'
    : selectedPeople.length
    ? `Show up to 10 recorded coauthors ${selectedName ? `of ${selectedName}` : 'of the researchers selected for this chat'}. Which publications do they share?`
    : "I'd like to explore a researcher's recorded coauthors. Can you help me choose a researcher?";
  const workflows = [
    ['Find a mentor', 'mentor', 'mentor', interests
      ? `I want to learn more about ${interests}. Which researchers and papers could I learn from?`
      : "I'm a wet-lab biologist moving into AI-ready data standards. Who could I learn from?"],
    ['Build a team', 'collaborator', 'team', interests
      ? `I'm planning a study on ${interests}. What complementary expertise would a team need, and who could contribute?`
      : "I'm planning a study to deploy a sepsis early-warning model across hospitals. Who could help with the clinical, machine learning, and implementation work?"],
    ['Find a niche', 'collaborator', 'sparkle', interests
      ? `What promising research directions could I explore at the intersection of ${interests}?`
      : 'I work on federated learning with EHR data. What promising research directions could I explore?'],
    ['Explore', 'collaborator', 'sparkle', explore],
    ['Collaborate', 'collaborator', 'team', interests
      ? `I'm planning a study on ${interests}. Which Bridge2AI researchers have relevant work I could build on?`
      : "I'm planning a study combining clinical phenotyping and responsible AI. Which Bridge2AI researchers could I collaborate with?"],
    ['Promote', 'collaborator', 'sparkle', papers.length
      ? "Who would be a relevant research audience for the first paper I've selected? Please suggest up to four researchers."
      : `I'm proposing a paper on ${interests || 'pathway enrichment and gene-expression networks for interpreting disease-associated molecular phenotypes'}. Who would be a relevant research audience? Please suggest up to four researchers.`],
  ];
  return workflows.map(([label,intent,icon,prompt]) => ({label,intent,icon,basis:label === 'Explore'
    ? linked ? `For ${String(profileName || 'your linked profile').slice(0, 120)}` : selectedPeople.length ? 'With your selected researchers' : 'Choose a researcher'
    : label === 'Promote' && papers.length ? 'With your selected paper' : basis,prompt}));
}
