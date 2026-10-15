// Keep all six workflows available, personalizing only from enabled chat context.
export function contextStarters({ choices = {}, people = [], personIds = [], linked = false, profileName = '' } = {}) {
  const quote = value => JSON.stringify(String(value || '').trim());
  const personal = choices.personalContext || {};
  const interests = personal.useInterests !== false ? String(personal.interests || '').trim() : '';
  const selectedPeople = people.filter(p => personIds.some(id => String(id) === String(p.authorId))).slice(0, 8);
  const papers = choices.publicationScope === 'selected' ? (choices.selectedPapers || []).filter(p => p.title).slice(0, 8) : [];
  const contacts = (personal.contacts || []).filter(p => p.included && p.name).slice(0, 20);
  const sources = [];
  const context = [];
  if (interests) { sources.push('enabled interests'); context.push(`My research interests are ${quote(interests)}.`); }
  if (papers.length) { sources.push('selected papers'); context.push(`Use these selected papers as context: ${papers.map(p => quote(p.title)).join('; ')}. Do not assume I authored them.`); }
  if (selectedPeople.length) { sources.push('selected people'); context.push(`The researchers selected for this chat are ${selectedPeople.map(p => quote(p.name)).join(', ')}.`); }
  if (contacts.length) { sources.push('personal connections'); context.push(`My self-reported connections include ${contacts.map(p => quote(p.name)).join(', ')}. Do not assume collaboration, availability, or a verified graph identity.`); }
  if (linked && profileName && choices.publicationScope !== 'selected') { sources.push('linked profile'); context.push(`My linked graph profile is ${quote(profileName)}. Use only eligible publications within this chat's filters, if available.`); }
  if (choices.publicationScope === 'selected' && !papers.length) context.push('No publications are selected as my personal context. Do not infer my background from publications.');
  const basis = sources.length ? `Based on ${sources.join(', ')}` : 'Start with your goals';
  const prefix = context.length ? `${context.join(' ')}\n\n` : '';
  const targeted = interests || papers.length;
  const workflows = [
    ['Find a mentor', 'mentor', 'mentor', `Help me find a potential mentor.${targeted ? ' Use my enabled research context to identify a learning direction, and ask what I want to learn if unclear.' : ' First ask what I want to learn and what relevant experience I have.'} Support recommendations with publications; do not assume availability.`],
    ['Build a team', 'collaborator', 'team', `Help me build a research team.${targeted ? ' Use my enabled context to suggest complementary roles, then clarify the project and expertise needed.' : ' First ask about my proposed project and the expertise needed.'} Recommend people for distinct roles with supporting evidence.`],
    ['Find a niche', 'collaborator', 'sparkle', `Help me identify a promising research niche.${targeted ? ' Start from the themes in my enabled context.' : ' First ask which topics I want to explore.'} Explain possible gaps and adjacent directions, and distinguish supported evidence from speculation about how crowded a field is.`],
    ['Explore', 'collaborator', 'sparkle', selectedPeople.length || (linked && profileName) ? `Explore the recorded coauthor network of ${selectedPeople.length ? selectedPeople.map(p => quote(p.name)).join(', ') : quote(profileName)}. Show up to 10 coauthors with supporting shared publications, and help me filter by date or affiliation. Respect my publication-context selection.` : 'Help me choose a researcher, then explore their recorded coauthors with supporting shared publications. Let me narrow the results by date or affiliation. Do not assume I have a graph profile.'],
    ['Collaborate', 'collaborator', 'team', `Find potential collaborators for a proposed study.${targeted ? ' Ground the search in my enabled research context and clarify the study aim if needed.' : ' First ask me for the study topic and goal.'} Explain each person’s relevance using publications rather than assuming an existing relationship.`],
    ['Promote', 'collaborator', 'sparkle', `Help me identify a relevant research audience for a paper or project.${papers.length ? ' Start with the selected papers, then ask which work I want to share.' : ' First ask what paper or project I want to share.'} Find up to four researchers and explain their relevance with different supporting publications where possible. Do not assume I authored any selected paper or contact anyone.`],
  ];
  return workflows.map(([label,intent,icon,prompt])=>({label,intent,icon,basis,prompt:prefix+prompt}));
}
