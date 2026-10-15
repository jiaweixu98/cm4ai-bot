// Derive suggestions locally from the same opt-in context used by chat requests.
// Never infer a user's expertise or relationships from a display name.
export function contextStarters({ choices = {}, people = [], personIds = [], linked = false, profileName = '' } = {}) {
  const starters = [];
  const add = (label, basis, prompt, intent = 'collaborator') => starters.push({ label, basis, prompt, intent, icon: intent === 'mentor' ? 'mentor' : 'team' });
  const quote = value => JSON.stringify(String(value || '').trim());
  const personal = choices.personalContext || {};
  const interests = personal.useInterests !== false ? String(personal.interests || '').trim() : '';
  if (interests) {
    add('Find collaborators', 'Your enabled research interests', `My research interests are ${quote(interests)}. Help me narrow a collaboration goal, then find researchers with supporting publications.`);
    add('Find a mentor', 'Your enabled research interests', `I want to learn more about ${quote(interests)}. Find potential mentors using their publications; do not assume they are available to mentor me.`, 'mentor');
  }
  const selectedPeople = people.filter(p => personIds.some(id => String(id) === String(p.authorId))).slice(0, 8);
  if (selectedPeople.length) {
    const names = selectedPeople.map(p => quote(p.name)).join(', ');
    add(selectedPeople.length > 1 ? 'Compare selected people' : 'Explore selected researcher', 'People selected for this chat', `Explore ${names} using the publications allowed in this chat. ${interests ? `Explain their relevance to my interests in ${quote(interests)}.` : 'Help me identify a research question to explore with this context.'} Distinguish evidence from speculation.`);
  }
  if (choices.publicationScope === 'selected') {
    const papers = (choices.selectedPapers || []).filter(p => p.title).slice(0, 8);
    if (papers.length) add('Explore selected papers', 'Papers selected for this chat', `Use only my selected papers (${papers.map(p => quote(p.title)).join('; ')}) to identify research themes and potential follow-up questions. Do not assume I authored them.`);
  } else if (linked && profileName) {
    add('Explore my publications', 'Your linked graph profile', `Explore the eligible publications associated with my linked graph profile, ${quote(profileName)}, respecting this chat's publication filters. Suggest research questions grounded in those papers, and tell me if none are available.`);
  }
  const contacts = (personal.contacts || []).filter(p => p.included && p.name).slice(0, 20);
  if (contacts.length) add('Explore my connections', 'Your enabled personal connections', `Help me plan research conversations with these people from my personal context: ${contacts.map(p => quote(p.name)).join(', ')}.${interests ? ` Focus on ${quote(interests)}.` : ' Ask me what I want to discuss first.'} Treat these connections as self-reported; do not assume collaboration, availability, or a graph identity.`);
  if (!starters.length) {
    add('Define a research goal', 'Start with your goals', 'Help me define a research goal. Ask about my interests and the problem I want to solve before recommending researchers.');
    add('Find a mentor', 'Start with what you want to learn', 'Help me find a potential mentor. First ask what I want to learn and what relevant experience I have.', 'mentor');
    add('Build a research team', 'Start with your project', 'Help me plan a research team. First ask about my project and the expertise I need before suggesting people.');
  }
  return starters;
}
