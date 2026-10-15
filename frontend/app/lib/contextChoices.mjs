export const publicationKey = paper => `${paper.author_id}:${paper.work_id}`;

// An empty explicit selection must stay empty. Only the dedicated action opts
// back into all eligible publications; unchecking a paper must not broaden scope.
export function selectPublications(choices, selectedPapers) {
  return { ...choices, version: 3, paperScope: 'chosen', publicationScope: 'selected', paperTitles: [], selectedPapers: selectedPapers.slice(0, 8) };
}

export function removePublication(choices, paper) {
  return selectPublications(choices, (choices.selectedPapers || []).filter(other => publicationKey(other) !== publicationKey(paper)));
}

export function useEligiblePublications(choices) {
  return { ...choices, version: 3, paperScope: 'chosen', publicationScope: 'eligible', selectedPapers: [], paperTitles: [] };
}

export function contextChoicesForRequest(choices) {
  return {
    research_interests: choices.personalContext?.useInterests === false ? '' : (choices.personalContext?.interests || ''),
    known_people: (choices.personalContext?.contacts || []).filter(p => p.included).map(p => ({name:p.name, relationship:p.relationship, note:p.note, author_id:p.authorId || null})),
    same_place: choices.samePlace,
    recent_years: choices.recentYears,
    paper_scope: choices.paperScope,
    publication_scope: choices.publicationScope || ((choices.selectedPapers || []).length ? 'selected' : 'eligible'),
    publication_filters: choices.publicationFilters || [],
    paper_titles: [],
    selected_publications: (choices.selectedPapers || []).slice(0, 8).map(p => ({ author_id: String(p.author_id), work_id: p.work_id })),
  };
}
