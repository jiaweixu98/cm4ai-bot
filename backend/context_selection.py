"""Explicit, live publication references for each conversation."""
from paper_library import title_key


def resolve_references(library, references=(), legacy_titles=(), author_id=''):
    if library is None: return []
    refs = list(references or [])[:8]
    # Only older explicit 'chosen' titles reach this migration. Automatic recent
    # papers never become selections. Ambiguous or hidden titles are discarded.
    if legacy_titles and str(author_id).isdigit():
        aid = library.canonical_author(author_id)
        works = library.author_works(aid)
        for title in legacy_titles[:8]:
            matches = [p for p in works if title_key(p['title']) == title_key(title)]
            if len(matches) == 1: refs.append({'author_id': aid, 'work_id': matches[0]['work_id']})
    rows = library.selected_publications(refs[:8])
    return [{'author_id': str(p['author_id']), 'work_id': p['work_id'], 'title': p['title'], 'year': p.get('year')}
            for p in rows]


def revalidate_session(session, library):
    if not session: return session
    state = dict(session.get('state') or {})
    choices = dict(state.get('contextChoices') or {})
    selected = resolve_references(library,choices.get('selectedPapers') or [],
                                  choices.get('paperTitles') or [] if choices.get('paperScope') == 'chosen' else [],
                                  str(session.get('aid') or session.get('focal_author_id') or ''))
    state['contextChoices'] = {'version': 2, 'samePlace': False, 'recentYears': 0,
                               'paperScope': 'chosen', 'paperTitles': [], 'selectedPapers': selected}
    return {**session, 'state': state}
