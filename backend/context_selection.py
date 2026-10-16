"""Explicit, live publication references for each conversation."""
from paper_library import title_key


def resolve_references(library, references=(), legacy_titles=(), author_id='', filters=()):
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
    rows = library.selected_publications(refs[:8],filters)
    return [{'author_id': str(p['author_id']), 'work_id': p['work_id'], 'title': p['title'], 'year': p.get('year')}
            for p in rows]


def revalidate_session(session, library):
    if not session: return session
    state = dict(session.get('state') or {})
    choices = dict(state.get('contextChoices') or {})
    owner = str(session.get('aid') or session.get('focal_author_id') or '')
    if library is not None: owner = library.canonical_author(owner)
    own_filters = [f for f in (choices.get('publicationFilters') or []) if isinstance(f,dict) and
        (library.canonical_author(str(f.get('author_id',''))) if library is not None else str(f.get('author_id',''))) == owner]
    filters = [{'author_id':str(f['author_id']),'affiliation':f.get('affiliation','all')}
        for f in own_filters[:25]
        if isinstance(f,dict) and str(f.get('author_id','')).isdigit() and len(str(f['author_id']))<=20
        and isinstance(f.get('affiliation','all'),str) and len(f.get('affiliation','all'))<=600]
    discarded_filter = len(filters) != len(own_filters)
    valid_filters = []
    for item in filters:
        try:
            if library is not None:
                library.publication_condition(library.canonical_author(item['author_id']),item['affiliation'])
            valid_filters.append(item)
        except ValueError:
            discarded_filter = True
    own_refs = [p for p in (choices.get('selectedPapers') or []) if isinstance(p,dict) and
        (library.canonical_author(str(p.get('author_id',''))) if library is not None else str(p.get('author_id',''))) == owner]
    selected = resolve_references(library,own_refs,
                                  choices.get('paperTitles') or [] if choices.get('paperScope') == 'chosen' else [],
                                  str(session.get('aid') or session.get('focal_author_id') or ''),filters)
    scope = choices.get('publicationScope') or ('selected' if choices.get('selectedPapers') or choices.get('paperTitles') else 'eligible')
    if scope not in {'eligible','selected'}: scope = 'selected' if selected else 'eligible'
    if discarded_filter:
        selected, scope = [], 'selected'
    personal = choices.get('personalContext') or {}
    topics = list(dict.fromkeys(t for t in personal.get('topics',[]) if isinstance(t,str) and t.strip() and len(t)<=200))[:8]
    state['contextChoices'] = {'personalContext': {'topics':topics,'interests':'; '.join(topics),
                                'useInterests':personal.get('useInterests') is not False,'contacts':[]},'version': 3, 'samePlace': False, 'recentYears': 0,
                               'paperScope': 'chosen', 'paperTitles': [], 'selectedPapers': selected,
                               'publicationScope':scope,'publicationFilters':valid_filters}
    # Keep historical conversation-only notes available for analysis without
    # making them active interests or editable connections in the new UI.
    legacy = choices.get('legacyPersonalContext')
    if legacy is None and 'topics' not in personal and (personal.get('interests') or personal.get('contacts')):
        legacy = personal
    if legacy is not None:
        state['contextChoices']['legacyPersonalContext'] = legacy
    return {**session, 'state': state}
