"""Deterministic paper-audience and recorded-coauthor discovery.

People are ordered by retrieved publication evidence, never by the language model.
Geography uses current ORCID or owner-approved affiliations, resolved through
sourced ROR locations.
"""
import geography as geo
from paper_library import title_key
from tkg_publications import normalise_institution

_MARK_CHUNK = 400
_FETCH_ROWS = 2000
_SECOND_PAPER_WINDOW = 200
_INITIALISM_STOP = {'of', 'the', 'and', 'at', 'for', 'in'}


def _chunks(values, size=_MARK_CHUNK):
    values = list(values)
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _work_ids_for_rowids(library, rowids):
    found = {}
    for chunk in _chunks(rowids):
        marks = ','.join('?' for _ in chunk)
        found.update(library._db().execute(f'SELECT rowid,work_id FROM papers WHERE rowid IN ({marks})', chunk).fetchall())
    return found


def _existing_works(library, works):
    present = set()
    for chunk in _chunks(works):
        marks = ','.join('?' for _ in chunk)
        present.update(r[0] for r in library._db().execute(f'SELECT work_id FROM papers WHERE work_id IN ({marks})', chunk))
    return present


def audience(library, vectors, identifier='', title='', abstract='', exclude_ids=(), limit=8):
    source = (library.work(identifier) or library.by_doi(identifier) or library.by_title(identifier)) if identifier else None
    if source:
        title, abstract = source['title'], source.get('abstract') or ''
    elif title.strip():
        # A pasted title of a paper already in the catalog: its authors are not its audience.
        source = library.by_title(title)
        if source and not abstract.strip():
            abstract = source.get('abstract') or ''
    if not title.strip() and not abstract.strip():
        return {'status': 'needs_paper', 'people': []}
    query = f'{title.strip()} {abstract.strip()}'[:12000]
    excluded = {int(x) for x in exclude_ids if str(x).isdigit()}
    if source:
        excluded.update(int(aid) for aid, _ in library.authors(source['work_id']))
    ranks = {}
    lexical = sorted(library.score_query(query).items(), key=lambda x: (-x[1], x[0]))[:_FETCH_ROWS]
    works = _work_ids_for_rowids(library, [rowid for rowid, _ in lexical])
    for rank, (rowid, _) in enumerate(lexical, 1):
        if rowid in works:
            ranks[works[rowid]] = 1 / (60 + rank)
    method = 'publication_bm25'
    if vectors is not None:
        import numpy as np
        row = vectors.row_of.get(source['work_id']) if source else None
        vector = vectors.matrix[row] if row is not None else vectors.embed(query)
        _, indices = vectors.index.search(np.ascontiguousarray(vector, dtype=np.float32).reshape(1, -1),
                                          min(_FETCH_ROWS, vectors.index.ntotal))
        hits = [(rank, vectors.ids[int(index)]) for rank, index in enumerate(indices[0], 1) if index >= 0]
        present = _existing_works(library, [work for _, work in hits])
        for rank, work in hits:
            if work in present:
                ranks[work] = ranks.get(work, 0) + 1 / (60 + rank)
        method = 'publication_rrf_bm25_specter2'
    limit = max(1, min(limit, 20))
    people, extra = {}, 0
    source_title = title_key(title)
    # Order is by descending fused score and a person keeps their best paper's score, so
    # once `limit` people are collected nobody later can outrank them. Keep reading a
    # short while only to give those people a second supporting paper.
    for work, score in sorted(ranks.items(), key=lambda x: (-x[1], x[0])):
        if len(people) >= limit:
            if all(len(p['matched_papers']) >= 2 for p in people.values()) or extra >= _SECOND_PAPER_WINDOW:
                break
            extra += 1
        paper = library.work(work)
        if not paper or title_key(paper['title']) == source_title:
            continue
        for aid, _ in library.authors(work):
            aid = int(aid)
            if aid in excluded:
                continue
            candidate = people.get(aid)
            if candidate is None:
                if len(people) >= limit:
                    continue
                candidate = people[aid] = {'author_id': str(aid), 'score': score, 'matched_papers': []}
            if len(candidate['matched_papers']) < 2 and not any(
                    title_key(p['title']) == title_key(paper['title']) for p in candidate['matched_papers']):
                candidate['matched_papers'].append(paper)
    ordered = sorted(people.values(), key=lambda p: (-p['score'], int(p['author_id'])))[:limit]
    return {'status': 'ok' if ordered else 'empty', 'people': ordered, 'method': method,
            'source_paper': source or {'title': title, 'abstract': abstract, 'source': 'user_supplied'},
            'excluded_source_authors': sorted(excluded), 'snapshot_version': library.snapshot_version}


def _institution_matches(wanted, institution):
    """Punctuation and spacing do not matter; a short all-capitals name may be initials
    (UCSF, MIT, NIH) of the printed institution."""
    text = normalise_institution(institution)
    key = normalise_institution(wanted)
    if not key or not text:
        return False
    if key in text:
        return True
    raw = str(wanted).strip()
    if raw.isalpha() and raw.isupper() and 2 < len(raw) <= 8:
        initials = ''.join(word[0] for word in text.split() if word not in _INITIALISM_STOP)
        return key in initials
    return False


def _affiliations(library, pairs):
    """{(author_id, work_id): [{institution, ror_id}]} for the given shared papers."""
    wanted = set(pairs)
    found = {}
    for chunk in _chunks({work for _, work in wanted}):
        marks = ','.join('?' for _ in chunk)
        for row in library._db().execute(f'''SELECT af.author_id,af.work_id,g.institution,g.ror_id
                FROM publication_affiliations af JOIN publication_groups g USING(author_id,group_id)
                WHERE af.work_id IN ({marks})''', chunk):
            key = (int(row[0]), row[1])
            if key in wanted:
                found.setdefault(key, []).append({'institution': row[2], 'ror_id': row[3]})
    return found


def explore_people(library, author_id='', topics=(), specialty='', institution='', geography='',
                   from_year=None, to_year=None, coauthors_only=False, offset=0, limit=500,
                   locations=None, affiliation_lookup=None):
    """One filtering contract for the Graph and MATRIX. All publication constraints
    apply to the same retained paper; geography uses current trusted affiliations."""
    if from_year is not None and to_year is not None and from_year > to_year:
        raise ValueError('The start year must be before the end year')
    revision = library.correction_revision()
    cache_key = (revision,str(author_id),tuple(topics),specialty,institution,geography,from_year,to_year,coauthors_only,locations,affiliation_lookup)
    with library._explore_lock:
        cached = library._explore_cache.get(cache_key)
        if cached is not None:
            library._explore_cache.move_to_end(cache_key)
            return {**cached, 'people':cached['people'][offset:offset+limit], 'offset':offset, 'limit':limit, 'has_more':offset+limit<cached['total']}
    conditions, args = [], []
    if coauthors_only:
        if not str(author_id).isdigit(): raise ValueError('Choose a focal researcher for recorded coauthors')
        conditions.append('EXISTS (SELECT 1 FROM visible_paper_authors a WHERE a.work_id=b.work_id AND a.author_id=?)')
        args.append(int(author_id))
        conditions.append('b.author_id!=?'); args.append(int(author_id))
    if from_year is not None: conditions.append('p.year>=?'); args.append(int(from_year))
    if to_year is not None: conditions.append('p.year<=?'); args.append(int(to_year))
    terms, weights = [], {}
    matched_works = None
    if specialty.strip():
        scores = library.score_query(specialty)
        matched_works = set(_work_ids_for_rowids(library, scores).values())
        terms = library.query_terms(specialty)
        weights = {term: library.term_weight(term) for term in terms}
    wanted_topics = [str(t).strip().casefold() for t in topics if str(t).strip()]
    if wanted_topics:
        conditions.append('lower(p.primary_topic) IN (' + ','.join('?' for _ in wanted_topics) + ')'); args.extend(wanted_topics)
    scope_sql, scope_args = ' AND '.join(conditions) or '1', list(args)
    if terms:
        expression = ' OR '.join('"' + t.replace('"','') + '"' for t in terms)
        conditions.append('p.rowid IN (SELECT rowid FROM papers_fts WHERE papers_fts MATCH ?)'); args.append(expression)
    clauses = ' AND '.join(conditions) or '1'
    columns = 'p.*' if terms else 'p.work_id,p.title,p.year,p.venue,p.doi,p.pmid,p.cited_by,p.primary_topic,NULL AS abstract'
    people = {}
    rows = library._db().execute(f"""SELECT b.author_id,{columns} FROM visible_paper_authors b
        JOIN papers p USING(work_id) WHERE {clauses} ORDER BY p.year DESC,p.work_id,b.author_id""", args)
    for record in rows:
        paper = dict(record); aid = int(paper.pop('author_id'))
        if matched_works is not None and paper['work_id'] not in matched_works: continue
        text = paper['title'] + ' ' + (paper.get('abstract') or '')
        if terms:
            covered = set(library._covered(terms, text))
            ranked = sorted(terms, key=lambda term: weights[term], reverse=True)
            needed = ranked[:max(1, (len(ranked) + 1) // 2)]
            if any(term not in covered for term in needed): continue
        person = people.setdefault(aid, {'author_id': str(aid), 'matched_papers': [], 'matching_paper_count': 0})
        person['matching_paper_count'] += 1
        if len(person['matched_papers']) < 3: person['matched_papers'].append(paper)
    ordered = sorted(people.values(), key=lambda p: (-p['matching_paper_count'], -(p['matched_papers'][0]['year'] or 0), int(p['author_id'])))
    nearby_topics = []
    if specialty.strip() and not ordered:
        nearby_topics = [row[0] for row in library._db().execute(
            f"""SELECT p.primary_topic FROM visible_paper_authors b JOIN papers p USING(work_id)
                WHERE {scope_sql} AND COALESCE(p.primary_topic,'')!=''
                GROUP BY p.primary_topic ORDER BY COUNT(DISTINCT b.author_id) DESC, p.primary_topic LIMIT 3""",
            scope_args)]
    if institution or geography:
        for person in ordered:
            affiliations = affiliation_lookup(person['author_id']) if affiliation_lookup else library.affiliations(person['author_id'])
            person['institutions'] = [a for a in affiliations if a.get('trusted') and a.get('is_current') and not a.get('end_year')]
        if institution:
            ordered = [p for p in ordered if any(_institution_matches(institution, a['institution']) for a in p['institutions'])]
    report = {}
    if geography.strip():
        ordered, report = _filter_geography(ordered, geography, locations if locations is not None else geo.default())
    for person in ordered:
        if coauthors_only: person['shared_paper_count'] = person['matching_paper_count']
        person.pop('institutions', None)
    total = len(ordered)
    result = {'status': 'ok' if ordered else 'empty', 'people': ordered, 'total': total,
            'offset': offset, 'limit': limit, 'has_more': offset+limit < total,
            'filters': dict(author_id=author_id,topics=list(topics),specialty=specialty,institution=institution,geography=geography,
                            from_year=from_year,to_year=to_year,coauthors_only=coauthors_only),
            'nearby_topics': nearby_topics,
            'geography_basis': 'current_trusted_affiliations',
            'geography_complete': report.get('complete', True), 'geography_checked_people': report.get('checked', 0),
            'geography_unresolved_people': report.get('unresolved', 0),
            **({'geography_matched_people': total, 'geography_interpretation': report['interpretation'],
                'geography_locations_version': report['version']} if report else {}),
            'snapshot_version': library.snapshot_version,
            'revision': revision}
    if library.correction_revision() != revision: raise ValueError('The review state changed; try the filters again')
    with library._explore_lock:
        library._explore_cache[cache_key] = result
        while len(library._explore_cache)>3: library._explore_cache.popitem(last=False)
    return {**result,'people':ordered[offset:offset+limit]}


def explore(library, author_id, specialty='', institution='', geography='', from_year=None, to_year=None,
            limit=8, locations=None, affiliation_lookup=None):
    return explore_people(library,author_id,specialty=specialty,institution=institution,geography=geography,
                          from_year=from_year,to_year=to_year,coauthors_only=True,limit=max(1,min(limit,20)),
                          locations=locations,affiliation_lookup=affiliation_lookup)


def _filter_geography(ordered, typed, places):
    """Keep people whose current trusted institution (by ROR location) matches the typed place."""
    parts = places.parse(typed)
    if not parts:
        return ordered, {}
    rors = [a['ror_id'] for p in ordered for a in p['institutions'] if a['ror_id']]
    found, lookups_complete = places.resolve(rors)
    kept, checked, unresolved, interpretation = [], 0, 0, {}
    for person in ordered:
        evidence, gap, seen = [], not person['institutions'], set()
        for a in person['institutions']:
            key = geo.ror_key(a['ror_id'])
            row = found.get(key)
            if row is None:
                gap = True  # no ROR, or a ROR that cannot be placed: this institution is unresolved
                continue
            matched = places.match(row, parts)
            if matched and key not in seen:
                seen.add(key)
                evidence.append({**{k: row.get(k) for k in ('country_code', 'country_name', 'subdivision_name', 'city', 'continent_name')},
                                 'institution': a['institution'], 'matched_on': sorted({m['field'] for m in matched}),
                                 'source': f'https://ror.org/{key}'})
        if evidence:
            person['location_evidence'] = evidence
            kept.append(person)
            checked += 1
            for item in evidence:
                for field in item['matched_on']:
                    interpretation[field] = interpretation.get(field, 0) + 1
        elif gap:
            unresolved += 1
        else:
            checked += 1
    return kept, {'complete': lookups_complete and unresolved == 0, 'checked': checked, 'unresolved': unresolved,
                  'interpretation': {'typed': typed, 'parts': parts, 'matched_fields': interpretation},
                  'version': places.version}
