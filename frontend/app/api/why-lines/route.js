import { fallbackNote, validateNotes } from '../../lib/matchNotes.mjs';

const BACKEND = (process.env.MATRIX_BACKEND_URL || 'http://127.0.0.1:8100').replace(/\/$/, '');

function paperTitles(value, limit = 10) {
  return (Array.isArray(value) ? value : [])
    .map((paper) => String(typeof paper === 'string' ? paper : paper?.Title || paper?.title || '').trim())
    .filter(Boolean)
    .slice(0, limit);
}

export async function POST(request) {
  let body;
  try { body = await request.json(); } catch { return Response.json({ error: 'Invalid request' }, { status: 400 }); }
  if (!body || typeof body !== 'object') return Response.json({ error: 'Invalid request' }, { status: 400 });
  const candidates = Array.isArray(body.candidates) ? body.candidates.slice(0, 8).filter((c) => c && c.author_id) : [];
  const query = String(body.query || '').trim().slice(0, 2000);
  if (!query || !candidates.length) return Response.json({ error: 'query and candidates are required' }, { status: 400 });
  const intent = body.intent === 'mentor' ? 'mentor' : 'collaborator';
  const seekerPapers = paperTitles(body.seeker_papers, 10);
  const teamNames = Array.isArray(body.team_people) ? body.team_people.map((person) => person?.name).filter(Boolean) : [];
  const noteOptions = { intent, query, teamNames };
  const fallback = (results = candidates.map((candidate) => fallbackNote(candidate, noteOptions))) =>
    Response.json({
      context_basis: seekerPapers.length && teamNames.length ? 'need_profile_team' : seekerPapers.length ? 'need_profile' : teamNames.length ? 'need_team' : 'need',
      team_count: teamNames.length,
      source: 'fallback',
      results,
    });
  try {
    const response = await fetch(`${BACKEND}/api/why-notes`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        intent,
        candidates,
        seeker_papers: seekerPapers,
        team_member_ids: Array.isArray(body.team_member_ids)
          ? body.team_member_ids.map((id) => String(id)).filter(Boolean).slice(0, 25)
          : [],
        team_people: Array.isArray(body.team_people)
          ? body.team_people.map((person) => ({
              ...person,
              author_id: String(person?.author_id ?? person?.authorId ?? '').trim(),
            })).filter((person) => person.author_id)
          : [],
      }),
      signal: AbortSignal.any([request.signal, AbortSignal.timeout(20000)]),
      cache: 'no-store',
    });
    if (!response.ok) return fallback();
    const payload = await response.json();
    const results = validateNotes(candidates, payload, noteOptions);
    const teamCount = Number(payload.team_count) || teamNames.length;
    const profile = seekerPapers.length > 0;
    return Response.json({
      results,
      context_basis: payload.context_basis || (profile ? (teamCount ? 'need_profile_team' : 'need_profile') : (teamCount ? 'need_team' : 'need')),
      team_count: teamCount,
      source: results.every((row) => row.source === 'llm') ? 'llm' : results.every((row) => row.source === 'fallback') ? 'fallback' : 'mixed',
    });
  } catch {
    return fallback();
  }
}
