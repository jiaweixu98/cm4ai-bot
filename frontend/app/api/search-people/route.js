const BACKEND = (
  process.env.MATRIX_BACKEND_URL ||
  "http://127.0.0.1:8100"
).replace(/\/$/, "");

function asCandidate(row) {
  const authorId = String(row?.author_id ?? row?.authorId ?? "").trim();
  if (!authorId) return null;
  return {
    author_id: authorId,
    name: String(row?.name || "").trim() || `Researcher ${authorId}`,
    affiliation: String(row?.affiliation || "").trim() || "Affiliation unavailable",
    is_bridge2ai_member: Boolean(row?.is_bridge2ai_member ?? row?.isBridge2AiMember),
    retrieval_score: Number(row?.retrieval_score ?? row?.score ?? row?.similarity ?? 0) || 0,
    hops: row?.hops,
    mutual_coauthors: Array.isArray(row?.mutual_coauthors) ? row.mutual_coauthors : [],
    papers: Array.isArray(row?.papers) ? row.papers.slice(0, 3) : [],
  };
}

async function enrichPapers(candidate) {
  if (candidate.papers.length > 0) return candidate;
  try {
    const response = await fetch(`${BACKEND}/api/author/${encodeURIComponent(candidate.author_id)}/details`, {
      cache: "no-store",
    });
    if (!response.ok) return candidate;
    const details = await response.json();
    const papers = Array.isArray(details?.papers) ? details.papers.slice(0, 3) : [];
    return { ...candidate, papers };
  } catch {
    return candidate;
  }
}

async function proxyJson(path, body, headers = {}) {
  const response = await fetch(`${BACKEND}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...headers },
    body: JSON.stringify(body),
    cache: "no-store",
  });
  const payload = await response.json().catch(() => ({}));
  return { ok: response.ok, status: response.status, payload };
}

export async function POST(request) {
  try {
    const body = await request.json();
    const query = String(body.query || "").trim();
    const aid = String(body.aid || "unlinked");
    const topK = Math.max(1, Math.min(10, Number(body.top_k) || 8));
    const bridge2aiOnly = Boolean(body.bridge2ai_only);
    const teamMemberIds = Array.isArray(body.team_member_ids)
      ? body.team_member_ids.map(String).filter(Boolean).slice(0, 25)
      : [];
    if (!query) {
      return Response.json({ error: "query is required" }, { status: 400 });
    }

    let rows = [];
    const titles = Array.isArray(body.representative_titles) ? body.representative_titles.filter((t) => typeof t === 'string' && t.trim()).slice(0, 10).map((t) => t.trim().slice(0, 500)) : [];
    const paperSearch = bridge2aiOnly && titles.length > 0;
    if (paperSearch) {
      const token = process.env.BRIDGE_INTERNAL_API_TOKEN;
      const { ok, status, payload } = await proxyJson('/api/author-preview', {
        full_name: 'Researcher', affiliation: '', papers: titles.map((title) => ({ title })), top_k: topK + (/^\d+$/.test(aid) ? 1 : 0), bridge2ai_only: true,
      }, token ? { 'x-bridge-api-token': token } : {});
      if (!ok) return Response.json({ error: 'Paper-based mentor search is unavailable' }, { status });
      rows = (payload.nearest_authors || []).filter((person) => String(person.author_id) !== aid);
    } else if (bridge2aiOnly) {
      const { ok, status, payload } = await proxyJson(
        "/api/search",
        {
          aid,
          query,
          top_k: topK,
          bridge2ai_only: true,
          outside_network: false,
          team_member_ids: teamMemberIds,
        },
      );
      if (!ok) {
        return Response.json({ error: "Mentor search is unavailable" }, { status: status || 502 });
      }
      rows = payload.candidates || [];
    } else {
      const { ok, status, payload } = await proxyJson("/api/search-outside-network", {
        aid,
        query,
        top_k: topK,
        team_member_ids: teamMemberIds,
      });
      if (!ok) {
        return Response.json({ error: "Collaborator search is unavailable" }, { status: status || 502 });
      }
      rows = payload.candidates || [];
    }

    const candidates = (
      await Promise.all((Array.isArray(rows) ? rows : []).map(asCandidate).filter(Boolean).map(enrichPapers))
    ).slice(0, topK);
    return Response.json({ candidates: candidates.map((candidate) => ({ ...candidate, search_basis: paperSearch ? 'representative_papers' : 'research_need' })), total: candidates.length });
  } catch (error) {
    console.error("search-people failed:", error);
    return Response.json({ error: "Search failed" }, { status: 500 });
  }
}
