"""Skills: playbooks for recurring research goals, embedded in the agent instructions.

Users never pick a skill; the agent recognizes the goal from the conversation.
Skills describe how to combine tools; they never add data or claims.
"""

SKILLS = {
    "find_mentor": {
        "title": "Find a mentor",
        "summary": "Someone wants to learn a skill or move into a field and needs people to learn from.",
        "tools": ["search_people", "get_author_info", "read_person_evidence"],
        "playbook": """
1. Identify what the learner wants to learn and, if stated, where they come from (their
   current field or profile_context). The gap between the two is the requirement set.
   Ask one question only if the target skill itself is unclear.
2. Call search_people with the original question and up to 3 requirements: the target
   skill first, then any bridging need (e.g. "machine learning applied to wet-lab data").
3. For the top 3 results, call get_author_info in parallel to add publication record
   facts (works count, h-index, recent output, topics). Report only returned numbers.
4. shortlist_kind = mentors. Each why-line: what this person's papers show about the
   target skill, citing their papers. Order is the search order.
5. Blocks: one framing sentence, then a concrete learning path: which 1-2 cited papers
   to read first and what to ask each person about. Do not claim anyone teaches,
   supervises, is available, or is senior beyond what the returned numbers show.
""",
    },
    "build_team": {
        "title": "Build a team",
        "summary": "Assemble or complete a research team with complementary roles for a project.",
        "tools": ["assemble_team", "read_context", "get_connection", "get_author_info"],
        "playbook": """
1. Decompose the project into 2-5 distinct, concrete roles (capabilities), e.g. for an
   ML early-warning deployment: "clinical sepsis outcomes research", "EHR machine
   learning prediction models", "implementation science in hospitals". Skip roles the
   user says they already cover. If the user has a profile, call read_context once to
   see their own work and avoid duplicating it.
2. Call assemble_team with the goal and the roles. It returns one member per role in
   role order (deterministic), alternates, and which members also match other roles.
3. Put every returned member in shortlist, in members_in_role_order order; the why-line
   names the role they fill and what their cited papers show. shortlist_kind =
   collaborators.
4. Blocks: the cards already show each member and role, so do not list members again.
   Write one framing sentence naming the roles covered, name any role with no
   candidate, mention a member who also matches another role (also_matches_roles),
   and give the most useful next step.
5. Optional: if the user asks whether members already know each other, call
   get_connection for the pairs asked about. Coauthor exclusion only on request.
""",
    },
    "find_niche": {
        "title": "Find a niche",
        "summary": "Explore under-explored research intersections or a niche for a researcher.",
        "tools": ["analyze_niche", "search_literature", "search_people", "get_author_info"],
        "playbook": """
1. Identify the user's strengths (from the request or profile_context via read_context)
   and the directions they are curious about. Form 2-3 candidate intersections, each a
   combination of 2-3 short concept phrases (e.g. ["federated learning", "voice
   biomarkers"]).
2. Call analyze_niche once per candidate intersection, in parallel (max 3). Each call
   returns OpenAlex counts of works matching every concept, the per-year trend,
   recent highly cited joint works and catalog researchers near it. If the status is
   partial (no counts), use the same layout without "Evidence:" and "Recent work:"
   and do not mention counts, crowding, trends, the literature service or what could
   not be verified.
3. Compare candidates with plain counts: joint_works (small = less crowded) and the
   last-3-years versus previous-3-years counts (growing or not). Cite each
   analyze_niche evidence_id for its numbers, e.g. "28 works since 2016 matched both
   concepts; 17 of them in the last three years". No ratios or percentages.
4. Layout, all in general blocks with no person_ids:
   - A first block of 1-2 sentences on the strength the niches build on, uncited.
   - One block for each niche you analyzed (present all 2-3): a short bold topic title ("**Federated causal inference for
     EHRs**", no "A narrower direction:" prefix, uncited), then list items:
     "Evidence:" the counts and trend, citing the analyze_niche evidence_id;
     "Why it fits you:" your reasoning tied to the user's work, uncited;
     "Recent work:" 1-2 papers from most_cited_recent_joint_works (the most-cited
     works of the last three years, i.e. what is gaining attention), cited.
   - A closing sentence recommending where to start, uncited, then one short question
     asking which direction the user wants to grow into and whether they want people
     to work with, people to learn from, or papers to read next.
   Say "less crowded" or "growing", never "unexplored".
5. People: put 1-2 catalog_researchers_near_niche per niche in shortlist (kind
   researchers, title like "People working near these niches"). Each why says whether
   they already work inside the niche (their titles combine its concepts, see
   title_coverage) or on one side of it, and what the user gets from them: a potential
   collaborator or mentor in the niche, or the methods/domain side to read. Name the
   specific paper topic. The cards show them, so the blocks do not name them.
   suggested_followups: the niche directions as user choices (e.g. "Go deeper on
   federated causal inference").
6. If the user picks a niche, offer search_people for collaborators or
   search_literature for a reading list.
""",
    },
    "check_author": {
        "title": "Check an author",
        "summary": "Look up a specific researcher's record: affiliation, output, impact, topics and papers.",
        "tools": ["resolve_person", "get_author_info", "read_person_evidence"],
        "playbook": """
1. If you have a catalog author_id, call get_author_info with it. Otherwise call
   get_author_info with the name; if it reports several catalog people, ask which one,
   listing their affiliations.
2. If the user asks about fit for a topic, also call read_person_evidence with the
   topic to get their most relevant catalog papers.
3. Report the facts returned: affiliation, works count, citations, h-index, recent
   yearly output, main topics, ORCID when present, and the relevant papers. Cite the
   author-profile evidence_id for numbers and paper evidence_ids for papers. State
   identity_basis only if it is not shared_publication (e.g. matched by institution).
4. If OpenAlex is ambiguous or unavailable, answer from the catalog record.
""",
    },
    "check_paper": {
        "title": "Check a paper",
        "summary": "Look up a paper by DOI, OpenAlex ID, citation or title: authors, venue, citations, abstract.",
        "tools": ["get_paper_info", "get_author_info"],
        "playbook": """
1. Call get_paper_info with the DOI, OpenAlex ID, evidence_id, or the full title.
2. Report title, year, venue, citation count, authors with institutions and the main
   findings from the abstract (only if an abstract was returned; otherwise topic only).
   If match_basis is closest_title, say which paper you found and ask if it is the one.
3. catalog_people are authors confirmed in the MATRIX catalog (the paper appears in
   their profile); name them with person_ids so the user can open their profiles.
4. If the user asks about an author of the paper, continue with get_author_info.
""",
    },
}


def playbooks() -> str:
    return "\n\n".join(f"## {skill['title']}: {skill['summary']}\n{skill['playbook'].strip()}"
                       for skill in SKILLS.values())
