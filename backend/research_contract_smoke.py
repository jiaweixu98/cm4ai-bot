"""Lightweight contract checks for MATRIX research tools and answer publication."""

from research_agent import AnswerBlock, AnswerPart, ResearchAnswer, ShortlistEntry, render_answer
from research_tools import ResearchTools


PEOPLE = {
    "1": {"name": "Alex One", "affiliation": "North", "papers": [{"title": "Alpha methods for health data"}]},
    "2": {"name": "Blair Two", "affiliation": "Central", "papers": [{"title": "Alpha and beta clinical systems"}]},
    "3": {"name": "Casey Three", "affiliation": "South", "papers": [{"title": "Beta methods in hospitals"}]},
    "4": {"name": "Dana Four", "affiliation": "East", "papers": [{"title": "Alpha methods for health data"}]},
}


def details(author_id):
    return PEOPLE.get(str(author_id))


def search(query, _scope, _excluded):
    if query == "alpha":
        return [{"author_id": "1"}, {"author_id": "2"}]
    if query == "beta":
        return [{"author_id": "2"}, {"author_id": "3"}]
    return []


def answer_for(tools, *, uncited=False):
    by_author = {record["author_id"]: evidence_id for evidence_id, record in tools.evidence.items()
                 if record["source"] == "local_catalog"}
    # Deliberately reverse model order. Publication must retain service order.
    entries = [
        ShortlistEntry(author_id=author_id, why=f"{PEOPLE[author_id]['name']} covers the requested topic.",
                       evidence_ids=[] if uncited else [by_author[author_id]])
        for author_id in ("3", "1", "2")
    ]
    return ResearchAnswer(
        blocks=[AnswerBlock(kind="general", person_ids=[],
                            parts=[AnswerPart(text="These results cover both requirements.", evidence_ids=[])])],
        shortlist=entries,
        shortlist_kind="researchers",
        shortlist_title="alpha and beta",
        result_update="keep",
        task_goal="Find researchers spanning alpha and beta",
        task_requirements=["alpha", "beta"],
        suggested_followups=[],
    )


def main():
    tools = ResearchTools(lambda *_: [], details, search)
    result = tools.search_people("Find researchers spanning alpha and beta", ["alpha", "beta"], "all", [])
    assert [person["author_id"] for person in result["people"]] == ["2", "1", "3"]
    assert result["ranking"] == "rrf-v1"
    assert result["people"][0]["facet_ranks"] == {"alpha": 2, "beta": 1}

    rendered = render_answer(answer_for(tools), tools)
    assert [person["author_id"] for person in rendered["shortlist"]] == ["2", "1", "3"]
    assert rendered["result_update"] == "replace"
    assert rendered["working_context"]["requirements"] == ["alpha", "beta"]

    uncited = render_answer(answer_for(tools, uncited=True), tools)
    assert "shortlist" not in uncited, "uncited cards must not be published"

    unsupported = ResearchAnswer(
        blocks=[
            AnswerBlock(kind="research", person_ids=["2"],
                        parts=[AnswerPart(text="Blair Two proved a clinical outcome.", evidence_ids=[])]),
            AnswerBlock(kind="general", person_ids=[],
                        parts=[AnswerPart(text="I need stronger evidence for the outcome claim.", evidence_ids=[])]),
        ],
        shortlist=[], shortlist_kind="researchers", shortlist_title="",
        result_update="keep", task_goal="Assess Blair Two", task_requirements=[], suggested_followups=[],
    )
    narrowed = render_answer(unsupported, tools)
    assert "proved a clinical outcome" not in narrowed["reply"]
    assert "stronger evidence" in narrowed["reply"]

    identity = ResearchAnswer(
        blocks=[AnswerBlock(kind="research", person_ids=["2"], parts=[
            AnswerPart(text="Blair Two is at Central.", evidence_ids=["profile:2"])])],
        shortlist=[], shortlist_kind="researchers", shortlist_title="", result_update="replace",
        task_goal="", task_requirements=[], suggested_followups=["Compare Blair Two and Alex One"])
    fresh = ResearchTools(lambda *_: [], details, search)
    fresh.read_person_evidence("2", "")
    shown = render_answer(identity, fresh)
    assert "at Central" in shown["reply"] and shown["result_update"] == "keep"
    assert shown["suggested_followups"] == ["Compare Blair Two and Alex One"]
    team_tools = ResearchTools(lambda *_: [], details, search)
    team = team_tools.assemble_team("alpha-beta project", ["alpha", "beta"], "all", [])
    assert [m["author_id"] for m in team["members_in_role_order"]] == ["1", "2"]
    assert team["members_in_role_order"][1]["team_role"] == "beta"
    team_answer = answer_for(team_tools)
    team_answer.shortlist = [e for e in team_answer.shortlist if e.author_id in {"1", "2"}]
    team_cards = render_answer(team_answer, team_tools)["shortlist"]
    assert [(c["author_id"], c["role"]) for c in team_cards] == [("1", "alpha"), ("2", "beta")]

    own = ResearchTools(lambda *_: [], details, search, self_id="2")
    assert "2" not in [p["author_id"] for p in own.search_people("q", ["alpha", "beta"], "all", [])["people"]]
    own.read_person_evidence("2", "")
    own.read_person_evidence("1", "")
    paper = {r["author_id"]: e for e, r in own.evidence.items() if r["source"] == "local_catalog"}
    yours = ResearchAnswer(
        blocks=[AnswerBlock(kind="general", person_ids=[], parts=[
            AnswerPart(text="Your clinical systems work is a base.", evidence_ids=[paper["2"]]),
            AnswerPart(text="Alex One studies alpha.", evidence_ids=[paper["1"]])])],
        shortlist=[], shortlist_kind="researchers", shortlist_title="", result_update="keep",
        task_goal="", task_requirements=[], suggested_followups=[])
    reply = render_answer(yours, own)["reply"]
    assert "Your clinical systems" in reply and "Alex One studies" not in reply
    titled = yours.model_copy(deep=True)
    titled.blocks = [AnswerBlock(kind="research", person_ids=["1"], parts=[
        AnswerPart(text="**Alpha methods niche**", evidence_ids=[]),
        AnswerPart(text="- Alex One studies alpha.", evidence_ids=[paper["1"]]),
        AnswerPart(text="Alex One proved alpha works.", evidence_ids=[])])]
    reply = render_answer(titled, own)["reply"]
    assert "**Alpha methods niche**" in reply and "proved" not in reply
    titled.blocks.append(AnswerBlock(kind="research", person_ids=["1"], parts=[
        AnswerPart(text="**Lonely niche**", evidence_ids=[]),
        AnswerPart(text="Alex One proved beta works.", evidence_ids=[])]))
    assert "Lonely niche" not in render_answer(titled, own)["reply"]

    shared = ResearchTools(lambda *_: [], details, search)
    shared.read_person_evidence("1", "")
    shared.read_person_evidence("4", "")
    ids = [e for e, r in shared.evidence.items() if r["source"] == "local_catalog"]
    both = ResearchAnswer(
        blocks=[AnswerBlock(kind="research", person_ids=["1", "4"], parts=[
            AnswerPart(text="Alex One and Dana Four coauthored alpha methods.", evidence_ids=ids)])],
        shortlist=[], shortlist_kind="researchers", shortlist_title="", result_update="keep",
        task_goal="", task_requirements=[], suggested_followups=[])
    shown = render_answer(both, shared)
    assert len(shown["citations"]) == 1 and shown["reply"].count("[1]") == 1, shown["reply"]

    niche = ResearchTools(lambda *_: [], details, search)
    niche.openalex_requests = 99
    near = niche.analyze_niche(["alpha"], None)
    assert near["status"] == "partial" and near["catalog_researchers_near_niche"]
    pick = near["catalog_researchers_near_niche"][0]
    niche_answer = ResearchAnswer(
        blocks=[AnswerBlock(kind="general", person_ids=[], parts=[
            AnswerPart(text="**Alpha niche**", evidence_ids=[]),
            AnswerPart(text="- Why it fits you: it extends your methods work.", evidence_ids=[])])],
        shortlist=[ShortlistEntry(author_id=pick["author_id"], why="Near the alpha niche.",
                                  evidence_ids=[pick["papers"][0]["evidence_id"]])],
        shortlist_kind="researchers", shortlist_title="People near these niches", result_update="keep",
        task_goal="", task_requirements=[], suggested_followups=[])
    shown = render_answer(niche_answer, niche)
    assert shown["shortlist"][0]["role"] == "alpha" and shown["result_update"] == "replace"
    assert "Why it fits you" in shown["reply"] and "[" not in shown["reply"]
    niche_answer.blocks = [AnswerBlock(kind="general", person_ids=[], parts=[
        AnswerPart(text="**Alpha niche** Why it fits: it extends your methods work.", evidence_ids=[]),
        AnswerPart(text="Recent work: none listed.", evidence_ids=[])])]
    reply = render_answer(niche_answer, niche)["reply"]
    assert reply == "**Alpha niche**\n- Why it fits: it extends your methods work.\n- Recent work: none listed.", reply

    mine = ResearchTools(lambda *_: [], details, search, self_id="2")
    mine.read_person_evidence("2", "")
    own_paper = next(e for e, r in mine.evidence.items() if r["source"] == "local_catalog")
    about_me = ResearchAnswer(
        blocks=[AnswerBlock(kind="general", person_ids=[], parts=[
            AnswerPart(text="Your clinical systems work is a base.", evidence_ids=[own_paper])])],
        shortlist=[], shortlist_kind="researchers", shortlist_title="", result_update="keep",
        task_goal="", task_requirements=[], suggested_followups=[])
    shown = render_answer(about_me, mine)
    assert shown["reply"] == "Your clinical systems work is a base." and not shown["citations"], shown

    import research_skills
    assert set(research_skills.SKILLS) >= {"find_mentor", "build_team", "find_niche"}
    print("research contract smoke: ok")


if __name__ == "__main__":
    main()
