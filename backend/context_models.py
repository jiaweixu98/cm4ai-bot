"""Validated conversation context, independent of catalog loading and server startup."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class SelectedPublication(BaseModel):
    author_id: str = Field(max_length=20)
    work_id: str = Field(max_length=100)


class PublicationFilter(BaseModel):
    author_id: str = Field(max_length=20, pattern=r'^\d+$')
    affiliation: str = Field(default='all',max_length=600)


class KnownPersonContext(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    relationship: Literal['met', 'collaborated', 'know', 'interested']
    note: str = Field(default='', max_length=500)
    author_id: str | None = Field(default=None, max_length=20, pattern=r'^\d+$')


class ContextChoices(BaseModel):
    """What the context panel sends into this chat, and how recommendations are limited."""

    model_config = ConfigDict(extra="ignore")

    research_interests: str = Field(default="", max_length=2000)
    known_people: list[KnownPersonContext] = Field(default_factory=list, max_length=20)
    same_place: bool = False
    recent_work: bool = False
    recent_years: Literal[0, 5, 10] = 0
    paper_scope: Literal["profile", "papers", "chosen"] = "profile"
    paper_titles: list[str] = Field(default_factory=list, max_length=8)
    selected_publications: list[SelectedPublication] = Field(default_factory=list,max_length=8)
    publication_scope: Literal['eligible','selected'] = 'eligible'
    publication_filters: list[PublicationFilter] = Field(default_factory=list,max_length=25)


