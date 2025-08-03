from typing import Optional, Any
from pydantic import BaseModel, BeforeValidator, Field
from typing_extensions import Annotated
import json


class SearchParameters(BaseModel):
    q: str
    type: Annotated[
        str,
        BeforeValidator(
            lambda v: (
                v if v in ["organic", "people_also_ask", "related_searches"] else None
            )
        ),
    ]
    num: Annotated[int, BeforeValidator(lambda v: v if 1 <= v <= 10 else None)]
    engine: Annotated[
        str, BeforeValidator(lambda v: v if v in ["google", "bing"] else None)
    ]


class OrganicResult(BaseModel):
    title: str
    link: str
    snippet: str
    position: int
    sitelinks: Optional[list[dict[str, Any]]] = Field(default_factory=list)


class PeopleAlsoAsk(BaseModel):
    question: str
    snippet: str
    title: str
    link: str


class RelatedSearch(BaseModel):
    query: str


class SearchResponse(BaseModel):
    searchParameters: SearchParameters
    organic: list[OrganicResult]
    peopleAlsoAsk: list[PeopleAlsoAsk]
    relatedSearches: list[RelatedSearch]
    credits: Annotated[int, Field(ge=0)]
