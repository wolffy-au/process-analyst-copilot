from typing import Type, TypeVar, ClassVar, Optional, Any
import json
import logging
import os
from pydantic import BaseModel, model_validator

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.WARNING),
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class SearchParameters(BaseModel):
    q: str
    type: str
    num: int
    engine: str


class OrganicResult(BaseModel):
    title: str
    link: str
    snippet: str
    position: int
    sitelinks: Optional[list[dict[str, Any]]] = None  # For optional fields


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
    credits: int
