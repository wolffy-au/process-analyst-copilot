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

    # implement model_validator to validate the response
    @model_validator(mode="after")
    def validate_response(
        cls: "SearchResponse", values: dict[str, Any]
    ) -> dict[str, Any]:
        search_params = values.get("searchParameters", {})

        if search_params.get("type") not in [
            "organic",
            "people_also_ask",
            "related_searches",
        ]:
            raise ValueError("Invalid search type")

        if not (1 <= search_params.get("num", 0) <= 10):
            raise ValueError("Invalid number of results")

        if search_params.get("engine") not in ["google", "bing"]:
            raise ValueError("Invalid search engine")

        if values.get("credits", 0) < 0:
            raise ValueError("Invalid credits")

        return values
