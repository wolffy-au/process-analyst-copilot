import pytest
from pydantic import ValidationError
from typing import Any
from process_analyst_copilot.data import (
    SearchResponse,
    OrganicResult,
)


# --- Pytest Fixtures ---
@pytest.fixture
def valid_response_data() -> dict[str, Any]:
    """Fixture that provides a valid search response dictionary."""
    return {
        "searchParameters": {
            "q": "python pydantic",
            "type": "organic",
            "num": 5,
            "engine": "google",
        },
        "organic": [
            {
                "title": "Pydantic",
                "link": "https://pydantic-docs.help/en/latest/",
                "snippet": "Pydantic is a data validation and settings management library...",
                "position": 1,
                "sitelinks": [
                    {
                        "title": "Documentation",
                        "link": "https://pydantic-docs.help/en/latest/",
                    }
                ],
            },
            {
                "title": "Pydantic GitHub",
                "link": "https://github.com/samuelcolvin/pydantic",
                "snippet": "Data validation using Python type hints.",
                "position": 2,
            },
        ],
        "peopleAlsoAsk": [
            {
                "question": "What is Pydantic used for?",
                "snippet": "Pydantic is used to validate data, particularly in web APIs.",
                "title": "What is Pydantic used for?",
                "link": "https://example.com/pydantic-use-cases",
            }
        ],
        "relatedSearches": [
            {"query": "pydantic fastapi"},
            {"query": "pydantic tutorial"},
        ],
        "credits": 10,
    }


# --- Test Functions ---


def test_search_response_valid_data(valid_response_data: dict[str, Any]) -> None:
    """
    Test that a valid dictionary can be successfully parsed into a SearchResponse object.
    """
    response = SearchResponse(**valid_response_data)
    assert isinstance(response, SearchResponse)
    assert response.searchParameters.q == "python pydantic"
    assert len(response.organic) == 2
    assert response.organic[0].title == "Pydantic"
    assert response.organic[0].sitelinks is not None
    assert len(response.peopleAlsoAsk) == 1
    assert response.peopleAlsoAsk[0].question == "What is Pydantic used for?"
    assert len(response.relatedSearches) == 2
    assert response.credits == 10


def test_search_response_invalid_search_type(
    valid_response_data: dict[str, Any],
) -> None:
    """
    Test the model_validator raises a ValueError for an invalid search type.
    """
    invalid_data = valid_response_data
    invalid_data["searchParameters"]["type"] = "invalid_type"
    with pytest.raises(ValidationError, match="Input should be a valid string"):
        SearchResponse(**invalid_data)


def test_search_response_invalid_num_0(valid_response_data: dict[str, Any]) -> None:
    """
    Test the model_validator raises a ValueError for an invalid number of results.
    """
    # Test `num` less than 1
    invalid_data_low_num = valid_response_data
    invalid_data_low_num["searchParameters"]["num"] = 0
    with pytest.raises(ValidationError, match="Input should be a valid integer"):
        SearchResponse(**invalid_data_low_num)


def test_search_response_invalid_num_11(valid_response_data: dict[str, Any]) -> None:
    """
    Test the model_validator raises a ValueError for an invalid number of results.
    """
    # Test `num` greater than 10
    invalid_data_high_num = valid_response_data
    invalid_data_high_num["searchParameters"]["num"] = 11
    with pytest.raises(ValidationError, match="Input should be a valid integer"):
        SearchResponse(**invalid_data_high_num)


def test_search_response_invalid_engine(valid_response_data: dict[str, Any]) -> None:
    """
    Test the model_validator raises a ValueError for an invalid search engine.
    """
    invalid_data = valid_response_data
    invalid_data["searchParameters"]["engine"] = "duckduckgo"
    with pytest.raises(ValidationError, match="Input should be a valid string"):
        SearchResponse(**invalid_data)


def test_search_response_invalid_credits(valid_response_data: dict[str, Any]) -> None:
    """
    Test the model_validator raises a ValueError for negative credits.
    """
    invalid_data = valid_response_data
    invalid_data["credits"] = -1
    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 0"
    ):
        SearchResponse(**invalid_data)


def test_search_response_missing_required_field(
    valid_response_data: dict[str, Any],
) -> None:
    """
    Test that Pydantic's built-in validation raises ValidationError for missing fields.
    """
    invalid_data = valid_response_data
    del invalid_data["organic"]
    with pytest.raises(ValidationError):
        SearchResponse(**invalid_data)


def test_organic_result_optional_field_with_sitelinks() -> None:
    """
    Test that the `sitelinks` field is correctly handled when it's `None` or missing.
    """
    data_with_sitelinks = {
        "title": "Example 1",
        "link": "http://example.com/1",
        "snippet": "Snippet 1",
        "position": 1,
        "sitelinks": [{"title": "Subpage", "link": "http://example.com/1/sub"}],
    }
    result_with_sitelinks = OrganicResult(**data_with_sitelinks)
    assert result_with_sitelinks.sitelinks is not None


def test_organic_result_optional_field_without_sitelinks() -> None:
    """
    Test that the `sitelinks` field is correctly handled when it's `None` or missing.
    """
    data_without_sitelinks = {
        "title": "Example 2",
        "link": "http://example.com/2",
        "snippet": "Snippet 2",
        "position": 2,
    }
    result_without_sitelinks = OrganicResult(**data_without_sitelinks)
    assert result_without_sitelinks.sitelinks == []
