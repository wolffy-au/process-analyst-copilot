import pytest
from pytest import MonkeyPatch
import spacy
import spacy.cli
from spacy.language import Language
from process_analyst_copilot.SemanticAssert import semantic_assert


@pytest.mark.parametrize(
    "expected, actual, expected_result",
    [
        (
            # 1.0 score
            "blue",
            "Blue.",
            True,
        ),
        (
            # tests/unit/test_SemanticAssert.py::test_semantic_similarity[blue--False]
            # SemanticAssert.py:63: UserWarning: [W008] Evaluating Doc.similarity based on empty vectors.
            # similarity_score: float = expected_embedding.similarity(actual_embedding)
            # division by 0 error
            "blue",
            "",
            False,
        ),
        (
            # 1.0 score
            "Step 1: Boil water. Step 2: Add tea leaves. Step 3: Pour water into cup.",
            "Step 1: Boil water. Step 2: Add tea leaves. Step 3: Pour water into cup.",
            True,
        ),
        (
            # 0.5697722177715019 score
            "Step 1: Boil water. Step 2: Add tea leaves. Step 3: Pour water into cup.",
            "Step 1: Heat the water. Step 2: Put tea leaves into the cup. Step 3: Pour the hot water.",
            True,
        ),
        (
            # 0.45684131124271454 score
            "Step 1: Boil water. Step 2: Add tea leaves. Step 3: Pour water into cup.",
            "Step 1: Heat the water. Step 2: Put coffee into the cup. Step 3: Pour the hot water.",
            False,
        ),
        (
            # -1.6159933112628946 score
            "Step 1: Boil water. Step 2: Add tea leaves. Step 3: Pour water into cup.",
            "This is a dog.",
            False,
        ),
        (
            # -0.11948626185654221 score
            "Step 1: Boil water. Step 2: Add tea leaves. Step 3: Pour water into cup.",
            "Step 1: Run around the park. Step 2: Do twenty situps. Step 3: Perform cool down stretches.",
            False,
        ),
    ],
)
def test_semantic_similarity(expected: str, actual: str, expected_result: bool) -> None:
    result: bool = semantic_assert(expected, actual, verbose=True)
    assert result == expected_result


def test_semantic_assert_model_loading_error(monkeypatch: MonkeyPatch) -> None:
    def mock_spacy_is_package(model_name: str) -> bool:
        return False

    def mock_spacy_load(model_name: str) -> Language:
        raise IOError("Mocked model loading error")

    def mock_spacy_download(model_name: str) -> None:
        pass

    monkeypatch.setattr(spacy.util, "is_package", mock_spacy_is_package)
    monkeypatch.setattr(spacy, "load", mock_spacy_load)
    monkeypatch.setattr(spacy.cli, "download", mock_spacy_download)

    with pytest.raises(IOError, match="Mocked model loading error"):
        semantic_assert("test", "test")


# if __name__ == "__main__":
#     #     # pytest.main()
#     print(
#         semantic_assert(
#             "blue",
#             "Blue.",
#             verbose=True,
#         )
#     )
# print(
#     semantic_assert(
#         "blue",
#         "",
#         verbose=True,
#     )
# )
#     print(
#         semantic_assert(
#             "Step 1: Boil water. Step 2: Add tea leaves. Step 3: Pour water into cup.",
#             "Step 1: Boil water. Step 2: Add tea leaves. Step 3: Pour water into cup.",
#             verbose=True,
#         )
#     )
#     print(
#         semantic_assert(
#             "Step 1: Boil water. Step 2: Add tea leaves. Step 3: Pour water into cup.",
#             "Step 1: Heat the water. Step 2: Put tea leaves into the cup. Step 3: Pour the hot water.",
#             verbose=True,
#         )
#     )
#     print(
#         semantic_assert(
#             "Step 1: Boil water. Step 2: Add tea leaves. Step 3: Pour water into cup.",
#             "Step 1: Heat the water. Step 2: Put coffee into the cup. Step 3: Pour the hot water.",
#             verbose=True,
#         )
#     )
#     print(
#         semantic_assert(
#             "Step 1: Boil water. Step 2: Add tea leaves. Step 3: Pour water into cup.",
#             "This is a dog.",
#             verbose=True,
#         )
#     )
#     print(
#         semantic_assert(
#             "Step 1: Boil water. Step 2: Add tea leaves. Step 3: Pour water into cup.",
#             "Step 1: Run around the park. Step 2: Do twenty situps. Step 3: Perform cool down stretches.",
#             verbose=True,
#         )
#     )
