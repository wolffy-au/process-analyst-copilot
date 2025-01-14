import pytest
from process_analyst_copilot.SemanticAssert import semantic_assert


@pytest.mark.parametrize(
    "expected, actual, expected_result",
    [
        (
            # 1.0 score
            "Step 1: Boil water. Step 2: Add tea leaves. Step 3: Pour water into cup.",
            "Step 1: Boil water. Step 2: Add tea leaves. Step 3: Pour water into cup.",
            True,
        ),
        (
            # 0.943 score
            "Step 1: Boil water. Step 2: Add tea leaves. Step 3: Pour water into cup.",
            "Step 1: Heat the water. Step 2: Put tea leaves into the cup. Step 3: Pour the hot water.",
            True,
        ),
        (
            # 0.647 score
            "Step 1: Boil water. Step 2: Add tea leaves. Step 3: Pour water into cup.",
            "This is a dog.",
            False,
        ),
        (
            # 0.867 score
            "Step 1: Boil water. Step 2: Add tea leaves. Step 3: Pour water into cup.",
            "Step 1: Run around the park. Step 2: Do twenty situps. Step 3: Perform cool down stretches.",
            False,
        ),
    ],
)
def test_semantic_similarity(expected: str, actual: str, expected_result: bool) -> None:
    result: bool = semantic_assert(expected, actual)
    assert result == expected_result


# if __name__ == "__main__":
#     # pytest.main()
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
