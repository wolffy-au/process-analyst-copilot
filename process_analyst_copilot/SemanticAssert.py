import spacy


# Example test case for semantic similarity check
def semantic_assert(
    expected_output: str,
    actual_output: str,
    threshold: float = 0.9,
    verbose: bool = False,
) -> bool:
    # Load the spaCy model
    nlp = spacy.load("en_core_web_md")

    # Process the texts
    expected_embedding = nlp(expected_output)
    actual_embedding = nlp(actual_output)

    # Calculate the similarity
    similarity_score = expected_embedding.similarity(actual_embedding)

    if verbose:
        print(f"Similarity score: {similarity_score}")

    return similarity_score >= threshold
