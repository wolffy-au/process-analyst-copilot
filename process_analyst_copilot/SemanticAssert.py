import math
import spacy
from spacy.language import Language
import spacy.cli
from spacy.tokens.doc import Doc


def semantic_assert(
    expected_output: str,
    actual_output: str,
    threshold: float = 0.5,
    verbose: bool = False,
) -> bool:
    # Load the spaCy model
    nlp: Language
    try:
        nlp = spacy.load("en_core_web_md")
    except OSError as e:
        # raise IOError(f"Error loading spaCy model: {e}")
        print(f"Whoops! {e}")
        print("Attempting to download the 'en_core_web_md' model...")
        spacy.cli.download("en_core_web_md")  # type: ignore
        nlp = spacy.load("en_core_web_md")

    # Process the texts
    expected_output = "".join(
        char if char.isalnum() else " " for char in expected_output
    )
    actual_output = "".join(char if char.isalnum() else " " for char in actual_output)
    expected_embedding: Doc = nlp(expected_output)
    actual_embedding: Doc = nlp(actual_output)

    # Calculate the similarity
    similarity_score: float = expected_embedding.similarity(actual_embedding)
    # Transform the similarity score to follow an exponential curve
    similarity_score = math.log(similarity_score**4) + 1

    if verbose:
        print(f"Similarity score: {similarity_score}")

    return similarity_score >= threshold
