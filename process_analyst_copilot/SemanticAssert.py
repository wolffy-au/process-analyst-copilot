"""
This module provides a function to assert the semantic similarity between two strings
using spaCy's language models.
"""

import math
import os
import spacy
from spacy.language import Language
import spacy.cli
from spacy.tokens.doc import Doc
import logging

# Configure logging
# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.WARNING),
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def semantic_assert(
    expected_output: str,
    actual_output: str,
    threshold: float = 0.5,
    verbose: bool = False,
) -> bool:
    """
    Compares the semantic similarity between two text strings using spaCy's 'en_core_web_md' model.
    Args:
        expected_output (str): The expected output text.
        actual_output (str): The actual output text.
        threshold (float, optional): The similarity threshold to determine if the texts are considered similar. Defaults to 0.5.
        verbose (bool, optional): If True, logs the similarity score. Defaults to False.
    Returns:
        bool: True if the similarity score is greater than or equal to the threshold, False otherwise.
    Raises:
        OSError: If there is an error loading the spaCy model.
    """
    # Load the spaCy model
    nlp: Language
    try:
        if not spacy.util.is_package("en_core_web_md"):
            logging.info(
                "The 'en_core_web_md' model is not installed. Attempting to download..."
            )
            spacy.cli.download("en_core_web_md")  # type: ignore
        nlp = spacy.load("en_core_web_md")
    except OSError as e:
        raise OSError(f"Error loading spaCy model after download attempt: {e}")

    # Process the texts
    def clean_text(text: str) -> str:
        return "".join(char if char.isalnum() else " " for char in text)

    expected_output = clean_text(expected_output)
    actual_output = clean_text(actual_output)
    expected_embedding: Doc = nlp(expected_output)
    actual_embedding: Doc = nlp(actual_output)

    # Calculate the similarity
    similarity_score: float = expected_embedding.similarity(actual_embedding)
    if similarity_score != 0.0:
        # Transform the similarity score to follow an exponential curve
        similarity_score = math.log(similarity_score**4) + 1

    if verbose:
        logging.info(f"Similarity score: {similarity_score}")

    return similarity_score >= threshold
