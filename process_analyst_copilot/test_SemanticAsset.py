import numpy as np
from process_analyst_copilot import OllamaLLM


# Function to get embeddings from OpenAI's API
def get_embeddings(text):
    response = openai.Embedding.create(
        input=text, model="text-embedding-ada-002"  # Use OpenAI's embedding model
    )
    return np.array(response["data"][0]["embedding"])


# Function to compute cosine similarity
def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )


# Example test case for semantic similarity check
def test_semantic_similarity(expected_output, actual_output):
    # Get embeddings for both expected and actual outputs
    expected_embedding = get_embeddings(expected_output)
    actual_embedding = get_embeddings(actual_output)

    # Calculate cosine similarity
    similarity_score = cosine_similarity(expected_embedding, actual_embedding)

    # Define a threshold for "semantic equivalence"
    threshold = 0.9

    print(f"Similarity score: {similarity_score}")
    if similarity_score >= threshold:
        print("The responses are semantically the same!")
    else:
        print("The responses are different.")


# Example usage:
expected = "Step 1: Boil water. Step 2: Add tea leaves. Step 3: Pour water into cup."
actual = "Step 1: Heat the water. Step 2: Put tea leaves into the cup. Step 3: Pour the hot water."

print(test_semantic_similarity(expected, actual))
