from dotenv import load_dotenv, find_dotenv
import litellm
import os

# litellm._turn_on_debug()

load_dotenv(find_dotenv())

# By default, litellm will connect to http://localhost:11434 for Ollama.
# If your Ollama server is running on a different host or port,
# you can set the OLLAMA_API_BASE environment variable.
# For example:
# os.environ["OLLAMA_API_BASE"] = "http://your-ollama-host:11434"

print(f"OLLAMA_API_BASE: {os.environ["OLLAMA_API_BASE"]}")


def chat_with_ollama() -> None:
    """
    Sends a request to an Ollama model using litellm and prints the response.
    """
    try:
        # 1. Set the model name.
        # The 'ollama/' prefix is crucial. It tells litellm to route this
        # request to the Ollama API.
        # Ensure 'llama3' is a model you have pulled with `ollama pull llama3`.
        model_name = "ollama/llama3.1:8b"

        # 2. Define the message payload in the standard OpenAI format.
        messages = [
            {
                "role": "user",
                "content": "Hi! Can you write a short, 3-line poem about coding?",
            }
        ]

        # 3. Call the completion function.
        print(f"Sending request to model: {model_name}...")
        response = litellm.completion(
            model=model_name,
            messages=messages,
            # Set a larger context window for the model.
            num_ctx=131072,
        )

        # 4. Print the response from the model.
        print("\n✅ Response from Ollama:")
        print(response.choices[0].message.content)

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")


if __name__ == "__main__":
    chat_with_ollama()
