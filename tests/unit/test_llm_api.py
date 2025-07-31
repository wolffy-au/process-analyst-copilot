from dotenv import load_dotenv
import os
from crewai import LLM

load_dotenv(override=True)


def test_llm_api() -> None:
    llm = LLM(model=os.getenv("MODEL", "gpt-4o-mini"))
    response = llm.call(messages="Hello, how are you?")
    print(f"LLM Response: {response}")
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
    print("LLM API test passed successfully.")


if __name__ == "__main__":
    test_llm_api()
