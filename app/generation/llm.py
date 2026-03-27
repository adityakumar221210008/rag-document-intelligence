import os

BACKEND = os.getenv("LLM_BACKEND", "openai")  # or "ollama"


def generate_answer(query: str, context: str, history: list[dict]) -> str:
    prompt = _build_prompt(query, context)

    if BACKEND == "openai":
        return _openai_generate(prompt, history)
    elif BACKEND == "ollama":
        return _ollama_generate(prompt, history)
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {BACKEND}")


def _build_prompt(query: str, context: str) -> str:
    return f"""You are a helpful assistant that answers questions based on provided document context.

CONTEXT:
{context}

INSTRUCTIONS:
- Answer using ONLY the information in the context above.
- If the answer is not in the context, say "I couldn't find this in the provided documents."
- Be concise and cite which part of the context supports your answer.

QUESTION: {query}

ANSWER:"""


def _openai_generate(prompt: str, history: list[dict]) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {"role": "system", "content": "You are a precise document Q&A assistant."},
        *history,
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def _ollama_generate(prompt: str, history: list[dict]) -> str:
    """Use a local Ollama model (e.g. llama3, mistral). Run: ollama pull llama3"""
    import requests
    model = os.getenv("OLLAMA_MODEL", "llama3")
    messages = [
        {"role": "system", "content": "You are a precise document Q&A assistant."},
        *history,
        {"role": "user", "content": prompt},
    ]
    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={"model": model, "messages": messages, "stream": False},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()
